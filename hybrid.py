"""
Hybrid detection system: matched filter for candidate generation (high
recall -- don't miss real peaks), small local CNN for refinement (per-
candidate confidence, location offset, gamma) -- instead of either the
whole-signal U-Net or the matched filter alone.

This directly answers "where does the CNN add value on top of physics"
rather than "can the CNN replace physics": the matched filter proposes
WHERE something might be; the CNN only has to answer, locally, "is this
real, and exactly where/what is it" -- a much easier, better-conditioned
problem than finding peaks in a 1168-sample record from scratch.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import find_peaks
from torch.utils.data import Dataset, DataLoader

import simulator as sim


# ---------------------------------------------------------------------------
# 1. Candidate generation (matched filter, tuned for HIGH RECALL)
# ---------------------------------------------------------------------------

def generate_candidates(y: np.ndarray, x_excitation: np.ndarray, fs: float,
                         height_threshold: float = 0.02, min_distance_samples: int = 5) -> list[int]:
    """
    Deliberately permissive: low height threshold, short minimum distance
    between peaks. The point is to not miss real peaks -- false positives
    here get filtered by the refinement CNN downstream, not by being
    conservative at this stage.

    height_threshold is relative to the SEARCH REGION's own max, not the
    whole envelope's max -- the whole envelope is dominated by the direct
    pulse's own huge autocorrelation peak near zero lag, which made the
    effective threshold far too strict for genuine (much smaller) reflection
    peaks later in the trace. This was caught by a recall check (44.8% before
    the fix) before committing to training the refinement network on top of it.
    """
    envelope, zero_lag_idx = sim._envelope_matched_filter(y, x_excitation)
    footprint = sim._pulse_footprint_samples(x_excitation)
    search_start = zero_lag_idx + footprint // 2  # allow candidates a bit into the dead zone too --
                                                     # recall matters more than precision here
    search_region = envelope[search_start:]
    if search_region.max() <= 0:
        return []
    peak_idx, _ = find_peaks(search_region, height=height_threshold * search_region.max(),
                              distance=min_distance_samples)
    candidates = [int(idx + search_start - zero_lag_idx) for idx in peak_idx]
    return candidates


# ---------------------------------------------------------------------------
# 2. Small local refinement network
# ---------------------------------------------------------------------------

class LocalRefinementNet(nn.Module):
    """
    Input: [B, 1, W] windowed trace segment centered on a candidate.
    Outputs: confidence logit (real peak vs noise), location offset
    (samples, relative to window center), gamma estimate.
    """

    def __init__(self, window: int = 128, base_ch: int = 8):
        super().__init__()
        self.window = window
        self.conv = nn.Sequential(
            nn.Conv1d(1, base_ch, 9, padding=4), nn.BatchNorm1d(base_ch), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch, base_ch * 2, 9, padding=4), nn.BatchNorm1d(base_ch * 2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(base_ch * 2, base_ch * 4, 9, padding=4), nn.BatchNorm1d(base_ch * 4), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Linear(base_ch * 4, 32), nn.ReLU())
        self.confidence_head = nn.Linear(32, 1)
        self.offset_head = nn.Linear(32, 1)
        self.gamma_head = nn.Linear(32, 1)

    def forward(self, x):
        feat = self.head(self.conv(x))
        return self.confidence_head(feat).squeeze(-1), self.offset_head(feat).squeeze(-1), self.gamma_head(feat).squeeze(-1)


def local_refinement_loss(confidence_logit, offset_pred, gamma_pred,
                           is_true_target, offset_target, gamma_target,
                           pos_weight: float = 1.0, offset_scale: float = 20.0):
    conf_loss = F.binary_cross_entropy_with_logits(
        confidence_logit, is_true_target, pos_weight=torch.tensor(pos_weight))

    mask = is_true_target  # only supervise offset/gamma where there IS a true peak
    offset_loss = ((offset_pred - offset_target / offset_scale).pow(2) * mask).sum() / mask.sum().clamp(min=1.0)
    gamma_eps = 0.05
    rel_gamma_err = (gamma_pred - gamma_target) / (gamma_target.abs() + gamma_eps)
    gamma_loss = (rel_gamma_err.pow(2) * mask).sum() / mask.sum().clamp(min=1.0)

    total = conf_loss + offset_loss + gamma_loss
    return total, {"conf": conf_loss.item(), "offset": offset_loss.item(), "gamma": gamma_loss.item()}


# ---------------------------------------------------------------------------
# 3. Windowed candidate dataset (built from an existing Bucket 3 split)
# ---------------------------------------------------------------------------

def _ground_truth_peaks_samples(label: dict, fs: float) -> list[dict]:
    v_p = label["signal"]["v_p"]
    peaks = []
    for f in label["faults"]:
        delay = 2 * f["location_m"] / v_p * fs
        peaks.append({"delay_samples": delay, "gamma": f["gamma"]})
    er = label["end_reflection"]
    delay = 2 * er["location_m"] / v_p * fs
    peaks.append({"delay_samples": delay, "gamma": er["gamma"]})
    return peaks


class CandidateWindowDataset(Dataset):
    """
    Builds (window, is_true, offset, gamma) examples by running the
    candidate generator on each trace in split_dir and matching candidates
    to ground truth. Negative examples (false candidates) are kept too --
    the confidence head needs both to learn anything.
    """

    def __init__(self, split_dir: str, excitation: np.ndarray, fs: float = 1e9,
                 window: int = 128, match_tolerance_samples: float = 15.0):
        import glob
        self.window = window
        self.excitation = excitation
        self.fs = fs
        self.examples = []  # list of (trace_id, candidate_idx, is_true, offset, gamma)
        self._trace_cache: dict[str, np.ndarray] = {}

        ids = sorted({os.path.basename(p).split("_")[1].split(".")[0]
                      for p in glob.glob(os.path.join(split_dir, "x_*.npy"))})
        self.split_dir = split_dir

        for sid in ids:
            x = np.load(os.path.join(split_dir, f"x_{sid}.npy")).astype(np.float32)
            with open(os.path.join(split_dir, f"y_{sid}.json")) as f:
                label = json.load(f)
            truth = _ground_truth_peaks_samples(label, fs)

            candidates = generate_candidates(x, excitation, fs)
            truth_remaining = list(truth)
            for cand in candidates:
                if truth_remaining:
                    dists = [abs(cand - t["delay_samples"]) for t in truth_remaining]
                    best = int(np.argmin(dists))
                    if dists[best] <= match_tolerance_samples:
                        matched = truth_remaining.pop(best)
                        self.examples.append((sid, cand, 1.0, matched["delay_samples"] - cand, matched["gamma"]))
                        continue
                self.examples.append((sid, cand, 0.0, 0.0, 0.0))

    def __len__(self):
        return len(self.examples)

    def _get_trace(self, sid: str) -> np.ndarray:
        if sid not in self._trace_cache:
            self._trace_cache[sid] = np.load(os.path.join(self.split_dir, f"x_{sid}.npy")).astype(np.float32)
        return self._trace_cache[sid]

    def __getitem__(self, idx):
        sid, cand, is_true, offset, gamma = self.examples[idx]
        trace = self._get_trace(sid)
        n = len(trace)
        w = self.window
        lo, hi = cand - w // 2, cand + w // 2
        window = np.zeros(w, dtype=np.float32)
        src_lo, src_hi = max(lo, 0), min(hi, n)
        dst_lo, dst_hi = src_lo - lo, src_hi - lo
        window[dst_lo:dst_hi] = trace[src_lo:src_hi]

        return {
            "window": torch.from_numpy(window).unsqueeze(0),
            "is_true": torch.tensor(is_true, dtype=torch.float32),
            "offset": torch.tensor(offset, dtype=torch.float32),
            "gamma": torch.tensor(gamma, dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# 4. Training
# ---------------------------------------------------------------------------

def train_local_refinement(dataset_dir: str, excitation: np.ndarray, epochs: int = 10,
                            batch_size: int = 64, lr: float = 1e-3, window: int = 128,
                            checkpoint_path: str | None = None, resume: bool = False,
                            device: str = "cpu"):
    train_ds = CandidateWindowDataset(os.path.join(dataset_dir, "train"), excitation, window=window)
    val_ds = CandidateWindowDataset(os.path.join(dataset_dir, "val"), excitation, window=window)
    print(f"train examples: {len(train_ds)} ({sum(e[2] for e in train_ds.examples):.0f} positive), "
          f"val examples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    n_pos = sum(e[2] for e in train_ds.examples)
    n_neg = len(train_ds) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    model = LocalRefinementNet(window=window).to(device)
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"resumed from {checkpoint_path}")
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            conf, off, gam = model(batch["window"].to(device))
            loss, _ = local_refinement_loss(conf, off, gam, batch["is_true"].to(device),
                                             batch["offset"].to(device), batch["gamma"].to(device),
                                             pos_weight=pos_weight)
            optim.zero_grad(); loss.backward(); optim.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses, correct, total = [], 0, 0
        with torch.no_grad():
            for batch in val_loader:
                conf, off, gam = model(batch["window"].to(device))
                loss, _ = local_refinement_loss(conf, off, gam, batch["is_true"].to(device),
                                                 batch["offset"].to(device), batch["gamma"].to(device),
                                                 pos_weight=pos_weight)
                val_losses.append(loss.item())
                pred = (torch.sigmoid(conf) > 0.5).float()
                correct += (pred == batch["is_true"].to(device)).sum().item()
                total += pred.shape[0]

        print(f"epoch {epoch}: train_loss={np.mean(train_losses):.4f} "
              f"val_loss={np.mean(val_losses):.4f} val_conf_acc={correct/total:.3f}")

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
# 5. Hybrid inference: candidates -> refinement -> final detections
# ---------------------------------------------------------------------------

def hybrid_detect(trace: np.ndarray, excitation: np.ndarray, model: LocalRefinementNet,
                   fs: float, v_p: float, confidence_threshold: float = 0.5,
                   window: int = 128, device: str = "cpu") -> list[dict]:
    candidates = generate_candidates(trace, excitation, fs)
    if not candidates:
        return []

    n = len(trace)
    windows = []
    for cand in candidates:
        lo, hi = cand - window // 2, cand + window // 2
        w = np.zeros(window, dtype=np.float32)
        src_lo, src_hi = max(lo, 0), min(hi, n)
        dst_lo, dst_hi = src_lo - lo, src_hi - lo
        w[dst_lo:dst_hi] = trace[src_lo:src_hi]
        windows.append(w)

    batch = torch.from_numpy(np.stack(windows)).unsqueeze(1).to(device)  # [N, 1, W]
    model.eval()
    with torch.no_grad():
        conf, off, gam = model(batch)
    conf_prob = torch.sigmoid(conf).cpu().numpy()
    off = off.cpu().numpy() * 20.0  # matches offset_scale in local_refinement_loss
    gam = gam.cpu().numpy()

    detections = []
    for cand, c, o, g in zip(candidates, conf_prob, off, gam):
        if c >= confidence_threshold:
            refined_delay = cand + o
            location_m = (refined_delay / fs) * v_p / 2
            detections.append({"location_m": float(location_m), "gamma": float(g), "confidence": float(c)})
    return detections
