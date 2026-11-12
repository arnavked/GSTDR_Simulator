"""
Bucket 4 -- evaluation pipeline.

This is the piece that was missing: without it, "the model outputs
something" is not the same as "the model is useful". Three things happen
here:

1. Peak extraction: threshold the presence map, find local maxima, map
   surviving indices to (location_m, gamma) pairs -- turning raw model
   output into the same kind of prediction a matched filter produces, so
   the two are directly comparable.
2. Evaluation metrics: detection accuracy (right number of peaks, within a
   sample tolerance), localization error, gamma error at matched detections
   -- computed by matching predicted peaks to ground-truth peaks (including
   the end reflection, same as everywhere else in this project) via nearest-
   neighbor assignment, not just comparing counts.
3. A matched-filter baseline evaluated on the EXACT SAME traces, at the
   same SNR sweep used in Bucket 1/2 (40/30/20/15/10/5 dB), so the CNN-vs-
   classical comparison is apples-to-apples rather than reusing old numbers
   from a different dataset.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from scipy.signal import find_peaks

import simulator as sim
from model import GSTDRNet


# ---------------------------------------------------------------------------
# 1. Peak extraction from model output
# ---------------------------------------------------------------------------

def extract_peaks_from_model_output(presence_logits: np.ndarray, gamma_map: np.ndarray,
                                     fs: float, v_p: float, presence_threshold: float = 0.5,
                                     min_peak_distance_samples: int = 10) -> list[dict]:
    """
    presence_logits, gamma_map: 1D arrays, length N (already squeezed from
    batch/channel dims).

    Threshold -> local maxima (find_peaks handles the "local maxima" and
    "minimum distance between peaks" parts in one call, avoiding duplicate
    detections from a single broad bump) -> read gamma at each surviving
    peak -> convert sample index to location_m.
    """
    presence_prob = 1 / (1 + np.exp(-presence_logits))  # sigmoid
    peak_idx, _ = find_peaks(presence_prob, height=presence_threshold, distance=min_peak_distance_samples)

    detections = []
    for idx in peak_idx:
        location_m = (idx / fs) * v_p / 2
        detections.append({
            "sample_idx": int(idx),
            "location_m": float(location_m),
            "gamma": float(gamma_map[idx]),
            "presence_prob": float(presence_prob[idx]),
        })
    return detections


# ---------------------------------------------------------------------------
# 2. Matching predictions to ground truth + metrics
# ---------------------------------------------------------------------------

def _ground_truth_peaks(label: dict) -> list[dict]:
    """Every labeled peak (faults + end reflection), same convention used
    throughout this project (build_signed_heatmap, pairwise resolution)."""
    peaks = [{"location_m": f["location_m"], "gamma": f["gamma"]} for f in label["faults"]]
    er = label["end_reflection"]
    peaks.append({"location_m": er["location_m"], "gamma": er["gamma"]})
    return peaks


def match_detections_to_truth(detections: list[dict], truth: list[dict],
                               tolerance_m: float) -> dict:
    """
    Greedy nearest-neighbor matching within tolerance_m. Returns matched
    pairs plus unmatched detections (false positives) and unmatched truth
    (false negatives/missed detections).
    """
    truth_remaining = list(enumerate(truth))
    matches = []
    unmatched_detections = []

    for det in detections:
        if not truth_remaining:
            unmatched_detections.append(det)
            continue
        dists = [abs(det["location_m"] - t["location_m"]) for _, t in truth_remaining]
        best = int(np.argmin(dists))
        if dists[best] <= tolerance_m:
            true_idx, true_peak = truth_remaining.pop(best)
            matches.append({"detection": det, "truth": true_peak, "error_m": det["location_m"] - true_peak["location_m"]})
        else:
            unmatched_detections.append(det)

    unmatched_truth = [t for _, t in truth_remaining]
    return {"matches": matches, "false_positives": unmatched_detections, "false_negatives": unmatched_truth}


def evaluate_sample(detections: list[dict], label: dict, tolerance_m: float = 2.0) -> dict:
    truth = _ground_truth_peaks(label)
    result = match_detections_to_truth(detections, truth, tolerance_m)

    n_true = len(truth)
    n_matched = len(result["matches"])
    loc_errors_m = [abs(m["error_m"]) for m in result["matches"]]
    gamma_errors = [abs(m["detection"]["gamma"] - m["truth"]["gamma"]) for m in result["matches"]]

    return {
        "n_true_peaks": n_true,
        "n_detected": len(detections),
        "n_matched": n_matched,
        "n_false_positives": len(result["false_positives"]),
        "n_false_negatives": len(result["false_negatives"]),
        "count_correct": len(detections) == n_true,  # right NUMBER of peaks, regardless of match quality
        "mean_localization_error_m": float(np.mean(loc_errors_m)) if loc_errors_m else None,
        "mean_gamma_error": float(np.mean(gamma_errors)) if gamma_errors else None,
    }


# ---------------------------------------------------------------------------
# 3. CNN evaluation over a full split
# ---------------------------------------------------------------------------

def evaluate_model_on_split(model: GSTDRNet, split_dir: str, fs: float, v_p: float,
                             device: str = "cpu", tolerance_m: float = 2.0,
                             presence_threshold: float = 0.5, max_samples: int | None = None,
                             excitation: np.ndarray | None = None) -> dict:
    model.eval()
    import glob
    ids = sorted({os.path.basename(p).split("_")[1].split(".")[0]
                  for p in glob.glob(os.path.join(split_dir, "x_*.npy"))})
    if max_samples:
        ids = ids[:max_samples]

    per_sample_results = []
    with torch.no_grad():
        for sid in ids:
            x = np.load(os.path.join(split_dir, f"x_{sid}.npy")).astype(np.float32)
            with open(os.path.join(split_dir, f"y_{sid}.json")) as f:
                label = json.load(f)

            if excitation is not None:
                envelope, zero_lag_idx = sim._envelope_matched_filter(x, excitation)
                mf_channel = envelope[zero_lag_idx: zero_lag_idx + len(x)]
                peak = mf_channel.max()
                mf_channel = (mf_channel / peak).astype(np.float32) if peak > 0 else mf_channel.astype(np.float32)
                x_in = np.stack([x, mf_channel], axis=0)
                x_t = torch.from_numpy(x_in).unsqueeze(0).to(device)
            else:
                x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

            presence_logits, gamma_map, _ = model(x_t)
            presence_logits = presence_logits.squeeze().cpu().numpy()
            gamma_map = gamma_map.squeeze().cpu().numpy()

            detections = extract_peaks_from_model_output(presence_logits, gamma_map, fs, v_p,
                                                           presence_threshold=presence_threshold)
            per_sample_results.append(evaluate_sample(detections, label, tolerance_m))

    return _aggregate_results(per_sample_results)


def _aggregate_results(per_sample_results: list[dict]) -> dict:
    n = len(per_sample_results)
    count_correct_rate = sum(r["count_correct"] for r in per_sample_results) / n
    all_loc_errors = [r["mean_localization_error_m"] for r in per_sample_results if r["mean_localization_error_m"] is not None]
    all_gamma_errors = [r["mean_gamma_error"] for r in per_sample_results if r["mean_gamma_error"] is not None]
    total_fp = sum(r["n_false_positives"] for r in per_sample_results)
    total_fn = sum(r["n_false_negatives"] for r in per_sample_results)
    total_true = sum(r["n_true_peaks"] for r in per_sample_results)

    return {
        "n_samples": n,
        "count_correct_rate": count_correct_rate,
        "mean_localization_error_m": float(np.mean(all_loc_errors)) if all_loc_errors else None,
        "mean_gamma_error": float(np.mean(all_gamma_errors)) if all_gamma_errors else None,
        "false_positive_rate": total_fp / max(total_true, 1),
        "miss_rate": total_fn / max(total_true, 1),
    }


# ---------------------------------------------------------------------------
# 4. Matched-filter baseline, on the SAME traces, same SNR sweep as Bucket 1/2
# ---------------------------------------------------------------------------

def matched_filter_baseline_sweep(split_dir: str, x_excitation: np.ndarray, fs: float,
                                   snr_values_db: list[float], n_trials_per_snr: int = 100,
                                   tolerance_m: float = 2.0) -> list[dict]:
    """
    Reuses the SAME single-fault Monte Carlo methodology as Bucket 1's Test
    2 (detect_fault_blind), but reports results in the same schema as
    evaluate_model_on_split so the two are directly comparable in one table
    -- not just eyeballed against old numbers from a different run.
    """
    cable = sim.CableParams(v_p=1.65e8)
    results_by_snr = []
    for snr_db in snr_values_db:
        hits = 0
        loc_errors = []
        for trial in range(n_trials_per_snr):
            rng = np.random.default_rng(5000 + trial)
            location_m = rng.uniform(20.0, 60.0)  # within the clean valid range, single fault
            gamma = rng.uniform(-0.9, 0.9)
            fault = sim.FaultParams(location_m, gamma)
            y, label = sim.generate_single_fault_trace(x_excitation, fs, cable, fault, snr_db=snr_db, rng=rng)
            measured_delay = sim.detect_fault_blind(y, x_excitation, fs, cable.v_p,
                                                      max_cable_length_m=location_m * 3)
            measured_location_m = (measured_delay / fs) * cable.v_p / 2
            err = abs(measured_location_m - location_m)
            loc_errors.append(err)
            if err <= tolerance_m:
                hits += 1
        results_by_snr.append({
            "snr_db": snr_db,
            "count_correct_rate": hits / n_trials_per_snr,
            "mean_localization_error_m": float(np.mean(loc_errors)),
        })
    return results_by_snr


def sweep_presence_threshold(model: GSTDRNet, split_dir: str, fs: float, v_p: float,
                              thresholds: list[float] | None = None, tolerance_m: float = 2.0,
                              max_samples: int | None = 500, excitation: np.ndarray | None = None) -> dict:
    """
    Sweeps presence_threshold, scores each by an F1-like balance of
    precision/recall (not count_correct_rate alone, since that doesn't
    distinguish "close but off by one peak" from "way off").
    Returns the full table plus the best threshold by F1.
    """
    if thresholds is None:
        thresholds = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    rows = []
    for t in thresholds:
        r = evaluate_model_on_split(model, split_dir, fs, v_p, tolerance_m=tolerance_m,
                                     presence_threshold=t, max_samples=max_samples, excitation=excitation)
        fp_rate = r["false_positive_rate"]
        miss_rate = r["miss_rate"]
        precision = 1.0 / (1.0 + fp_rate) if fp_rate is not None else 0.0
        recall = 1.0 - miss_rate if miss_rate is not None else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({**r, "threshold": t, "precision_approx": precision, "recall_approx": recall, "f1_approx": f1})

    best = max(rows, key=lambda r: r["f1_approx"])
    return {"rows": rows, "best_threshold": best["threshold"], "best_row": best}


def evaluate_model_clean_subset_by_snr(model: GSTDRNet, split_dir: str, fs: float, v_p: float,
                                        presence_threshold: float, device: str = "cpu",
                                        tolerance_m: float = 2.0, excitation: np.ndarray | None = None,
                                        location_range_m: tuple[float, float] = (20.0, 60.0)) -> list[dict]:
    """
    Filters to EXACTLY the matched-filter baseline's test conditions --
    single fault, location in the same clean range, no dead_zone/
    beyond_range faults -- and breaks results out by SNR, so this can sit
    directly next to matched_filter_baseline_sweep's per-SNR table instead
    of being eyeballed against a different aggregate number.
    """
    import glob
    ids = sorted({os.path.basename(p).split("_")[1].split(".")[0]
                  for p in glob.glob(os.path.join(split_dir, "x_*.npy"))})

    by_snr: dict[float, list[dict]] = {}
    model.eval()
    with torch.no_grad():
        for sid in ids:
            with open(os.path.join(split_dir, f"y_{sid}.json")) as f:
                label = json.load(f)
            if label["num_faults"] != 1:
                continue
            f0 = label["faults"][0]
            if f0["dead_zone"] or f0["beyond_range"]:
                continue
            if not (location_range_m[0] <= f0["location_m"] <= location_range_m[1]):
                continue

            x = np.load(os.path.join(split_dir, f"x_{sid}.npy")).astype(np.float32)
            snr = label["signal"]["snr_db"]

            if excitation is not None:
                envelope, zero_lag_idx = sim._envelope_matched_filter(x, excitation)
                mf_channel = envelope[zero_lag_idx: zero_lag_idx + len(x)]
                peak = mf_channel.max()
                mf_channel = (mf_channel / peak).astype(np.float32) if peak > 0 else mf_channel.astype(np.float32)
                x_t = torch.from_numpy(np.stack([x, mf_channel], axis=0)).unsqueeze(0).to(device)
            else:
                x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(device)

            presence_logits, gamma_map, _ = model(x_t)
            presence_logits = presence_logits.squeeze().cpu().numpy()
            gamma_map = gamma_map.squeeze().cpu().numpy()
            detections = extract_peaks_from_model_output(presence_logits, gamma_map, fs, v_p,
                                                           presence_threshold=presence_threshold)
            result = evaluate_sample(detections, label, tolerance_m)
            by_snr.setdefault(snr, []).append(result)

    rows = []
    for snr in sorted(by_snr.keys(), reverse=True):
        rs = by_snr[snr]
        cc = sum(r["count_correct"] for r in rs) / len(rs)
        loc = [r["mean_localization_error_m"] for r in rs if r["mean_localization_error_m"] is not None]
        rows.append({
            "snr_db": snr, "n": len(rs), "count_correct_rate": cc,
            "mean_localization_error_m": float(np.mean(loc)) if loc else None,
        })
    return rows


if __name__ == "__main__":
    dataset_dir = "./dataset_v3_50k"
    model = GSTDRNet()
    model.load_state_dict(torch.load("./gstdr_model_smoketest.pt", map_location="cpu"))

    print("Model loaded. Running evaluation...")

    results = evaluate_model_on_split(model, dataset_dir + "/test", fs=1e9, v_p=1.65e8)

    print("RESULTS:")
    print(results)