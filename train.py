"""
Bucket 4 -- dataset loader + training loop.

Reads directly from a Bucket 3 output directory (train/val/test subfolders
of x_*.npy, heatmap_*.npy, y_*.json). Derives the presence_target (binary,
for the detection head) from the object-list label's peak locations rather
than by thresholding the signed heatmap -- thresholding would itself
under-detect small-gamma peaks, which is exactly the problem this two-head
design exists to avoid.

Modal usage
-----------
This file can still be run locally exactly as before:
    python train.py ./dataset_v3_50k

To run the same training on a cloud GPU via Modal:
    1) One-time: upload your dataset into a Modal Volume so you don't
       re-upload 50k samples on every run.
           modal run train.py::upload_dataset --local-dir ./dataset_v3_50k
    2) Launch training on a GPU:
           modal run train.py --epochs 5 --batch-size 16 --gpu-type T4

    (Windows note: if `modal` isn't on PATH, use `python -m modal` instead
    of `modal` in the commands above.)
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from model import GSTDRNet, gstdr_loss
import simulator as sim

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
# Current Modal API (1.0+): local files/dirs are attached to the Image via
# .add_local_dir / .add_local_file / .add_local_python_source, NOT via the
# old modal.Mount class (removed) or the mounts= kwarg on @app.function.
#
# model.py and simulator.py are plain local modules (this project isn't a
# package), so they won't be auto-included -- they're added explicitly below.

app = modal.App("gstdr-training-job")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "numpy")
    # Excitation waveform files train() reads by default -- small text
    # files, safe to bake into the image. copy=True makes this an actual
    # build step, so it has to happen BEFORE add_local_python_source below
    # (which uses copy=False -- files synced in at container startup, not
    # built into a layer -- and Modal requires all copy=False add_local_*
    # calls to be the very last thing in the chain).
    .add_local_file("25MHz1V (1).txt", "/root/25MHz1V (1).txt", copy=True)
    .add_local_file("25MHz1V_asciiheader (1).txt", "/root/25MHz1V_asciiheader (1).txt", copy=True)
    # copy=False (default) -- must be last: synced at container startup,
    # not baked into an image layer, so PyTorch etc. above stay cached
    # across runs even when model.py/simulator.py change.
    .add_local_python_source("model", "simulator")
)

# Persistent storage for the dataset + checkpoints so you upload once and
# reuse across runs instead of re-syncing 50k samples every launch.
dataset_volume = modal.Volume.from_name("gstdr-dataset-v3-50k", create_if_missing=True)
DATA_ROOT = "/data"


class GSTDRDataset(Dataset):
    def __init__(self, split_dir: str, n_samples_record: int = 1168,
                 fs: float = 1e9, presence_sigma_samples: float = 2.5,
                 excitation: np.ndarray | None = None):
        self.split_dir = split_dir
        self.ids = sorted({
            os.path.basename(p).split("_")[1].split(".")[0]
            for p in glob.glob(os.path.join(split_dir, "x_*.npy"))
        })
        if not self.ids:
            raise ValueError(f"No samples found in {split_dir} -- did you generate the dataset first?")
        self.n = n_samples_record
        self.fs = fs
        self.presence_sigma = presence_sigma_samples
        self._t = np.arange(self.n)
        self.excitation = excitation  # if set, adds a matched-filter channel

    def _matched_filter_channel(self, y: np.ndarray) -> np.ndarray:
        """
        Envelope of the correlation between y and the excitation template,
        aligned to trace samples (same zero-lag convention used everywhere
        else in this project -- see simulator._envelope_matched_filter).
        Per-sample max-normalized so it's on a comparable scale to the raw
        trace regardless of the excitation's own energy.
        """
        envelope, zero_lag_idx = sim._envelope_matched_filter(y, self.excitation)
        channel = envelope[zero_lag_idx: zero_lag_idx + self.n]
        peak = channel.max()
        return (channel / peak).astype(np.float32) if peak > 0 else channel.astype(np.float32)

    def __len__(self):
        return len(self.ids)

    def _build_presence_target(self, label: dict) -> np.ndarray:
        """Binary-ish (Gaussian, unit height) presence target from the
        object-list label's peak locations -- independent of gamma
        magnitude, unlike the signed heatmap."""
        presence = np.zeros(self.n, dtype=np.float32)
        for f in label["faults"]:
            center = 2 * f["location_m"] / label["signal"]["v_p"] * self.fs
            presence += np.exp(-((self._t - center) ** 2) / (2 * self.presence_sigma ** 2))
        er = label["end_reflection"]
        center = 2 * er["location_m"] / label["signal"]["v_p"] * self.fs
        presence += np.exp(-((self._t - center) ** 2) / (2 * self.presence_sigma ** 2))
        return np.clip(presence, 0, 1)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        x = np.load(os.path.join(self.split_dir, f"x_{sid}.npy")).astype(np.float32)
        heatmap = np.load(os.path.join(self.split_dir, f"heatmap_{sid}.npy")).astype(np.float32)
        with open(os.path.join(self.split_dir, f"y_{sid}.json")) as f:
            label = json.load(f)

        presence = self._build_presence_target(label)
        num_faults = min(label["num_faults"], 4)  # classifier head caps at 4, matching FAULT_COUNT_WEIGHTS

        if self.excitation is not None:
            mf_channel = self._matched_filter_channel(x)
            x_tensor = torch.from_numpy(np.stack([x, mf_channel], axis=0))  # [2, N]
        else:
            x_tensor = torch.from_numpy(x).unsqueeze(0)  # [1, N]

        return {
            "x": x_tensor,
            "gamma_target": torch.from_numpy(heatmap).unsqueeze(0),  # [1, N]
            "presence_target": torch.from_numpy(presence).unsqueeze(0),  # [1, N]
            "num_faults": torch.tensor(num_faults, dtype=torch.long),
        }


def train(dataset_dir: str, epochs: int = 5, batch_size: int = 16,
          lr: float = 1e-3, device: str = "cpu", checkpoint_path: str | None = None,
          resume: bool = False, use_matched_filter_channel: bool = False,
          excitation_data_path: str = "25MHz1V (1).txt",
          excitation_header_path: str = "25MHz1V_asciiheader (1).txt",
          num_workers: int = 0, log_every: int = 50):
    excitation = None
    if use_matched_filter_channel:
        excitation, _ = sim.load_excitation(excitation_data_path, excitation_header_path)

    train_ds = GSTDRDataset(os.path.join(dataset_dir, "train"), excitation=excitation)
    val_ds = GSTDRDataset(os.path.join(dataset_dir, "val"), excitation=excitation)
    # pin_memory only helps when transfers below use non_blocking=True, and
    # persistent_workers avoids re-forking worker processes every epoch
    # (worth it here since epochs are short relative to worker startup cost).
    loader_kwargs = dict(num_workers=num_workers, pin_memory=(device == "cuda"))
    if num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **loader_kwargs)

    in_channels = 2 if use_matched_filter_channel else 1
    model = GSTDRNet(in_channels=in_channels).to(device)
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"resumed from {checkpoint_path}")
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # AMP (mixed precision): meaningful speedup on Ampere+ GPUs (A10G, A100)
    # once the data pipeline isn't the bottleneck. No-op on CPU.
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    history = []
    for epoch in range(epochs):
        model.train()
        train_losses = []
        n_batches = len(train_loader)
        for i, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            gamma_target = batch["gamma_target"].to(device, non_blocking=True)
            presence_target = batch["presence_target"].to(device, non_blocking=True)
            num_faults = batch["num_faults"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                presence_logits, gamma_map, num_faults_logits = model(x)
                loss, parts = gstdr_loss(presence_logits, gamma_map, num_faults_logits,
                                          presence_target, gamma_target, num_faults)

            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            train_losses.append(loss.item())

            if log_every and (i + 1) % log_every == 0:
                print(f"  epoch {epoch} batch {i + 1}/{n_batches} loss={loss.item():.4f}", flush=True)

        model.eval()
        val_losses = []
        val_class_correct = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                gamma_target = batch["gamma_target"].to(device, non_blocking=True)
                presence_target = batch["presence_target"].to(device, non_blocking=True)
                num_faults = batch["num_faults"].to(device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                    presence_logits, gamma_map, num_faults_logits = model(x)
                    loss, parts = gstdr_loss(presence_logits, gamma_map, num_faults_logits,
                                              presence_target, gamma_target, num_faults)
                val_losses.append(loss.item())
                val_class_correct += (num_faults_logits.argmax(dim=1) == num_faults).sum().item()
                val_n += num_faults.shape[0]

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_acc = val_class_correct / val_n
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_class_acc": val_acc})
        print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_num_faults_acc={val_acc:.3f}", flush=True)

    if checkpoint_path:
        torch.save(model.state_dict(), checkpoint_path)
        print(f"saved checkpoint to {checkpoint_path}")

    return model, history


# ---------------------------------------------------------------------------
# Modal entrypoints
# ---------------------------------------------------------------------------

@app.function(volumes={DATA_ROOT: dataset_volume}, timeout=3600)
def upload_dataset_worker(local_dir: str):
    """Runs in the cloud; the actual byte transfer happens via the Volume
    batch_upload API driven from the local_entrypoint below."""
    dataset_volume.reload()
    print("volume contents:", os.listdir(DATA_ROOT))


@app.local_entrypoint()
def upload_dataset(local_dir: str = "./dataset_v3_50k"):
    """One-time (or whenever the dataset changes) upload into the Volume.
    Run with: modal run train.py::upload_dataset --local-dir ./dataset_v3_50k
    """
    print(f"uploading {local_dir} -> volume 'gstdr-dataset-v3-50k' ...")
    with dataset_volume.batch_upload(force=True) as batch:
        batch.put_directory(local_dir, "/")
    print("done.")


LOCAL_DATA_ROOT = "/local_data"  # container-local disk, not the network Volume


@app.function(
    image=image,
    gpu="T4",       # overridden per-call from main() below via with_options
    cpu=8,           # must be >= num_workers passed to train(), or workers
                     # have no real cores to run on and num_workers does ~nothing
    timeout=14400,  # 4 hour cap
    volumes={DATA_ROOT: dataset_volume},
)
def run_remote_training(epochs: int = 5, batch_size: int = 16, lr: float = 1e-3,
                         use_matched_filter_channel: bool = False,
                         checkpoint_name: str = "gstdr_model_smoketest.pt",
                         num_workers: int = 8):
    import shutil
    import time

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"training on cloud device: {device}")

    # Copy train/val off the network-mounted Volume onto the container's
    # local disk once, up front. One bulk copy is far cheaper than the
    # dataset's individual files being opened one-by-one over the network
    # on every __getitem__ call during training.
    t0 = time.time()
    # unzip.py extracted the zip's internal top-level folder as-is, so the
    # actual layout is /data/dataset_v3_50k/{train,val,test}, not /data/{train,val,test}.
    EXTRACTED_SUBDIR = "dataset_v3_50k"
    for split in ("train", "val"):
        src = os.path.join(DATA_ROOT, EXTRACTED_SUBDIR, split)
        dst = os.path.join(LOCAL_DATA_ROOT, split)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"{src} not found on the Volume -- did you run upload_dataset?")
        shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f"copied dataset to local disk in {time.time() - t0:.1f}s", flush=True)

    checkpoint_path = os.path.join(DATA_ROOT, checkpoint_name)  # checkpoint still goes to the Volume
    model_, history = train(
        dataset_dir=LOCAL_DATA_ROOT,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        checkpoint_path=checkpoint_path,
        use_matched_filter_channel=use_matched_filter_channel,
        num_workers=num_workers,
    )
    dataset_volume.commit()  # persist the checkpoint back to the Volume
    return history


@app.local_entrypoint()
def main(epochs: int = 5, batch_size: int = 16, lr: float = 1e-3,
         use_matched_filter_channel: bool = False,
         checkpoint_name: str = "gstdr_model_smoketest.pt",
         gpu_type: str = "T4",
         num_workers: int = 8):
    """
    modal run train.py::main --epochs 5 --batch-size 16 --gpu-type T4
    (Windows: python -m modal run train.py::main --epochs 5 ...)
    """
    print("packaging and sending training job to Modal...")
    # gpu_type used to be accepted here but silently ignored -- the
    # decorator's gpu="T4" was fixed regardless of what you passed.
    # with_options() overrides it per-call.
    history = run_remote_training.with_options(gpu=gpu_type).remote(
        epochs=epochs, batch_size=batch_size, lr=lr,
        use_matched_filter_channel=use_matched_filter_channel,
        checkpoint_name=checkpoint_name,
        num_workers=num_workers,
    )
    print("training complete. history:")
    for row in history:
        print(row)
    print(
        f"\ncheckpoint saved in the Volume as '{checkpoint_name}'. "
        f"Download it with:\n"
        f"  modal volume get gstdr-dataset-v3-50k {checkpoint_name} ."
    )


if __name__ == "__main__":
    # Local (non-Modal) usage is unchanged.
    import sys
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./dataset_v3_50k"
    train(dataset_dir, epochs=5, batch_size=16, checkpoint_path="./gstdr_model_smoketest.pt")