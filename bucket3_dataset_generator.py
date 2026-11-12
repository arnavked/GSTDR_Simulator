"""
Bucket 3 — dataset generator.

Builds on the Bucket 1+2 simulator (simulator.py) to produce a labeled
dataset of GSTDR traces, using a CONTROLLED sampling strategy (not blind
randomization) over fault count, location, separation regime, gamma
category, and SNR.

Label schema (see build_label() for the authoritative version):
    - object-based: explicit list of faults (location, gamma, dead_zone/
      beyond_range flags), cable-end info, SNR, v_p
    - pairwise: min separation across all fault pairs + which specific
      pairs fall below the resolution limit (NOT a single collapsed bool --
      that would silently discard which pairs are actually the problem)
    - dense target: a SIGNED heatmap (Gaussian bumps scaled by gamma, not
      unit amplitude) over the record, so polarity survives in the dense
      representation too, not just the object list

Deliberately excluded from this module (left for Bucket 4, training time):
    - any notion of sample weighting for hard/below-resolution cases --
      the dataset records FACTS (pairs_below_resolution), the training loop
      decides what to do with them. Baking a weighting policy into the
      dataset would mean regenerating data every time that policy changes.
    - heatmap amplitude imbalance (open/short peaks ~1.0, soft-fault peaks
      ~0.02-0.1) is physically correct and intentionally NOT rescaled here.
      If small-gamma peaks vanish in a naive MSE loss, address that with a
      weighted loss or log-scaled target at training time, not by distorting
      the physics in the generator.
    - heatmap sigma is a fixed 2.5 samples for all peaks. Real pulse width
      broadens somewhat with distance/attenuation (see the amplitude-vs-
      distance validation in Bucket 1), so sigma could eventually scale with
      distance too -- not urgent, noted for later.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict

import numpy as np

import simulator as sim


# ---------------------------------------------------------------------------
# 1. Gamma sampling -- physical fault-type mixture, not a generic distribution
# ---------------------------------------------------------------------------

# Category probabilities and gamma sub-distributions, anchored to the
# Table-1-style definitions from the papers reviewed earlier in this
# project (short ~ -1, open ~ +1, soft fault = Z_c +/-10% excluding the
# near-nominal band, healthy = 0). These category probabilities themselves
# are NOT from the papers (the papers don't state dataset-generation
# priors) -- they're a reasonable placeholder split, flagged as such.
GAMMA_CATEGORY_WEIGHTS = {"open": 0.20, "short": 0.20, "soft": 0.55, "healthy_like": 0.05}


def sample_gamma(rng: np.random.Generator) -> tuple[float, str]:
    """Returns (gamma, category_label)."""
    category = rng.choice(list(GAMMA_CATEGORY_WEIGHTS.keys()),
                           p=list(GAMMA_CATEGORY_WEIGHTS.values()))
    if category == "open":
        gamma = float(np.clip(rng.normal(0.95, 0.03), 0.7, 1.0))
    elif category == "short":
        gamma = float(np.clip(rng.normal(-0.95, 0.03), -1.0, -0.7))
    elif category == "soft":
        # Z_c +/-10% translates (via the Gamma formula) to roughly |gamma|
        # in [0.02, 0.10]; sign random.
        magnitude = rng.uniform(0.02, 0.10)
        gamma = magnitude * rng.choice([-1.0, 1.0])
    else:  # healthy_like -- a very small residual discontinuity
        gamma = float(rng.uniform(-0.02, 0.02))
    return gamma, category


# ---------------------------------------------------------------------------
# 2. Fault count and location sampling
# ---------------------------------------------------------------------------

FAULT_COUNT_WEIGHTS = {0: 0.10, 1: 0.40, 2: 0.30, 3: 0.15, 4: 0.05}

# Each fault, independently, is deliberately placed into one of these
# location regimes -- so dead_zone/beyond_range are real, labeled cases in
# the dataset rather than fields that are always False. dead_zone/beyond_range
# faults skip the nearest-neighbor separation control below (that logic is
# only meaningful for faults within the cleanly measurable range).
#
# These are the actual intended rates now -- sample_cable_length_m
# guarantees room for "beyond_range" placements (see its docstring), so
# there's no fallback path to compensate for anymore. An earlier version
# used beyond_range=0.18 to compensate for a ~50% silent-fallback rate;
# that compensation was fragile (pulse-dependent, had to be re-tuned by
# hand) and is gone now that the root cause is fixed structurally instead.
LOCATION_REGIME_WEIGHTS = {"dead_zone": 0.10, "beyond_range": 0.07, "valid": 0.83}

# Separation regime targets, applied only among "valid"-regime faults when
# 2+ of them are placed. Each newly placed fault's separation from its
# nearest existing "valid" neighbor is steered into one of
# of these bins so the dataset contains real examples of all three regimes,
# not just the easy one.
SEPARATION_REGIME_WEIGHTS = {"below_resolution": 0.20, "near_resolution": 0.30, "above_resolution": 0.50}


def sample_fault_count(rng: np.random.Generator) -> int:
    return int(rng.choice(list(FAULT_COUNT_WEIGHTS.keys()), p=list(FAULT_COUNT_WEIGHTS.values())))


END_TRUNCATION_PROB = 0.25  # independent, explicit target rate for
                              # end_reflection.beyond_range -- NOT an
                              # emergent side effect of a shared uniform
                              # range anymore (previously this drifted to
                              # 48.8%, then 36.5%, as a side effect of
                              # sample_cable_length_m's range -- now it's a
                              # directly-set probability, verified by test)


def sample_fault_regimes(rng: np.random.Generator, num_faults: int) -> list[str]:
    """Decide each fault's location regime BEFORE cable length is sampled,
    so cable length can be conditioned on what's actually needed -- this is
    what eliminates the fallback path entirely."""
    return [rng.choice(list(LOCATION_REGIME_WEIGHTS.keys()), p=list(LOCATION_REGIME_WEIGHTS.values()))
            for _ in range(num_faults)]


def sample_cable_length_m(rng: np.random.Generator, valid_range: dict, fault_regimes: list[str]) -> float:
    """
    Cable length, conditioned on the already-decided fault regimes:
      - if any fault needs "beyond_range" placement, the cable MUST extend
        far enough past safe_max_m to give it room -- guaranteed here,
        not left to chance (this is what removes the old fallback-to-valid
        path, which silently changed a fault's regime when there wasn't
        room and was flagged as "dangerous long-term": a generator that
        silently changes what it generates without changing what it reports).
      - otherwise, whether the cable extends past safe_max_m (and thus
        truncates the end reflection) is decided independently via
        END_TRUNCATION_PROB, not as a side effect of the overall range.
    """
    dead_zone_max_m = valid_range["dead_zone_max_m"]
    safe_max_m = valid_range["safe_max_m"]

    needs_beyond_range_room = "beyond_range" in fault_regimes
    extends_past_safe = needs_beyond_range_room or (rng.random() < END_TRUNCATION_PROB)

    if extends_past_safe:
        low = safe_max_m * 1.05
        high = safe_max_m * 1.3
    else:
        low = dead_zone_max_m * 3
        high = safe_max_m * 0.98

    return float(rng.uniform(low, high))


def sample_fault_locations(rng: np.random.Generator, fault_regimes: list[str],
                            valid_range: dict, fs: float, v_p: float,
                            cable_length_m: float) -> list[float]:
    """
    Places one location per entry in fault_regimes (already decided by
    sample_fault_regimes, and cable_length_m already guaranteed to have
    room for any "beyond_range" entries) -- no fallback needed here anymore.

    Only "valid"-regime faults participate in the nearest-neighbor
    separation-regime control (SEPARATION_REGIME_WEIGHTS) -- that control is
    about resolvability among cleanly-measurable faults specifically.
    """
    dead_zone_max_m = valid_range["dead_zone_max_m"]
    safe_max_m = valid_range["safe_max_m"]
    min_sep_m = valid_range["min_fault_separation_m"]
    cable_margin_m = 1.0  # keep faults clearly distinct from the cable-end reflection itself

    min_delay_samples = dead_zone_max_m * 2 * fs / v_p
    max_delay_samples = min(safe_max_m, cable_length_m - cable_margin_m) * 2 * fs / v_p
    min_sep_samples = min_sep_m * 2 * fs / v_p

    locations_m = []
    valid_delays_samples = []

    for regime in fault_regimes:
        if regime == "dead_zone":
            hi = min(dead_zone_max_m * 0.95, cable_length_m - 0.1)
            loc = rng.uniform(0.1, max(hi, 0.2))
            locations_m.append(float(loc))
            continue

        if regime == "beyond_range":
            lo = safe_max_m * 1.02
            hi = cable_length_m - 0.1
            assert hi > lo, (
                f"cable_length_m={cable_length_m:.2f} has no room for a beyond_range "
                f"fault (needs > {lo:.2f}m) -- sample_cable_length_m should have guaranteed "
                f"this; this assertion existing is the point, silent fallback is not"
            )
            locations_m.append(float(rng.uniform(lo, hi)))
            continue

        # regime == "valid"
        if not valid_delays_samples:
            d = rng.uniform(min_delay_samples, max_delay_samples)
        else:
            sep_regime = rng.choice(list(SEPARATION_REGIME_WEIGHTS.keys()),
                                     p=list(SEPARATION_REGIME_WEIGHTS.values()))
            if sep_regime == "below_resolution":
                target_nn_sep = rng.uniform(0.1 * min_sep_samples, 0.9 * min_sep_samples)
            elif sep_regime == "near_resolution":
                target_nn_sep = rng.uniform(0.9 * min_sep_samples, 1.3 * min_sep_samples)
            else:
                target_nn_sep = rng.uniform(1.3 * min_sep_samples, 4.0 * min_sep_samples)

            d = rng.uniform(min_delay_samples, max_delay_samples)  # fallback if no candidate accepted
            for _attempt in range(20):
                anchor = rng.choice(valid_delays_samples)
                sign = rng.choice([-1, 1])
                candidate = float(np.clip(anchor + sign * target_nn_sep, min_delay_samples, max_delay_samples))
                actual_nn_sep = min(abs(candidate - dd) for dd in valid_delays_samples)
                if sep_regime == "below_resolution" and actual_nn_sep < min_sep_samples:
                    d = candidate
                    break
                if sep_regime == "near_resolution" and 0.9 * min_sep_samples <= actual_nn_sep <= 1.3 * min_sep_samples:
                    d = candidate

                    break
                if sep_regime == "above_resolution" and actual_nn_sep > 1.3 * min_sep_samples:
                    d = candidate
                    break
        valid_delays_samples.append(d)
        locations_m.append(float(d * v_p / (2 * fs)))

    return sorted(locations_m)


def sample_snr_db(rng: np.random.Generator) -> float:
    return float(rng.choice([5, 10, 15, 20, 30, 40]))


def sample_end_termination(rng: np.random.Generator) -> tuple[str, float]:
    end_type = rng.choice(["open", "short", "matched"], p=[0.4, 0.3, 0.3])
    gamma = {"open": 1.0, "short": -1.0, "matched": 0.0}[end_type]
    return end_type, gamma


# ---------------------------------------------------------------------------
# 3. Dense signed heatmap target
# ---------------------------------------------------------------------------

def build_signed_heatmap(n_samples: int, fs: float, faults_with_delay: list[dict],
                          sigma_samples: float = 2.5) -> np.ndarray:
    """
    Dense target: a Gaussian bump AT EACH FAULT'S delay, scaled by that
    fault's gamma (signed), not a flat unit-amplitude bump -- so the dense
    target preserves polarity the way the object-list label already does.
    Overlapping bumps (below-resolution pairs) sum, which is honest: two
    overlapping reflections really do add in the physical trace too.
    """
    t = np.arange(n_samples)
    heatmap = np.zeros(n_samples)
    for f in faults_with_delay:
        center = f["tau_expected_s"] * fs
        bump = f["gamma"] * np.exp(-((t - center) ** 2) / (2 * sigma_samples ** 2))
        heatmap += bump
    return heatmap


# ---------------------------------------------------------------------------
# 4. Label schema assembly
# ---------------------------------------------------------------------------

def build_label(faults_m: list[float], gammas: list[float], categories: list[str],
                 valid_range: dict, cable: sim.CableParams, fs: float,
                 end_type: str, end_gamma: float, cable_length_m: float,
                 snr_db: float) -> dict:
    faults = []
    for loc, gamma, cat in zip(faults_m, gammas, categories):
        faults.append({
            "location_m": loc,
            "gamma": gamma,
            "gamma_category": cat,
            "dead_zone": sim.classify_fault_location(loc, valid_range) == "dead_zone",
            "beyond_range": sim.classify_fault_location(loc, valid_range) == "beyond_range",
        })

    # Pairwise resolution tracking. IMPORTANT: the end reflection is a real
    # peak (see end_reflection below, and the heatmap it contributes to)
    # subject to the exact same overlap physics as two faults -- a fault
    # placed close to the cable end will have its reflection overlap the
    # end reflection just like two close faults overlap each other. An
    # earlier version only checked fault-vs-fault pairs here, which silently
    # under-reported difficulty for near-end faults (found by visual
    # inspection: e.g. a fault 1.5m from a 13.2m-resolution-limit end
    # reflection, reported as having no resolution issues at all). Treat
    # the end reflection as an extra "virtual fault" at index "end".
    all_locations = list(faults_m) + [cable_length_m]
    all_indices = list(range(len(faults_m))) + ["end"]

    pairs_below_resolution = []
    min_pairwise_separation_m = None
    for a in range(len(all_locations)):
        for b in range(a + 1, len(all_locations)):
            sep = abs(all_locations[a] - all_locations[b])
            if min_pairwise_separation_m is None or sep < min_pairwise_separation_m:
                min_pairwise_separation_m = sep
            if sep < valid_range["min_fault_separation_m"]:
                pairs_below_resolution.append([all_indices[a], all_indices[b]])

    # Explicit end_reflection entry, same shape as a fault entry, so every
    # peak in the heatmap target has a corresponding, unambiguous label
    # entry -- heatmap peaks and (faults + [end_reflection]) are now 1:1.
    # Previously the end reflection only appeared in "cable" (length_m,
    # end_type, end_gamma), with no explicit location_m/dead_zone/
    # beyond_range fields matching the fault-entry shape, so a training
    # pipeline had to know cable.length_m WAS the end-reflection location
    # rather than being told directly.
    end_reflection = {
        "location_m": cable_length_m,
        "gamma": end_gamma,
        "dead_zone": sim.classify_fault_location(cable_length_m, valid_range) == "dead_zone",
        "beyond_range": sim.classify_fault_location(cable_length_m, valid_range) == "beyond_range",
    }

    return {
        "num_faults": len(faults_m),
        "faults": faults,
        "cable": {"length_m": cable_length_m, "end_type": end_type, "end_gamma": end_gamma},
        "end_reflection": end_reflection,
        "signal": {"snr_db": snr_db, "v_p": cable.v_p},
        "pairwise": {
            "min_separation_m": min_pairwise_separation_m,
            "pairs_below_resolution": pairs_below_resolution,
        },
    }


# ---------------------------------------------------------------------------
# 5. One sample end-to-end
# ---------------------------------------------------------------------------

def generate_one_sample(x: np.ndarray, fs: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict]:
    cable = sim.CableParams(v_p=1.65e8)  # v_p held fixed for V1 -- see module note if varying it later
    valid_range = sim.compute_valid_range(x, fs, cable)

    # order matters: fault count and regimes are decided BEFORE cable
    # length, so cable length can be conditioned on what those regimes
    # actually require (see sample_cable_length_m docstring)
    num_faults = sample_fault_count(rng)
    fault_regimes = sample_fault_regimes(rng, num_faults)
    cable_length_m = sample_cable_length_m(rng, valid_range, fault_regimes)
    locations_m = sample_fault_locations(rng, fault_regimes, valid_range, fs, cable.v_p, cable_length_m)
    gammas, categories = [], []
    for _ in range(num_faults):
        g, c = sample_gamma(rng)
        gammas.append(g)
        categories.append(c)

    end_type, end_gamma = sample_end_termination(rng)
    snr_db = sample_snr_db(rng)

    faults = [sim.FaultParams(loc, g) for loc, g in zip(locations_m, gammas)]
    y, sim_label = sim.generate_multi_fault_trace(
        x, fs, cable, faults=faults, cable_length_m=cable_length_m,
        end_gamma=end_gamma, snr_db=snr_db, rng=rng,
    )

    label = build_label(locations_m, gammas, categories, valid_range, cable, fs,
                         end_type, end_gamma, cable_length_m, snr_db)

    heatmap_faults = [{"tau_expected_s": f["tau_expected_s"], "gamma": f["gamma"]} for f in sim_label["faults"]]
    if sim_label["end_reflection"] is not None:
        heatmap_faults.append({"tau_expected_s": sim_label["end_reflection"]["tau_expected_s"],
                                "gamma": sim_label["end_reflection"]["end_gamma"]})
    heatmap = build_signed_heatmap(len(x), fs, heatmap_faults)

    return y, heatmap, label


# ---------------------------------------------------------------------------
# 6. Bulk generation with train/val/test split
# ---------------------------------------------------------------------------

GENERATOR_VERSION = "bucket3-v1.1"  # bump this whenever sampling strategy, label schema,
                                      # or calibration constants change -- a dataset without
                                      # a version tag is unreproducible the moment the code moves on


def _excitation_fingerprint(x: np.ndarray, data_path: str) -> dict:
    import hashlib
    is_real = os.path.exists(data_path)
    return {
        "source": "real_waveform" if is_real else "synthetic_placeholder",
        "source_file": data_path if is_real else None,
        "n_samples": len(x),
        "sha256_first16": hashlib.sha256(x.tobytes()).hexdigest()[:16],
    }


def generate_dataset(x: np.ndarray, fs: float, n_samples: int, output_dir: str,
                      split=(0.8, 0.1, 0.1), seed: int = 0,
                      excitation_data_path: str = "25MHz1V (1).txt") -> dict:
    assert abs(sum(split) - 1.0) < 1e-9
    rng_master = np.random.default_rng(seed)

    cable = sim.CableParams(v_p=1.65e8)  # matches the fixed cable() used inside generate_one_sample
    valid_range = sim.compute_valid_range(x, fs, cable)

    n_train = int(n_samples * split[0])
    n_val = int(n_samples * split[1])
    n_test = n_samples - n_train - n_val
    split_sizes = {"train": n_train, "val": n_val, "test": n_test}

    counts = {"train": 0, "val": 0, "test": 0}
    for split_name, n in split_sizes.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for i in range(n):
            rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
            y, heatmap, label = generate_one_sample(x, fs, rng)
            np.save(os.path.join(split_dir, f"x_{i:05d}.npy"), y.astype(np.float32))
            np.save(os.path.join(split_dir, f"heatmap_{i:05d}.npy"), heatmap.astype(np.float32))
            with open(os.path.join(split_dir, f"y_{i:05d}.json"), "w") as f:
                json.dump(label, f, indent=2)
            counts[split_name] += 1

    metadata = {
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "n_samples": n_samples,
        "split": {"train": n_train, "val": n_val, "test": n_test},
        "sample_rate_hz": fs,
        "excitation": _excitation_fingerprint(x, excitation_data_path),
        "cable_params": {"v_p": cable.v_p, "alpha0": cable.alpha0, "alpha1": cable.alpha1},
        "valid_range": valid_range,
        "sampling_weights": {
            "fault_count": FAULT_COUNT_WEIGHTS,
            "gamma_category": GAMMA_CATEGORY_WEIGHTS,
            "location_regime": LOCATION_REGIME_WEIGHTS,
            "separation_regime": SEPARATION_REGIME_WEIGHTS,
            "end_truncation_prob": END_TRUNCATION_PROB,
        },
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return {"output_dir": output_dir, "counts": counts, "total": sum(counts.values()), "metadata": metadata}


if __name__ == "__main__":
    x, fs = sim.load_excitation(
        data_path="25MHz1V (1).txt",
        header_path="25MHz1V_asciiheader (1).txt",
    )
    print(f"Loaded excitation: {len(x)} samples @ fs={fs:.3g} Hz")

    result = generate_dataset(x, fs, n_samples=50000, output_dir="./dataset_v3_50k")
    print(result)