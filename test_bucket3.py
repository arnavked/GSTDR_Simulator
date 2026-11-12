"""
Regression suite for the Bucket 3 dataset generator.

Run with:  python3 -m pytest test_bucket3.py -v

Generates a pilot batch once per test session (fixed seed, so deterministic)
and checks the invariants that have already broken once each by hand in this
project: fixed cable length, boundary violations, and regime rates silently
missing their targets due to fallback logic. These are enforced automatically
now instead of being re-discovered by manual inspection every time something
in the sampling strategy changes.
"""

import numpy as np
import pytest

import simulator as sim
import bucket3_dataset_generator as gen


PILOT_N = 300  # large enough for stable regime-rate statistics, small enough to run in seconds


@pytest.fixture(scope="module")
def excitation():
    return gen.sim.load_excitation("25MHz1V (1).txt", "25MHz1V_asciiheader (1).txt")


@pytest.fixture(scope="module")
def pilot_batch(excitation):
    """One fixed-seed pilot batch, shared across tests in this module."""
    x, fs = excitation
    rng_master = np.random.default_rng(999)
    samples = []
    for _ in range(PILOT_N):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        y, heatmap, label = gen.generate_one_sample(x, fs, rng)
        samples.append((y, heatmap, label))
    return samples


# ---------------------------------------------------------------------------
# 1. No boundary violations -- a fault must never land at/beyond the cable's
#    own end. Caught by hand once already (16 violations from an exact-
#    equality edge case).
# ---------------------------------------------------------------------------

def test_no_boundary_violations(pilot_batch):
    violations = [
        (label["cable"]["length_m"], f["location_m"])
        for _, _, label in pilot_batch
        for f in label["faults"]
        if f["location_m"] >= label["cable"]["length_m"]
    ]
    assert len(violations) == 0, f"faults placed at/beyond cable end: {violations[:5]}"


# ---------------------------------------------------------------------------
# 2. Regime distribution matches intended targets, within tolerance for
#    PILOT_N samples. Caught by hand once already (beyond_range realized at
#    1.9% against a 7% target, due to silent fallback).
# ---------------------------------------------------------------------------

def test_dead_zone_rate_near_target(pilot_batch):
    all_faults = [f for _, _, label in pilot_batch for f in label["faults"]]
    assert len(all_faults) > 20, "pilot batch too small / fault count sampling looks broken"
    rate = sum(f["dead_zone"] for f in all_faults) / len(all_faults)
    target = gen.LOCATION_REGIME_WEIGHTS["dead_zone"]
    assert abs(rate - target) < 0.06, f"dead_zone rate {rate:.1%} too far from target {target:.0%}"


def test_beyond_range_fault_rate_near_target(pilot_batch):
    all_faults = [f for _, _, label in pilot_batch for f in label["faults"]]
    rate = sum(f["beyond_range"] for f in all_faults) / len(all_faults)
    target = gen.LOCATION_REGIME_WEIGHTS["beyond_range"]
    assert abs(rate - target) < 0.06, f"beyond_range rate {rate:.1%} too far from target {target:.0%}"


def test_end_truncation_rate_reasonable(pilot_batch):
    """
    Not pinned tightly to END_TRUNCATION_PROB alone, since
    needs_beyond_range_room forces extension independently and adds on top
    -- checked against a wider, explicitly-reasoned band instead of a tight
    single-number target.
    """
    rate = sum(label["end_reflection"]["beyond_range"] for _, _, label in pilot_batch) / len(pilot_batch)
    assert 0.15 < rate < 0.45, f"end-reflection truncation rate {rate:.1%} outside the intended minority-case band"


# ---------------------------------------------------------------------------
# 3. Cable length is actually variable, not a fixed formula.
# ---------------------------------------------------------------------------

def test_cable_length_variability(pilot_batch):
    lengths = [label["cable"]["length_m"] for _, _, label in pilot_batch]
    n_distinct = len(set(np.round(lengths, 1)))
    assert n_distinct > PILOT_N * 0.5, (
        f"only {n_distinct}/{PILOT_N} distinct cable lengths -- looks close to fixed again"
    )


def test_below_resolution_pair_rate_reasonable(pilot_batch):
    """
    Every sample's end reflection counts as a "virtual fault" for pairwise
    resolution purposes (see build_label), so total pairs per sample =
    C(num_faults + 1, 2), and samples with num_faults >= 1 (not just >= 2)
    now have at least one pair to check (fault vs end).

    Not tightly pinned to a single target rate -- this statistic blends two
    different physical situations (fault-vs-fault proximity, which is
    directly controlled, and fault-vs-end proximity, which isn't controlled
    at all -- cable_length_m and fault placement are sampled independently).
    Wide band here to catch a large regression, not to enforce a precise rate.
    """
    with_faults = [label for _, _, label in pilot_batch if label["num_faults"] >= 1]
    n_pairs = sum((label["num_faults"] + 1) * label["num_faults"] // 2 for label in with_faults)
    if n_pairs == 0:
        pytest.skip("no samples with faults in this pilot batch")
    n_below = sum(len(label["pairwise"]["pairs_below_resolution"]) for label in with_faults)
    rate = n_below / n_pairs
    assert 0.10 < rate < 0.65, f"below-resolution pair rate {rate:.1%} outside the expected band"


# ---------------------------------------------------------------------------
# 4. Heatmap peaks correspond exactly to (faults + end_reflection) --
#    the mismatch flagged and fixed as Issue 1.
# ---------------------------------------------------------------------------

def test_heatmap_matches_label_peak_count(excitation):
    x, fs = excitation
    rng = np.random.default_rng(42)
    # force a known multi-fault case for a deterministic, inspectable check
    cable = sim.CableParams(v_p=1.65e8)
    faults = [sim.FaultParams(20.0, 0.5), sim.FaultParams(50.0, -0.3)]
    y, sim_label = sim.generate_multi_fault_trace(
        x, fs, cable, faults=faults, cable_length_m=70.0, end_gamma=1.0, snr_db=None, rng=rng,
    )
    heatmap_faults = [{"tau_expected_s": f["tau_expected_s"], "gamma": f["gamma"]} for f in sim_label["faults"]]
    heatmap_faults.append({"tau_expected_s": sim_label["end_reflection"]["tau_expected_s"],
                            "gamma": sim_label["end_reflection"]["end_gamma"]})
    heatmap = gen.build_signed_heatmap(len(x), fs, heatmap_faults)

    # count resolvable peaks in the heatmap (local maxima above a small threshold)
    from scipy.signal import find_peaks
    peak_idx, _ = find_peaks(np.abs(heatmap), height=0.05, distance=10)
    # expect exactly 3 peaks: 2 faults + 1 end reflection (all well-separated
    # by construction in this test case)
    assert len(peak_idx) == 3, f"expected 3 heatmap peaks (2 faults + end), found {len(peak_idx)}"


# ---------------------------------------------------------------------------
# 5. Reproducibility -- same seed, same dataset sample.
# ---------------------------------------------------------------------------

def test_generation_is_reproducible_with_seed(excitation):
    x, fs = excitation
    y1, h1, label1 = gen.generate_one_sample(x, fs, np.random.default_rng(7))
    y2, h2, label2 = gen.generate_one_sample(x, fs, np.random.default_rng(7))
    np.testing.assert_array_equal(y1, y2)
    np.testing.assert_array_equal(h1, h2)
    assert label1 == label2


# ---------------------------------------------------------------------------
# 6. No-fallback guarantee -- the assertion in sample_fault_locations should
#    never fire across a large pilot batch (this IS the regression test for
#    "silent fallback" not coming back).
# ---------------------------------------------------------------------------

def test_no_fallback_assertion_ever_fires(excitation):
    x, fs = excitation
    rng_master = np.random.default_rng(12345)
    for _ in range(PILOT_N):
        rng = np.random.default_rng(rng_master.integers(0, 2**32 - 1))
        gen.generate_one_sample(x, fs, rng)  # raises AssertionError if the guarantee ever breaks