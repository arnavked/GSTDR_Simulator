"""
Regression suite for the Bucket 1 GSTDR simulator.

Run with:  python3 -m pytest test_simulator.py -v

Each test locks in something discovered by hand during Bucket 1 debugging,
so it gets caught automatically instead of re-discovered manually as
Bucket 2/3 add complexity. Uses the synthetic placeholder excitation by
default (real-waveform validation is a separate, explicitly marked test --
see test_uses_real_excitation_if_available).
"""

import numpy as np
import pytest

import simulator as sim


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def excitation():
    x = sim._synthetic_placeholder_pulse()
    fs = 1e9
    return x, fs


@pytest.fixture
def cable():
    return sim.CableParams(v_p=1.65e8)


@pytest.fixture
def valid_range(excitation, cable):
    x, fs = excitation
    return sim.compute_valid_range(x, fs, cable)


# ---------------------------------------------------------------------------
# 1. Simulator inserts reflections at the analytically expected delay
#    within 1 sample -- using the ISOLATED insertion check, which the
#    ablation experiment confirmed is the correct ground-truth test
#    (the full-trace matched filter has its own, separate bias -- see
#    test suite section 5).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("distance_m", [15.0, 19.0, 30.0, 46.0, 66.0, 80.0])
def test_isolated_insertion_delay_accuracy(excitation, cable, distance_m):
    x, fs = excitation
    fault = sim.FaultParams(location_m=distance_m, gamma=0.3)
    result = sim.isolated_insertion_delay(x, fs, cable, fault)
    assert result["error_samples"] is not None, f"delay fell outside record at {distance_m} m"
    assert abs(result["error_samples"]) <= 1.0, (
        f"insertion delay off by {result['error_samples']:.2f} samples at {distance_m} m"
    )


# ---------------------------------------------------------------------------
# 2. Reflection amplitude decreases monotonically with distance
# ---------------------------------------------------------------------------

def test_amplitude_monotonic_decay(excitation, cable, valid_range):
    x, fs = excitation
    lo = valid_range["dead_zone_max_m"] + 2
    hi = valid_range["safe_max_m"] - 2
    distances = np.linspace(lo, hi, 12)

    amplitudes = []
    for d in distances:
        fault = sim.FaultParams(location_m=d, gamma=0.3)
        result = sim.isolated_insertion_delay(x, fs, cable, fault)
        amplitudes.append(result["peak_amplitude"])

    amplitudes = np.array(amplitudes)
    diffs = np.diff(amplitudes)
    tolerance_frac = 0.02  # allow tiny floating-point noise, not real jumps
    violations = diffs > tolerance_frac * amplitudes[:-1]
    assert not violations.any(), (
        f"amplitude increased between consecutive distances at indices {np.where(violations)[0].tolist()}"
    )


# ---------------------------------------------------------------------------
# 3. Faults inside the dead zone are correctly flagged
# ---------------------------------------------------------------------------

def test_dead_zone_classification(excitation, cable, valid_range):
    x, fs = excitation
    dead_zone_example = valid_range["dead_zone_max_m"] * 0.5
    valid_example = (valid_range["dead_zone_max_m"] + valid_range["safe_max_m"]) / 2

    assert sim.classify_fault_location(dead_zone_example, valid_range) == "dead_zone"
    assert sim.classify_fault_location(valid_example, valid_range) == "valid"


# ---------------------------------------------------------------------------
# 4. Reflections beyond the maximum measurable range are flagged, not
#    silently generated as if they were normal
# ---------------------------------------------------------------------------

def test_beyond_range_classification(excitation, cable, valid_range):
    beyond_example = valid_range["safe_max_m"] + 5
    assert sim.classify_fault_location(beyond_example, valid_range) == "beyond_range"


def test_beyond_range_reflection_is_truncated_not_silent(excitation, cable, valid_range):
    """
    Sanity check that a beyond-range fault actually shows degraded/truncated
    behavior (large error or missing peak) rather than silently looking fine
    -- catches the case where classify_fault_location's boundary drifts out
    of sync with what the propagation model actually does.
    """
    x, fs = excitation
    beyond_example = valid_range["safe_max_m"] + 8
    fault = sim.FaultParams(location_m=beyond_example, gamma=0.3)
    result = sim.isolated_insertion_delay(x, fs, cable, fault)
    is_degraded = (result["error_samples"] is None) or (abs(result["error_samples"]) > 2)
    assert is_degraded, (
        f"expected truncation/degradation beyond safe_max_m="
        f"{valid_range['safe_max_m']:.1f}, but got clean error={result['error_samples']}"
    )


# ---------------------------------------------------------------------------
# 5. Reproducibility -- the RNG bug from earlier in this project (same code,
#    different results across runs) must never come back silently.
# ---------------------------------------------------------------------------

def test_noise_generation_is_reproducible_with_seed(excitation, cable):
    x, fs = excitation
    fault = sim.FaultParams(location_m=19.0, gamma=0.3)

    y1, _ = sim.generate_single_fault_trace(x, fs, cable, fault, snr_db=30,
                                             rng=np.random.default_rng(42))
    y2, _ = sim.generate_single_fault_trace(x, fs, cable, fault, snr_db=30,
                                             rng=np.random.default_rng(42))
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_give_different_noise(excitation, cable):
    """Guards against the opposite failure mode: a seed that's accidentally ignored."""
    x, fs = excitation
    fault = sim.FaultParams(location_m=19.0, gamma=0.3)

    y1, _ = sim.generate_single_fault_trace(x, fs, cable, fault, snr_db=30,
                                             rng=np.random.default_rng(1))
    y2, _ = sim.generate_single_fault_trace(x, fs, cable, fault, snr_db=30,
                                             rng=np.random.default_rng(2))
    assert not np.array_equal(y1, y2)


# ---------------------------------------------------------------------------
# 7. Bucket 2 -- multi-fault, cable-end reflection, gamma variation
# ---------------------------------------------------------------------------

def test_two_fault_case_matches_barc_9m_19m_isolated(excitation):
    """Validates propagation math only (each fault checked alone) -- see
    test_two_fault_case_combined_shows_resolution_limit for the real test."""
    x, fs = excitation
    result = sim.validate_two_fault_case(x, fs, gamma=0.3)
    assert result["pass"], result["per_fault"]


def test_two_fault_case_combined_shows_resolution_limit(excitation, valid_range):
    """
    The BARC 9m/19m case, with THIS excitation pulse (176-sample footprint),
    has only 121 samples of separation -- below min_fault_separation_m.
    Expect degraded recovery here: this documents a known, understood
    resolution limit, not a simulator bug. If this ever starts PASSING
    without the footprint/pulse changing, something else changed and is
    worth investigating.
    """
    x, fs = excitation
    separation_m = 19.0 - 9.0
    assert separation_m < valid_range["min_fault_separation_m"], (
        "test assumption stale: 9m/19m separation is no longer below the "
        "resolution limit for this excitation pulse -- check footprint"
    )
    result = sim.validate_two_fault_case_combined(x, fs, gamma=0.3)
    assert not result["pass"], (
        "expected degraded recovery due to insufficient fault separation, "
        "but got a clean pass -- resolution-limit assumption may be wrong"
    )


def test_well_separated_faults_pass_combined(excitation, valid_range):
    """Two faults separated by MORE than min_fault_separation_m should be
    cleanly recoverable together -- the actual 'no cross-talk' guarantee."""
    x, fs = excitation
    cable = sim.CableParams(v_p=1.65e8)
    separation_m = valid_range["min_fault_separation_m"] * 1.5
    loc_a, loc_b = 20.0, 20.0 + separation_m
    faults = [sim.FaultParams(loc_a, 0.3), sim.FaultParams(loc_b, -0.2)]
    y, label = sim.generate_multi_fault_trace(x, fs, cable, faults=faults, snr_db=None)

    zero_lag_idx = len(x) - 1
    envelope, _ = sim._envelope_matched_filter(y, x)
    for f in label["faults"]:
        expected = f["tau_expected_s"] * fs
        lo, hi = int(zero_lag_idx + expected - 20), int(zero_lag_idx + expected + 20)
        peak = np.argmax(envelope[lo:hi]) + lo
        err = (peak - zero_lag_idx) - expected
        assert abs(err) <= 1.0, f"fault at {f['location_m']}m: off by {err:.2f} samples despite adequate separation"


def test_cable_end_reflection_open_circuit(excitation):
    x, fs = excitation
    result = sim.validate_cable_end_reflection(x, fs, cable_length_m=30.0, end_gamma=1.0)
    assert result["pass"], result


def test_cable_end_reflection_short_circuit(excitation):
    x, fs = excitation
    result = sim.validate_cable_end_reflection(x, fs, cable_length_m=30.0, end_gamma=-1.0)
    assert result["pass"], result


@pytest.mark.parametrize("gamma", [-0.8, -0.3, -0.05, 0.05, 0.3, 0.8])
def test_delay_accuracy_across_gamma_sign_and_magnitude(excitation, cable, gamma):
    """
    The delay math shouldn't care about gamma's sign or magnitude -- only
    the propagation delay (location) determines timing; gamma only scales
    amplitude. Sweeping sign and magnitude here catches a bug where, e.g.,
    a negative gamma flips something in the envelope detector that a
    positive-only gamma=0.3 test (used everywhere else) would never expose.
    """
    x, fs = excitation
    fault = sim.FaultParams(location_m=40.0, gamma=gamma)
    result = sim.isolated_insertion_delay(x, fs, cable, fault)
    assert result["error_samples"] is not None
    assert abs(result["error_samples"]) <= 1.0, (
        f"gamma={gamma}: delay off by {result['error_samples']:.2f} samples"
    )




def test_uses_real_excitation_if_available():
    import os
    data_path = "25MHz1V (1).txt"
    if not os.path.exists(data_path):
        pytest.skip(
            "Real excitation file not present -- all other tests in this suite "
            "are validating simulator architecture against the placeholder pulse, "
            "not your actual hardware waveform. Not a failure, just a reminder."
        )
    x, fs = sim.load_excitation(
        data_path=data_path,
        header_path="25MHz1V_asciiheader (1).txt",
    )
    assert len(x) > 0