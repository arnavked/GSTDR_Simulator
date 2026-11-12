"""
Bucket 1+2 — GSTDR simulator core.

Bucket 1 (complete, regression-tested — see test_simulator.py):
    1. Load the real excitation waveform (your 25 MHz / 1 GSa/s AWG capture).
    2. Model a SINGLE fault reflection with frequency-dependent attenuation,
       anchored to a real RG58 datasheet figure (~21 dB/100m @ 100 MHz).
    3. Validate against the BARC paper's 19 m single-fault case.
    4. Dead-zone / safe-range boundaries characterized and classified.

Bucket 2 (this addition):
    5. Multiple simultaneous faults, each with independent location and gamma
       (sign and magnitude), summed into one trace.
    6. Cable-end reflection (open/short/matched termination at a given
       cable length), modeled the same way as a fault reflection.
    7. Validated against the BARC paper's two-fault case (9 m + 19 m).

Explicitly OUT of scope for Bucket 2 (flagged, not silently skipped):
    - re-reflections / multipath between faults (a pulse bouncing off fault 1,
      re-reflecting off fault 2, and returning again) -- first-order
      reflections only for now
    - domain-randomized bulk generation (Bucket 3)
    - quantization / colored noise (later refinement)

ASSUMPTIONS THAT ARE NOT YET VERIFIED AGAINST YOUR HARDWARE
(flag these the same way unverified register constants get flagged
elsewhere in this project line):
    - attenuation model alpha(f) = ALPHA0 + ALPHA1 * sqrt(f), with alpha1
      anchored to a generic RG58 datasheet figure, NOT your specific cable's
      measured or datasheet value. Replace before trusting absolute
      reflection amplitudes.
    - v_p = 1.65e8 m/s is the BARC paper's reported value for their EPR
      cable, used here to reproduce their validation cases.
    - end-reflection gamma values (+1 open, -1 short) are ideal-termination
      approximations, same convention the Gargouri papers use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# 1. Excitation loader
# ---------------------------------------------------------------------------

def load_excitation(data_path: str, header_path: str | None = None) -> tuple[np.ndarray, float]:
    """
    Load the real AWG excitation waveform.

    Expects a simple two-column (or single-column) ASCII export, e.g. the
    "25MHz1V (1).txt" / "25MHz1V_asciiheader (1).txt" pair described in this
    project. If the files aren't present on disk, falls back to a synthetic
    placeholder pulse with the SAME record format (fs=1 GHz, N=1168, ~25 MHz,
    ~160 ns duration) so this module still runs standalone -- but the
    placeholder is clearly flagged and should NOT be used as a stand-in for
    real hardware validation.

    Returns
    -------
    x : np.ndarray
        Voltage samples.
    fs : float
        Sample rate in Hz.
    """
    if os.path.exists(data_path):
        fs = _parse_sample_rate(header_path) if header_path else 1e9
        raw = _load_waveform_data(data_path)
        # Handle both single-column (voltage only) and two-column (time, voltage)
        x = raw[:, -1] if raw.ndim == 2 else raw
        return np.asarray(x, dtype=float), fs

    print(
        "[WARNING] Real excitation file not found at "
        f"'{data_path}'. Falling back to a SYNTHETIC placeholder pulse "
        "(25 MHz Gaussian-modulated sine, fs=1 GHz, N=1168). "
        "Do NOT use this placeholder for hardware-validation claims -- "
        "drop your actual 25MHz1V.txt export into this path instead."
    )
    return _synthetic_placeholder_pulse(), 1e9


def _load_waveform_data(data_path: str) -> np.ndarray:
    """
    Waveform-export formats vary across instruments -- the actual capture
    used in this project turned out to be comma-delimited with a one-line
    text header ("B,A"), which plain np.loadtxt (whitespace-delimited, no
    header) can't parse. Tries the formats actually seen; raises clearly if
    none work rather than silently returning garbage.
    """
    attempts = [
        dict(delimiter=",", skip_header=1),  # comma-delimited with header row (the real capture's format)
        dict(delimiter=",", skip_header=0),  # comma-delimited, no header
        dict(delimiter=None, skip_header=0),  # whitespace-delimited, no header
    ]
    last_error = None
    for kwargs in attempts:
        try:
            raw = np.genfromtxt(data_path, **kwargs)
            if raw.size > 0 and not np.all(np.isnan(raw)):
                return raw
        except Exception as e:
            last_error = e
    raise ValueError(f"Could not parse waveform file '{data_path}' with any known format. Last error: {last_error}")


def _parse_sample_rate(header_path: str) -> float:
    """
    Pull the sample rate out of the ascii header file. Falls back to 1 GHz
    (matching the described acquisition) if the header can't be parsed --
    this fallback is flagged, not silent.
    """
    try:
        with open(header_path, "r") as f:
            text = f.read()
        for line in text.splitlines():
            low = line.lower()
            if "sample rate" in low or "samplerate" in low:
                digits = "".join(ch for ch in line if ch.isdigit() or ch == ".")
                if digits:
                    val = float(digits)
                    # crude unit sniff: header values are typically in Hz already
                    return val
    except Exception as e:
        print(f"[WARNING] Could not parse header '{header_path}': {e}. Assuming fs=1e9.")
    return 1e9


def _synthetic_placeholder_pulse(fs: float = 1e9, n_samples: int = 1168,
                                  f0: float = 25e6, amp: float = 1.0) -> np.ndarray:
    """Flagged placeholder only -- matches the shape described for the real capture."""
    t = np.arange(n_samples) / fs
    t0 = 80e-9          # envelope peak near 80 ns, matching the real capture
    sigma = 35e-9        # tuned so the pulse is ~160 ns wide, matching the real capture
    envelope = np.exp(-((t - t0) ** 2) / (2 * sigma ** 2))
    return amp * envelope * np.sin(2 * np.pi * f0 * (t - t0))


# ---------------------------------------------------------------------------
# 2. Single-fault reflection model (frequency domain, zero-padded)
# ---------------------------------------------------------------------------

@dataclass
class CableParams:
    v_p: float = 1.65e8      # propagation velocity, m/s (BARC EPR cable value)
    alpha0: float = 0.0      # Np/m, frequency-independent loss term  [UNVERIFIED]
    alpha1: float = 2.4e-6   # Np/(m*sqrt(Hz)), skin-effect loss term.
                              # Anchored to a real RG58 datasheet figure (~21 dB/100m
                              # at 100 MHz -> 0.21 dB/m -> 0.0242 Np/m -> alpha1 =
                              # 0.0242/sqrt(100e6) = 2.4e-6). Still generic (not YOUR
                              # cable's measured datasheet or actual sample), but no
                              # longer producing the physically-impossible near-total
                              # high-frequency attenuation the earlier 2e-5 placeholder
                              # gave, which was smearing reflections across the entire
                              # record instead of a plausible attenuated echo -- caught
                              # by the safe_max_m regression test failing at 80 m.


@dataclass
class FaultParams:
    location_m: float        # distance from TX end, meters
    gamma: float              # reflection coefficient, signed, -1..1


def _attenuation_db_per_root_f(cable: CableParams, freqs_hz: np.ndarray) -> np.ndarray:
    """
    alpha(f) = alpha0 + alpha1 * sqrt(f), a standard first-order coax
    skin-effect + DC-loss approximation. Returns Nepers/meter, per frequency bin.
    NOTE: alpha0/alpha1 defaults are generic placeholders (see module docstring)
    -- swap in RG58 datasheet or measured values before trusting amplitudes.
    """
    freqs_hz = np.abs(freqs_hz)
    return cable.alpha0 + cable.alpha1 * np.sqrt(freqs_hz)


def reflect_single_fault(x: np.ndarray, fs: float, cable: CableParams,
                          fault: FaultParams, record_len: int | None = None) -> tuple[np.ndarray, float]:
    """
    Produce the fault-reflection component for one fault, using a
    frequency-domain model:

        X_f(f) = X(f) * Gamma * exp(-2 * alpha(f) * d) * exp(-j*2*pi*f*tau)

    i.e. the excitation spectrum, scaled by the (frequency-dependent)
    round-trip attenuation and the (frequency-independent, non-dispersive)
    propagation delay tau = 2*d/v_p.

    Zero-pads to `record_len` (or 2x len(x) if not given) BEFORE taking the
    FFT so the delayed copy cannot wrap around the record -- this was
    flagged as a correctness risk in the review of this approach.

    Returns
    -------
    reflected : np.ndarray, length record_len
        The reflected waveform, aligned to the same t=0 as `x`.
    tau : float
        The round-trip delay in seconds (for validation / labeling).
    """
    n = len(x)
    tau = 2 * fault.location_m / cable.v_p

    if record_len is None:
        record_len = n
    # generous padding: record_len should already be sized to comfortably
    # hold the largest expected delay; this is an internal safety pad only.
    nfft = record_len + n

    X = np.fft.rfft(x, n=nfft)
    freqs = np.fft.rfftfreq(nfft, d=1 / fs)

    atten_nepers_per_m = _attenuation_db_per_root_f(cable, freqs)
    round_trip_atten = np.exp(-2 * atten_nepers_per_m * fault.location_m)

    phase_delay = np.exp(-1j * 2 * np.pi * freqs * tau)

    X_reflected = X * fault.gamma * round_trip_atten * phase_delay
    reflected_full = np.fft.irfft(X_reflected, n=nfft)

    return reflected_full[:record_len], tau


# ---------------------------------------------------------------------------
# 3. Assemble one labeled trace (direct pulse + single fault + noise)
# ---------------------------------------------------------------------------

def generate_single_fault_trace(x: np.ndarray, fs: float, cable: CableParams,
                                 fault: FaultParams | None, snr_db: float | None = None,
                                 rng: np.random.Generator | None = None
                                 ) -> tuple[np.ndarray, dict]:
    """
    Build one trace: the directly-coupled excitation pulse, plus (if given)
    one fault reflection, plus (if given) additive white Gaussian noise
    calibrated to snr_db. Cable-end reflection is intentionally NOT modeled
    yet (Bucket 2).

    `rng` should be an explicit np.random.Generator (e.g. np.random.default_rng(seed))
    -- NOT bare np.random calls. An earlier version used np.random.normal directly
    with no seed, which made validation runs non-reproducible: the same code gave
    different matched-filter results on different runs, purely from noise-seed
    luck. Always pass a seeded generator for anything you intend to validate
    or compare against.
    """
    n = len(x)
    y = np.zeros(n, dtype=float)
    y[: len(x)] += x  # directly-coupled pulse, unattenuated (TX side)

    label = {
        "fault_present": False,
        "fault_location_m": None,
        "gamma": None,
        "tau_expected_s": None,
        "snr_db": snr_db,
        "v_p": cable.v_p,
    }

    if fault is not None:
        reflected, tau = reflect_single_fault(x, fs, cable, fault, record_len=n)
        y += reflected
        label.update(
            fault_present=True,
            fault_location_m=fault.location_m,
            gamma=fault.gamma,
            tau_expected_s=tau,
        )

    if snr_db is not None:
        # Reference signal power off the EXCITATION'S OWN footprint, not the
        # full record. The record is mostly zeros after the ~160 ns pulse,
        # so mean(y**2) over the whole array silently dilutes the "signal"
        # power and lets a nominal 30 dB SNR bury a real, small fault
        # reflection under noise strong enough to fool the matched filter.
        signal_power = np.mean(x ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        if rng is None:
            rng = np.random.default_rng()  # unseeded -- fine for bulk generation, NOT for validation
        noise = rng.normal(0, np.sqrt(noise_power), size=n)
        y = y + noise

    return y, label


# ---------------------------------------------------------------------------
# 4. Validation against the BARC 19 m single-fault case
# ---------------------------------------------------------------------------

def _pulse_footprint_samples(x: np.ndarray) -> int:
    """Width (in samples) of the excitation's own energy support, used to skip
    past its self-correlation sidelobes before searching for a reflection."""
    envelope = np.abs(x)
    footprint_idx = np.where(envelope > 0.01 * envelope.max())[0]
    return int(footprint_idx.max() - footprint_idx.min()) if len(footprint_idx) else 0


def _envelope_matched_filter(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, int]:
    """
    Cross-correlate y against template x, then take the Hilbert-envelope
    magnitude of the correlation rather than its raw (oscillatory) value.

    A 25 MHz Gaussian-modulated sine produces an oscillatory correlation --
    raw argmax(|corr|) can lock onto an RF side-oscillation one or two
    samples off the true arrival, or (worse, at low SNR) onto an unrelated
    noise excursion elsewhere in the record. The envelope of the correlation
    is what actually represents "how well does a delayed, scaled copy of the
    template match here", which is the physically meaningful detection
    statistic -- and what the BARC paper itself uses (matched filter +
    envelope), not a bespoke choice.

    Returns
    -------
    envelope : np.ndarray
        Envelope magnitude at each lag, aligned with zero_lag_idx = len(x)-1.
    zero_lag_idx : int
    """
    from scipy.signal import hilbert
    corr = np.correlate(y, x, mode="full")
    envelope = np.abs(hilbert(corr))
    zero_lag_idx = len(x) - 1
    return envelope, zero_lag_idx


def validate_simulation_correctness(x: np.ndarray, fs: float, cable: CableParams,
                                     fault: FaultParams, snr_db: float | None,
                                     rng: np.random.Generator,
                                     search_tolerance_samples: int = 50) -> dict:
    """
    TEST 1 -- Simulation correctness (uses ground truth ON PURPOSE).

    Confirms the generator itself places the reflection at the physically
    correct delay: searches only in a small window around the KNOWN,
    analytically-expected tau. This validates `reflect_single_fault`'s
    math, not any detection algorithm -- privileged ground truth is
    appropriate here because that's exactly what's being checked.
    """
    y, label = generate_single_fault_trace(x, fs, cable, fault, snr_db=snr_db, rng=rng)
    envelope, zero_lag_idx = _envelope_matched_filter(y, x)

    expected_delay_samples = int(round(label["tau_expected_s"] * fs))
    lo = zero_lag_idx + expected_delay_samples - search_tolerance_samples
    hi = zero_lag_idx + expected_delay_samples + search_tolerance_samples
    lo, hi = max(lo, 0), min(hi, len(envelope))

    local_peak_idx = np.argmax(envelope[lo:hi]) + lo
    measured_delay_samples = local_peak_idx - zero_lag_idx

    return {
        "expected_delay_samples": expected_delay_samples,
        "measured_delay_samples": measured_delay_samples,
        "error_samples": measured_delay_samples - expected_delay_samples,
        "pass": abs(measured_delay_samples - expected_delay_samples) <= 2,
    }


def detect_fault_blind(y: np.ndarray, x: np.ndarray, fs: float, v_p: float,
                        max_cable_length_m: float | None = None) -> int:
    """
    TEST 2 support -- Detection, with NO privileged ground truth.

    Searches the entire physically valid post-excitation region of the
    record for the strongest envelope peak. This is the search a real
    deployed detector would have to do (it doesn't know the answer),
    so it must NOT be narrowed around a known fault location.
    """
    envelope, zero_lag_idx = _envelope_matched_filter(y, x)
    footprint = _pulse_footprint_samples(x)
    search_start = zero_lag_idx + footprint + 10  # past self-correlation only

    search_end = len(envelope)
    if max_cable_length_m is not None:
        max_delay_s = 2 * max_cable_length_m / v_p
        search_end = min(search_end, zero_lag_idx + int(max_delay_s * fs))

    if search_end <= search_start:
        return -1  # no valid search region
    peak_idx = np.argmax(envelope[search_start:search_end]) + search_start
    return peak_idx - zero_lag_idx


def monte_carlo_snr_sweep(x: np.ndarray, fs: float, cable: CableParams,
                           fault: FaultParams, snr_values_db: list[float],
                           n_trials: int, tolerance_samples: int = 5,
                           base_seed: int = 1000) -> list[dict]:
    """
    TEST 2 -- Detection robustness (blind search, no ground truth used
    during detection; ground truth only used AFTER, to score success).

    For each SNR, runs n_trials independent noise realizations (seeded,
    reproducible) and reports detection rate + error statistics. This
    replaces the single-seed / 10-seed spot checks with an actual
    characterization curve, which is what a simulator feeding a
    100k+-sample dataset needs before you trust it.
    """
    results = []
    for snr_db in snr_values_db:
        errors = []
        hits = 0
        for trial in range(n_trials):
            rng = np.random.default_rng(base_seed + trial)
            y, label = generate_single_fault_trace(x, fs, cable, fault, snr_db=snr_db, rng=rng)
            measured_delay = detect_fault_blind(y, x, fs, cable.v_p,
                                                 max_cable_length_m=fault.location_m * 3)
            expected_delay = label["tau_expected_s"] * fs
            err = measured_delay - expected_delay
            errors.append(err)
            if abs(err) <= tolerance_samples:
                hits += 1
        errors = np.array(errors)
        results.append({
            "snr_db": snr_db,
            "n_trials": n_trials,
            "detection_rate": hits / n_trials,
            "mean_abs_error_samples": float(np.mean(np.abs(errors))),
            "p95_abs_error_samples": float(np.percentile(np.abs(errors), 95)),
        })
    return results


def generate_multi_fault_trace(x: np.ndarray, fs: float, cable: CableParams,
                                faults: list[FaultParams] | None = None,
                                cable_length_m: float | None = None,
                                end_gamma: float | None = None,
                                snr_db: float | None = None,
                                rng: np.random.Generator | None = None
                                ) -> tuple[np.ndarray, dict]:
    """
    Bucket 2 generator: directly-coupled pulse + N independent fault
    reflections + an optional cable-end reflection, summed into one trace.

    The cable-end reflection is modeled with the SAME mechanism as a fault
    reflection (reflect_single_fault at location=cable_length_m, gamma=
    end_gamma) -- physically that's exactly what it is: an impedance
    discontinuity at the termination. Typical end_gamma: +1.0 (open
    circuit), -1.0 (short circuit), 0.0 (matched, no end reflection).

    First-order reflections only: a pulse reflecting off fault 1 and then
    re-reflecting off fault 2 (multipath) is NOT modeled (flagged in the
    module docstring, not silently ignored).

    Label includes each fault's ground-truth (location, gamma, expected
    delay) plus the end reflection's, so Bucket 3 can write these straight
    into the dataset metadata without re-deriving them.
    """
    n = len(x)
    y = np.zeros(n, dtype=float)
    y[: len(x)] += x

    label = {
        "faults": [],
        "end_reflection": None,
        "snr_db": snr_db,
        "v_p": cable.v_p,
    }

    for fault in (faults or []):
        reflected, tau = reflect_single_fault(x, fs, cable, fault, record_len=n)
        y += reflected
        label["faults"].append({
            "location_m": fault.location_m,
            "gamma": fault.gamma,
            "tau_expected_s": tau,
        })

    if cable_length_m is not None and end_gamma is not None:
        end_fault = FaultParams(location_m=cable_length_m, gamma=end_gamma)
        end_reflected, end_tau = reflect_single_fault(x, fs, cable, end_fault, record_len=n)
        y += end_reflected
        label["end_reflection"] = {
            "cable_length_m": cable_length_m,
            "end_gamma": end_gamma,
            "tau_expected_s": end_tau,
        }

    if snr_db is not None:
        signal_power = np.mean(x ** 2)  # same footprint-anchored SNR reference as Bucket 1
        noise_power = signal_power / (10 ** (snr_db / 10))
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.normal(0, np.sqrt(noise_power), size=n)
        y = y + noise

    return y, label


def compute_valid_range(x: np.ndarray, fs: float, cable: CableParams) -> dict:
    """
    Derives the fault-location range over which delay measurements are
    physically meaningful, from the excitation footprint and record length:

      - dead_zone_max_m: below this, the reflection arrives while still
        overlapping the direct pulse's own footprint (the reflection and
        the direct pulse are not separable in time). This is a genuine TDR
        resolution limit, not a simulator defect.
      - safe_max_m: above this, the reflected pulse's footprint no longer
        fully fits before the record ends, so it gets truncated by the
        recording window.

    Between these two bounds is the "valid" range for clean delay recovery.
    """
    footprint = _pulse_footprint_samples(x)
    n = len(x)
    dead_zone_max_m = footprint * cable.v_p / (2 * fs)
    safe_max_m = (n - footprint) * cable.v_p / (2 * fs)
    # Two faults closer together than one pulse footprint produce OVERLAPPING
    # reflections in the trace -- regardless of gamma sign, a local envelope
    # search on the combined trace cannot cleanly separate them (confirmed
    # by the multi-fault regression test: 9m+19m, 121-sample separation,
    # with a 176-sample footprint, biases both faults' apparent position by
    # up to ~20 samples). This is the same physical mechanism as the direct
    # pulse / dead-zone overlap, just between two faults instead of fault-vs-
    # direct-pulse.
    min_fault_separation_m = footprint * cable.v_p / (2 * fs)
    return {"dead_zone_max_m": dead_zone_max_m, "safe_max_m": safe_max_m,
            "footprint_samples": footprint, "min_fault_separation_m": min_fault_separation_m}


def classify_fault_location(location_m: float, valid_range: dict) -> str:
    """Labels a fault location as 'dead_zone', 'valid', or 'beyond_range'."""
    if location_m < valid_range["dead_zone_max_m"]:
        return "dead_zone"
    if location_m > valid_range["safe_max_m"]:
        return "beyond_range"
    return "valid"


def isolated_insertion_delay(x: np.ndarray, fs: float, cable: CableParams,
                              fault: FaultParams, search_window_samples: int = 40) -> dict:
    """
    Ground-truth correctness check: measures where the PROPAGATION MODEL
    ITSELF inserts a reflection, with the reflection isolated (no direct
    pulse, no noise) so a full-trace detector's bias can't contaminate the
    result. This was confirmed (by ablation) to isolate simulator
    correctness from matched-filter/detector bias -- use THIS, not the
    full-trace matched filter, as the pass/fail criterion for simulator
    correctness. The full-trace matched filter remains useful, but only as
    a classical-detector baseline (see Test 2 / monte_carlo_snr_sweep),
    not as a simulator-correctness check.
    """
    reflected, tau = reflect_single_fault(x, fs, cable, fault, record_len=len(x))
    expected_delay = tau * fs

    envelope, zero_lag_idx = _envelope_matched_filter(reflected, x)
    lo = int(zero_lag_idx + expected_delay - search_window_samples)
    hi = int(zero_lag_idx + expected_delay + search_window_samples)
    lo, hi = max(lo, 0), min(hi, len(envelope))

    if hi <= lo:
        return {"expected_delay_samples": expected_delay, "measured_delay_samples": None,
                "error_samples": None, "peak_amplitude": None}

    local_peak_idx = np.argmax(envelope[lo:hi]) + lo
    measured_delay = local_peak_idx - zero_lag_idx
    return {
        "expected_delay_samples": expected_delay,
        "measured_delay_samples": measured_delay,
        "error_samples": measured_delay - expected_delay,
        "peak_amplitude": float(envelope[local_peak_idx]),
    }


def _isolated_delay_for_component(reflected: np.ndarray, x: np.ndarray, fs: float,
                                   expected_delay: float, search_window_samples: int = 40) -> dict:
    """Shared helper: localized envelope search for one isolated reflection component."""
    envelope, zero_lag_idx = _envelope_matched_filter(reflected, x)
    lo = int(zero_lag_idx + expected_delay - search_window_samples)
    hi = int(zero_lag_idx + expected_delay + search_window_samples)
    lo, hi = max(lo, 0), min(hi, len(envelope))
    if hi <= lo:
        return {"measured_delay_samples": None, "error_samples": None}
    peak_idx = np.argmax(envelope[lo:hi]) + lo
    measured = peak_idx - zero_lag_idx
    return {"measured_delay_samples": measured, "error_samples": measured - expected_delay}


def validate_two_fault_case(x: np.ndarray, fs: float, gamma: float = 0.3,
                             tolerance_samples: float = 1.0) -> dict:
    """
    Reproduces the BARC paper's two-simultaneous-fault case (damage at 9 m
    and 19 m, Fig. 22) -- but checks each fault's reflection IN ISOLATION
    (one at a time, not summed together). This validates that
    reflect_single_fault places each reflection correctly on its own.

    IMPORTANT: this does NOT validate that both faults are recoverable
    together in a real combined trace. With this simulator's excitation
    pulse (176-sample footprint) and only 121 samples of separation between
    9 m and 19 m, the two reflections physically overlap -- see
    validate_two_fault_case_combined for that check, and
    compute_valid_range()['min_fault_separation_m'] for the resolution
    limit this implies. A pass here only means the propagation math is
    right, not that these two specific fault locations are jointly
    resolvable with this pulse.
    """
    cable = CableParams(v_p=1.65e8)
    fault_9m = FaultParams(location_m=9.0, gamma=gamma)
    fault_19m = FaultParams(location_m=19.0, gamma=gamma)

    results = {}
    for name, fault in [("9m", fault_9m), ("19m", fault_19m)]:
        reflected, tau = reflect_single_fault(x, fs, cable, fault, record_len=len(x))
        expected_delay = tau * fs
        r = _isolated_delay_for_component(reflected, x, fs, expected_delay)
        r["expected_delay_samples"] = expected_delay
        r["pass"] = (r["error_samples"] is not None) and (abs(r["error_samples"]) <= tolerance_samples)
        results[name] = r

    all_pass = all(r["pass"] for r in results.values())
    return {"per_fault": results, "pass": all_pass}


def validate_two_fault_case_combined(x: np.ndarray, fs: float, gamma: float = 0.3,
                                      tolerance_samples: float = 1.0) -> dict:
    """
    The test that actually matters for whether two faults are jointly
    recoverable: generates ONE combined trace with both faults present
    (summed, as they would be in a real measurement) and checks each
    fault's delay in that combined trace. If separation < a pulse
    footprint, expect this to show real bias -- that's the resolution
    limit, not a bug (see compute_valid_range()['min_fault_separation_m']).
    """
    cable = CableParams(v_p=1.65e8)
    faults = [FaultParams(9.0, gamma), FaultParams(19.0, gamma)]
    y, label = generate_multi_fault_trace(x, fs, cable, faults=faults, snr_db=None)

    zero_lag_idx = len(x) - 1
    envelope, _ = _envelope_matched_filter(y, x)
    results = {}
    for f in label["faults"]:
        expected_delay = f["tau_expected_s"] * fs
        lo = int(zero_lag_idx + expected_delay - 40)
        hi = int(zero_lag_idx + expected_delay + 40)
        lo, hi = max(lo, 0), min(hi, len(envelope))
        peak_idx = np.argmax(envelope[lo:hi]) + lo
        measured = peak_idx - zero_lag_idx
        err = measured - expected_delay
        results[f"{f['location_m']}m"] = {
            "expected_delay_samples": expected_delay,
            "measured_delay_samples": measured,
            "error_samples": err,
            "pass": abs(err) <= tolerance_samples,
        }
    return {"per_fault": results, "pass": all(r["pass"] for r in results.values())}


def validate_cable_end_reflection(x: np.ndarray, fs: float, cable_length_m: float = 30.0,
                                   end_gamma: float = 1.0, tolerance_samples: float = 1.0) -> dict:
    """
    Validates the cable-end reflection: a healthy (no internal fault) cable
    of length cable_length_m, with an open-circuit (end_gamma=+1.0 default)
    termination. Expected end-reflection delay: tau_e = 2*L/v_p.
    """
    cable = CableParams(v_p=1.65e8)
    end_fault = FaultParams(location_m=cable_length_m, gamma=end_gamma)
    reflected, tau = reflect_single_fault(x, fs, cable, end_fault, record_len=len(x))
    expected_delay = tau * fs

    r = _isolated_delay_for_component(reflected, x, fs, expected_delay)
    r["expected_delay_samples"] = expected_delay
    r["pass"] = (r["error_samples"] is not None) and (abs(r["error_samples"]) <= tolerance_samples)
    return r


def distance_sweep_validation(x: np.ndarray, fs: float, cable: CableParams, gamma: float,
                               distances_m: list[float], snr_db: float | None = None,
                               tolerance_samples: int = 2) -> list[dict]:
    """
    Sweeps fault distance across the full valid range (at fixed gamma, SNR,
    v_p) and checks, for EACH distance:
      1. delay accuracy -- measured vs expected round-trip delay, via a
         localized (ground-truth-aware) envelope search, same method as
         Test 1, just repeated across many distances instead of one.
      2. reflection amplitude -- should decrease smoothly/monotonically
         with distance under the attenuation model. Oscillations, sudden
         jumps, or non-monotonic behavior would indicate an indexing or
         interpolation bug that a single 19 m test point can't catch
         (e.g. near-zero distances overlapping the direct pulse, or
         distances near the edge of the record where the reflected copy
         gets truncated).

    Default snr_db=None (noiseless) is deliberate: this isolates simulator
    physics/indexing correctness from detector noise-robustness, which is
    Test 2's job, not this one's.
    """
    results = []
    zero_lag_idx = len(x) - 1
    max_delay_samples = len(x) - 1  # record length bound (see load_excitation record size)

    for d in distances_m:
        fault = FaultParams(location_m=d, gamma=gamma)
        rng = np.random.default_rng(2000 + int(d * 100)) if snr_db is not None else None
        y, label = generate_single_fault_trace(x, fs, cable, fault, snr_db=snr_db, rng=rng)

        expected_delay_samples = label["tau_expected_s"] * fs
        envelope, _ = _envelope_matched_filter(y, x)

        lo = int(zero_lag_idx + expected_delay_samples - 20)
        hi = int(zero_lag_idx + expected_delay_samples + 20)
        lo, hi = max(lo, 0), min(hi, len(envelope))

        if hi <= lo:
            # expected delay falls outside the record entirely -- flag, don't silently skip
            results.append({
                "distance_m": d, "expected_delay_samples": expected_delay_samples,
                "measured_delay_samples": None, "error_samples": None,
                "peak_amplitude": None, "in_record": False, "pass": False,
            })
            continue

        local_peak_idx = np.argmax(envelope[lo:hi]) + lo
        measured_delay_samples = local_peak_idx - zero_lag_idx
        peak_amplitude = envelope[local_peak_idx]

        results.append({
            "distance_m": d,
            "expected_delay_samples": expected_delay_samples,
            "measured_delay_samples": measured_delay_samples,
            "error_samples": measured_delay_samples - expected_delay_samples,
            "peak_amplitude": float(peak_amplitude),
            "in_record": True,
            "pass": abs(measured_delay_samples - expected_delay_samples) <= tolerance_samples,
        })

    return results


def _plot_distance_sweep(results: list[dict], plot_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results if r["in_record"]]
    distances = [r["distance_m"] for r in valid]
    errors = [r["error_samples"] for r in valid]
    amplitudes = [r["peak_amplitude"] for r in valid]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    axes[0].axhline(0, color="gray", linewidth=0.8)
    axes[0].plot(distances, errors, "o-", color="crimson")
    axes[0].set_ylabel("Delay error (samples)")
    axes[0].set_title("Delay accuracy across fault distance")

    axes[1].plot(distances, amplitudes, "o-", color="steelblue")
    axes[1].set_xlabel("Fault distance (m)")
    axes[1].set_ylabel("Reflection peak amplitude")
    axes[1].set_title("Attenuation model: reflection amplitude vs distance (should decay smoothly)")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)


def _check_monotonic_decay(results: list[dict], tolerance_frac: float = 0.02) -> dict:
    """
    Flags non-monotonic amplitude behavior. A small tolerance_frac allows for
    floating-point noise, not for real jumps -- a genuine bug should show up
    as a violation well above this threshold.
    """
    valid = [r for r in results if r["in_record"]]
    amps = np.array([r["peak_amplitude"] for r in valid])
    dists = np.array([r["distance_m"] for r in valid])
    diffs = np.diff(amps)
    violations = np.where(diffs > tolerance_frac * amps[:-1])[0]
    return {
        "monotonic": len(violations) == 0,
        "n_violations": len(violations),
        "violation_distances_m": dists[violations + 1].tolist() if len(violations) else [],
    }


def _plot_validation(x: np.ndarray, y: np.ndarray, result: dict, plot_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t_ns = np.arange(len(y))  # 1 ns/sample at fs=1GHz assumption for display only

    axes[0].plot(t_ns, x, color="steelblue")
    axes[0].set_title("Excitation (direct pulse) template")
    axes[0].set_ylabel("Volts")

    axes[1].plot(t_ns, y, color="darkorange")
    axes[1].axvline(result["expected_delay_samples"], color="green", linestyle="--",
                     label=f"expected delay ({result['expected_delay_samples']:.0f} samples)")
    axes[1].axvline(result["measured_delay_samples"], color="red", linestyle=":",
                     label=f"measured delay ({result['measured_delay_samples']:.0f} samples)")
    axes[1].set_title("Synthetic trace: 19 m single-fault case")
    axes[1].set_xlabel("Sample index (ns @ 1 GHz)")
    axes[1].set_ylabel("Volts")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Manual run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x, fs = load_excitation(
        data_path="25MHz1V (1).txt",
        header_path="25MHz1V_asciiheader (1).txt",
    )
    print(f"Loaded excitation: {len(x)} samples @ fs={fs:.3g} Hz")

    cable = CableParams(v_p=1.65e8)
    fault = FaultParams(location_m=19.0, gamma=0.3)

    # TEST 1: simulation correctness (ground-truth-aware, localized search)
    rng = np.random.default_rng(42)
    corr_result = validate_simulation_correctness(x, fs, cable, fault, snr_db=30, rng=rng)
    print("\n--- TEST 1: simulation correctness (19 m case) ---")
    for k, v in corr_result.items():
        print(f"{k}: {v}")

    y_plot, _ = generate_single_fault_trace(x, fs, cable, fault, snr_db=30,
                                             rng=np.random.default_rng(42))
    _plot_validation(x, y_plot, corr_result, "./validation_19m.png")

    # TEST 1b: distance sweep (delay accuracy + attenuation monotonicity
    # across the full valid range, not just the single 19 m point)
    print("\n--- TEST 1b: distance sweep (noiseless, gamma=0.3) ---")
    max_valid_m = cable.v_p * (len(x) - 1) / 1e9 / 2  # record-length bound, in meters
    distances = list(np.arange(1, max_valid_m - 2, 5))  # 1m .. near-max, 5m steps, margin at top
    sweep_results = distance_sweep_validation(x, fs, cable, gamma=0.3, distances_m=distances)

    n_pass = sum(r["pass"] for r in sweep_results)
    print(f"delay accuracy: {n_pass}/{len(sweep_results)} distances within tolerance")
    for r in sweep_results:
        if not r["pass"]:
            print(f"  FAIL at {r['distance_m']:.1f} m: {r}")

    decay_check = _check_monotonic_decay(sweep_results)
    print(f"attenuation monotonic decay: {decay_check['monotonic']} "
          f"({decay_check['n_violations']} violations at {decay_check['violation_distances_m']})")

    _plot_distance_sweep(sweep_results, "./distance_sweep.png")

    # TEST 2: detection robustness (blind search, Monte Carlo SNR sweep)
    print("\n--- TEST 2: detection robustness (blind, Monte Carlo) ---")
    sweep = monte_carlo_snr_sweep(
        x, fs, cable, fault,
        snr_values_db=[40, 30, 20, 15, 10, 5],
        n_trials=200,  # bump to 1000 for the final characterization before Bucket 3
    )
    print(f"{'SNR(dB)':>8} {'trials':>7} {'det.rate':>9} {'mean|err|':>11} {'p95|err|':>9}")
    for r in sweep:
        print(f"{r['snr_db']:>8} {r['n_trials']:>7} {r['detection_rate']:>9.2f} "
              f"{r['mean_abs_error_samples']:>11.2f} {r['p95_abs_error_samples']:>9.2f}")