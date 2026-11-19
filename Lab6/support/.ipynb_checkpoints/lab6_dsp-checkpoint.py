#!/usr/bin/env python3
"""Reference mathematics and waveform processing for ECE 5377/7377 Lab 6."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np


FS = 1_000_000.0
CENTER_FREQUENCY_HZ = 2_450_000_000.0
PERIOD_SAMPLES = 65_536
REPEAT_COUNT = 16
WARMUP_PERIODS = 1
ACQUISITION_LEAD_SAMPLES = 2_000_000
CAPTURE_SECONDS = 6.0
CAPTURE_SAMPLES = int(FS * CAPTURE_SECONDS)
DIGITAL_PEAK = 0.10

GUARD0 = slice(0, 2_048)
VERIFY = slice(2_048, 6_144)
GUARD1 = slice(6_144, 8_192)
PILOT0 = slice(8_192, 16_384)
GUARD2 = slice(16_384, 18_432)
PILOT1 = slice(18_432, 26_624)
GUARD3 = slice(26_624, 28_672)
SIMO_TRAINING = slice(28_672, 32_768)
SIMO_PAYLOAD = slice(32_768, 49_152)
GUARD4 = slice(49_152, PERIOD_SAMPLES)

VERIFY_FREQUENCIES_HZ = (-100_000.0, 100_000.0)
MIMO_PILOT_FREQUENCY_HZ = 50_000.0
SIMO_OFFSET_HZ = -150_000.0
SAMPLES_PER_SYMBOL = 8
SIMO_TRAINING_SYMBOLS = (SIMO_TRAINING.stop - SIMO_TRAINING.start) // SAMPLES_PER_SYMBOL
SIMO_PAYLOAD_SYMBOLS = (SIMO_PAYLOAD.stop - SIMO_PAYLOAD.start) // SAMPLES_PER_SYMBOL
SIMO_PAYLOAD_BITS = 2 * SIMO_PAYLOAD_SYMBOLS


@dataclass(frozen=True)
class Lab6Waveform:
    tx0_period: np.ndarray
    tx1_period: np.ndarray
    training_symbols: np.ndarray
    payload_symbols: np.ndarray
    payload_bits: np.ndarray


def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Map pairs 00, 01, 11, 10 to unit-energy Gray QPSK."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if b.size % 2:
        raise ValueError("QPSK requires an even number of bits.")
    if np.any(b > 1):
        raise ValueError("QPSK input must contain only zero and one.")
    pairs = b.reshape(-1, 2)
    i = 1.0 - 2.0 * pairs[:, 1]
    q = 1.0 - 2.0 * pairs[:, 0]
    return ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex64)


def qpsk_detect(symbols: np.ndarray) -> np.ndarray:
    z = np.asarray(symbols).reshape(-1)
    return np.column_stack((z.imag < 0, z.real < 0)).astype(np.uint8).reshape(-1)


def rms_evm(estimate: np.ndarray, reference: np.ndarray) -> float:
    estimate = np.asarray(estimate)
    reference = np.asarray(reference)
    return float(
        np.sqrt(np.mean(np.abs(estimate - reference) ** 2) / np.mean(np.abs(reference) ** 2))
    )


def tone(length: int, frequency_hz: float, start_sample: int = 0, amplitude: float = DIGITAL_PEAK) -> np.ndarray:
    n = np.arange(start_sample, start_sample + length, dtype=np.float64)
    return (amplitude * np.exp(1j * 2.0 * np.pi * frequency_hz * n / FS)).astype(np.complex64)


def build_waveform(seed: int = 7611) -> Lab6Waveform:
    rng = np.random.default_rng(seed)
    training_bits = rng.integers(0, 2, 2 * SIMO_TRAINING_SYMBOLS, dtype=np.uint8)
    payload_bits = rng.integers(0, 2, SIMO_PAYLOAD_BITS, dtype=np.uint8)
    training_symbols = qpsk_map(training_bits)
    payload_symbols = qpsk_map(payload_bits)

    tx0 = np.zeros(PERIOD_SAMPLES, dtype=np.complex64)
    tx1 = np.zeros(PERIOD_SAMPLES, dtype=np.complex64)
    tx0[VERIFY] = tone(VERIFY.stop - VERIFY.start, VERIFY_FREQUENCIES_HZ[0], VERIFY.start)
    tx1[VERIFY] = tone(VERIFY.stop - VERIFY.start, VERIFY_FREQUENCIES_HZ[1], VERIFY.start)
    tx0[PILOT0] = tone(PILOT0.stop - PILOT0.start, MIMO_PILOT_FREQUENCY_HZ, PILOT0.start)
    tx1[PILOT1] = tone(PILOT1.stop - PILOT1.start, MIMO_PILOT_FREQUENCY_HZ, PILOT1.start)

    training_samples = np.repeat(training_symbols, SAMPLES_PER_SYMBOL)
    payload_samples = np.repeat(payload_symbols, SAMPLES_PER_SYMBOL)
    tx0[SIMO_TRAINING] = DIGITAL_PEAK * training_samples * np.exp(
        1j * 2.0 * np.pi * SIMO_OFFSET_HZ * np.arange(SIMO_TRAINING.start, SIMO_TRAINING.stop) / FS
    )
    tx0[SIMO_PAYLOAD] = DIGITAL_PEAK * payload_samples * np.exp(
        1j * 2.0 * np.pi * SIMO_OFFSET_HZ * np.arange(SIMO_PAYLOAD.start, SIMO_PAYLOAD.stop) / FS
    )
    return Lab6Waveform(tx0, tx1, training_symbols, payload_symbols, payload_bits)


def make_transport(period: np.ndarray) -> np.ndarray:
    period = np.asarray(period, dtype=np.complex64)
    return np.concatenate(
        (
            np.zeros(ACQUISITION_LEAD_SAMPLES, dtype=np.complex64),
            np.tile(period, REPEAT_COUNT),
        )
    )


def waveform_metadata(waveform: Lab6Waveform) -> dict[str, object]:
    tx0 = make_transport(waveform.tx0_period)
    tx1 = make_transport(waveform.tx1_period)
    return {
        "sample_rate_sps": FS,
        "center_frequency_hz": CENTER_FREQUENCY_HZ,
        "period_samples": PERIOD_SAMPLES,
        "repeat_count": REPEAT_COUNT,
        "warmup_periods": WARMUP_PERIODS,
        "measurement_periods": REPEAT_COUNT - WARMUP_PERIODS,
        "acquisition_lead_samples": ACQUISITION_LEAD_SAMPLES,
        "transport_samples": int(tx0.size),
        "transport_bytes_per_channel": int(tx0.nbytes),
        "capture_seconds": CAPTURE_SECONDS,
        "capture_samples_per_channel": CAPTURE_SAMPLES,
        "capture_bytes_per_channel": 8 * CAPTURE_SAMPLES,
        "digital_peak": DIGITAL_PEAK,
        "payload_bits_per_period": SIMO_PAYLOAD_BITS,
        "tx0_period_sha256": hashlib.sha256(waveform.tx0_period.tobytes()).hexdigest(),
        "tx1_period_sha256": hashlib.sha256(waveform.tx1_period.tobytes()).hexdigest(),
        "tx0_file_sha256": hashlib.sha256(tx0.tobytes()).hexdigest(),
        "tx1_file_sha256": hashlib.sha256(tx1.tobytes()).hexdigest(),
    }


def ls_channel_estimate(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Estimate H in Y = H X + N for X[nt, samples], Y[nr, samples]."""
    x = np.asarray(x, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("X and Y must be two-dimensional with the same sample count.")
    return y @ x.conj().T @ np.linalg.inv(x @ x.conj().T)


def mimo_capacity(h: np.ndarray, total_snr_linear: float) -> float:
    """Equal-total-power MIMO spectral efficiency in bit/s/Hz."""
    h = np.asarray(h, dtype=np.complex128)
    nr, nt = h.shape
    gram = h @ h.conj().T
    sign, logdet = np.linalg.slogdet(np.eye(nr) + (total_snr_linear / nt) * gram)
    if sign <= 0:
        raise ValueError("Capacity matrix must be positive definite.")
    return float(logdet / np.log(2.0))


def siso_capacity(h: complex, snr_linear: float) -> float:
    return float(np.log2(1.0 + snr_linear * np.abs(h) ** 2))


def simulate_mimo_capacity(
    snr_db_values: np.ndarray = np.arange(0.0, 26.0, 5.0),
    realizations: int = 5_000,
    seed: int = 7601,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    h = (rng.standard_normal((realizations, 2, 2)) + 1j * rng.standard_normal((realizations, 2, 2))) / np.sqrt(2.0)
    pilot = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    estimation_nmse = []
    capacities = []
    siso = []
    singular_ratio = []
    for snr_db in np.asarray(snr_db_values, dtype=float):
        rho = 10.0 ** (snr_db / 10.0)
        noise_variance = 1.0 / rho
        noise = np.sqrt(noise_variance / 2.0) * (
            rng.standard_normal((realizations, 2, 2)) + 1j * rng.standard_normal((realizations, 2, 2))
        )
        y = h @ pilot[None, :, :] + noise
        hhat = y @ pilot.conj().T @ np.linalg.inv(pilot @ pilot.conj().T)
        estimation_nmse.append(float(np.sum(np.abs(hhat - h) ** 2) / np.sum(np.abs(h) ** 2)))
        capacities.append(float(np.mean([mimo_capacity(one, rho) for one in hhat])))
        siso.append(
            [float(np.mean(np.log2(1.0 + rho * np.abs(hhat[:, r, t]) ** 2))) for r in range(2) for t in range(2)]
        )
        sv = np.linalg.svd(hhat, compute_uv=False)
        singular_ratio.append(float(np.mean(sv[:, 0] / np.maximum(sv[:, 1], 1e-12))))
    return {
        "snr_db": np.asarray(snr_db_values, dtype=float),
        "mimo_capacity_bps_hz": np.asarray(capacities),
        "siso_capacity_bps_hz": np.asarray(siso),
        "channel_nmse": np.asarray(estimation_nmse),
        "mean_singular_value_ratio": np.asarray(singular_ratio),
        "realizations": realizations,
    }


def combine_diversity(y: np.ndarray, h: np.ndarray, method: str) -> np.ndarray:
    """Combine y[L,N] with perfectly known flat-fading coefficients h[L,N]."""
    y = np.asarray(y)
    h = np.asarray(h)
    if y.shape != h.shape:
        raise ValueError("Y and H must have the same L by N shape.")
    if method == "sc":
        branch = np.argmax(np.abs(h) ** 2, axis=0)
        n = np.arange(y.shape[1])
        return y[branch, n] / h[branch, n]
    if method == "egc":
        weights = np.conj(h) / np.maximum(np.abs(h), 1e-12)
        return np.sum(weights * y, axis=0) / np.maximum(np.sum(np.abs(h), axis=0), 1e-12)
    if method == "mrc":
        return np.sum(np.conj(h) * y, axis=0) / np.maximum(np.sum(np.abs(h) ** 2, axis=0), 1e-12)
    raise ValueError(f"Unknown combining method: {method}")


def simulate_diversity(
    snr_db_values: np.ndarray = np.arange(0.0, 21.0, 4.0),
    branch_counts: tuple[int, ...] = (1, 2, 3, 4),
    bits_per_point: int = 240_000,
    seed: int = 7602,
) -> dict[str, object]:
    if bits_per_point % 2:
        raise ValueError("bits_per_point must be even.")
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, bits_per_point, dtype=np.uint8)
    symbols = qpsk_map(bits).astype(np.complex128)
    methods = ("sc", "egc", "mrc")
    ber = {method: np.zeros((len(branch_counts), len(snr_db_values))) for method in methods}
    errors = {method: np.zeros((len(branch_counts), len(snr_db_values)), dtype=int) for method in methods}
    for li, branches in enumerate(branch_counts):
        h = (rng.standard_normal((branches, symbols.size)) + 1j * rng.standard_normal((branches, symbols.size))) / np.sqrt(2.0)
        base_noise = (rng.standard_normal(h.shape) + 1j * rng.standard_normal(h.shape)) / np.sqrt(2.0)
        for si, snr_db in enumerate(np.asarray(snr_db_values, dtype=float)):
            gamma = 10.0 ** (snr_db / 10.0)
            y = h * symbols[None, :] + base_noise / np.sqrt(gamma)
            for method in methods:
                combined = combine_diversity(y, h, method)
                count = int(np.count_nonzero(qpsk_detect(combined) != bits))
                errors[method][li, si] = count
                ber[method][li, si] = count / bits.size
    return {
        "snr_db": np.asarray(snr_db_values, dtype=float),
        "branch_counts": np.asarray(branch_counts, dtype=int),
        "bits_per_point": bits_per_point,
        "ber": ber,
        "errors": errors,
    }


def _block_power(values: np.ndarray, block: int) -> np.ndarray:
    count = values.size // block
    return np.mean(np.abs(values[: count * block].reshape(count, block)) ** 2, axis=1)


def _fft_valid_correlation(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    full_length = values.size + template.size - 1
    fft_length = 1 << (full_length - 1).bit_length()
    full = np.fft.irfft(
        np.fft.rfft(values, fft_length) * np.fft.rfft(template[::-1], fft_length),
        fft_length,
    )[:full_length]
    return full[template.size - 1 : values.size]


def _separated_peaks(metric: np.ndarray, distance: int, threshold: float) -> np.ndarray:
    local = np.flatnonzero(
        (metric[1:-1] >= metric[:-2])
        & (metric[1:-1] > metric[2:])
        & (metric[1:-1] >= threshold)
    ) + 1
    selected: list[int] = []
    for candidate in local[np.argsort(metric[local])[::-1]]:
        if all(abs(int(candidate) - previous) >= distance for previous in selected):
            selected.append(int(candidate))
    return np.asarray(sorted(selected), dtype=int)


def detect_period_starts(rx0: np.ndarray, rx1: np.ndarray, waveform: Lab6Waveform) -> tuple[np.ndarray, np.ndarray]:
    """Detect uncropped periods from the known on/off envelope."""
    block = 64
    observed = _block_power(rx0, block) + _block_power(rx1, block)
    template = _block_power(waveform.tx0_period, block) + _block_power(waveform.tx1_period, block)
    template = template - np.mean(template)
    metric = _fft_valid_correlation(observed - np.median(observed), template)
    peaks = _separated_peaks(metric, int(0.85 * PERIOD_SAMPLES / block), 0.15 * np.max(metric))
    if peaks.size > REPEAT_COUNT:
        peaks = peaks[np.argsort(metric[peaks])[-REPEAT_COUNT:]]
    starts = np.sort(peaks.astype(int) * block)
    return starts, metric


def estimate_cfo_from_tone(values: np.ndarray, known: np.ndarray) -> float:
    z = np.asarray(values) * np.conj(np.asarray(known))
    return float(FS * np.angle(np.sum(z[1:] * np.conj(z[:-1]))) / (2.0 * np.pi))


def _tone_channel(values: np.ndarray, known: np.ndarray) -> complex:
    return complex(np.vdot(known, values) / np.vdot(known, known))


def analyze_hardware_frame(
    frame_rx0: np.ndarray,
    frame_rx1: np.ndarray,
    waveform: Lab6Waveform,
    noise_variances: np.ndarray,
) -> dict[str, object]:
    rx = np.stack((frame_rx0, frame_rx1))
    edge = 256
    pilot0_known = waveform.tx0_period[PILOT0][edge:-edge]
    pilot0_rx = rx[:, PILOT0][:, edge:-edge]
    best_branch = int(np.argmax(np.mean(np.abs(pilot0_rx) ** 2, axis=1)))
    coarse_cfo_hz = estimate_cfo_from_tone(pilot0_rx[best_branch], pilot0_known)
    n = np.arange(PERIOD_SAMPLES)
    coarse_corrected = rx * np.exp(-1j * 2.0 * np.pi * coarse_cfo_hz * n / FS)[None, :]

    nominal_training = SIMO_TRAINING.start
    search = np.arange(nominal_training - 96, nominal_training + 97)
    known_training = DIGITAL_PEAK * np.repeat(waveform.training_symbols, SAMPLES_PER_SYMBOL)
    known_training = known_training * np.exp(
        1j * 2.0 * np.pi * SIMO_OFFSET_HZ * np.arange(SIMO_TRAINING.start, SIMO_TRAINING.stop) / FS
    )
    correlations = []
    for start in search:
        segment = coarse_corrected[best_branch, start : start + known_training.size]
        correlations.append(np.abs(np.vdot(known_training, segment)))
    timing_shift = int(search[int(np.argmax(correlations))] - nominal_training)
    training_slice = slice(SIMO_TRAINING.start + timing_shift, SIMO_TRAINING.stop + timing_shift)
    payload_slice = slice(SIMO_PAYLOAD.start + timing_shift, SIMO_PAYLOAD.stop + timing_shift)
    known_training_shifted = DIGITAL_PEAK * np.repeat(waveform.training_symbols, SAMPLES_PER_SYMBOL)
    known_training_shifted = known_training_shifted * np.exp(
        1j * 2.0 * np.pi * SIMO_OFFSET_HZ * np.arange(training_slice.start, training_slice.stop) / FS
    )

    # The verification pilot provides a robust coarse CFO estimate. Real N310
    # captures can retain tens of hertz of error, which rotates a long QPSK
    # payload enough to produce an apparent BER near 0.5. Refine the estimate
    # with the known SIMO training field, then apply the combined correction to
    # the complete frame before estimating either channel.
    training_baseband = coarse_corrected[best_branch, training_slice] * np.exp(
        -1j * 2.0 * np.pi * SIMO_OFFSET_HZ * np.arange(training_slice.start, training_slice.stop) / FS
    )
    training_symbols_rx = training_baseband.reshape(-1, SAMPLES_PER_SYMBOL).mean(axis=1)
    training_phase_error = np.unwrap(
        np.angle(training_symbols_rx * np.conj(DIGITAL_PEAK * waveform.training_symbols))
    )
    phase_slope = np.polyfit(np.arange(training_phase_error.size), training_phase_error, 1)[0]
    residual_cfo_hz = float(phase_slope * FS / (2.0 * np.pi * SAMPLES_PER_SYMBOL))
    cfo_hz = coarse_cfo_hz + residual_cfo_hz
    corrected = rx * np.exp(-1j * 2.0 * np.pi * cfo_hz * n / FS)[None, :]

    h = np.empty((2, 2), dtype=np.complex128)
    known0 = waveform.tx0_period[PILOT0][edge:-edge]
    known1 = waveform.tx1_period[PILOT1][edge:-edge]
    for branch in range(2):
        h[branch, 0] = _tone_channel(corrected[branch, PILOT0][edge:-edge], known0)
        h[branch, 1] = _tone_channel(corrected[branch, PILOT1][edge:-edge], known1)

    pilot_energy = DIGITAL_PEAK**2
    total_snr_reference = 2.0 * pilot_energy / float(np.mean(noise_variances))
    capacity = mimo_capacity(h, total_snr_reference)
    siso = np.array(
        [siso_capacity(h[r, t], 2.0 * pilot_energy / noise_variances[r]) for r in range(2) for t in range(2)]
    )
    singular_values = np.linalg.svd(h, compute_uv=False)

    h_simo = np.array(
        [_tone_channel(corrected[branch, training_slice], known_training_shifted) for branch in range(2)]
    )

    payload = corrected[:, payload_slice]
    absolute = np.arange(payload_slice.start, payload_slice.stop)
    payload = payload * np.exp(-1j * 2.0 * np.pi * SIMO_OFFSET_HZ * absolute / FS)[None, :]
    payload_symbols_rx = payload.reshape(2, -1, SAMPLES_PER_SYMBOL).mean(axis=2)
    h_matrix = np.repeat(h_simo[:, None], payload_symbols_rx.shape[1], axis=1)
    methods = {}
    for method in ("sc", "egc", "mrc"):
        combined = combine_diversity(payload_symbols_rx, h_matrix, method) / DIGITAL_PEAK
        bits = qpsk_detect(combined)
        errors = int(np.count_nonzero(bits != waveform.payload_bits))
        methods[method] = {
            "symbols": combined,
            "bit_errors": errors,
            "ber": errors / waveform.payload_bits.size,
            "evm_rms": rms_evm(combined, waveform.payload_symbols),
        }
    branches = []
    for branch in range(2):
        equalized = payload_symbols_rx[branch] / (DIGITAL_PEAK * h_simo[branch])
        errors = int(np.count_nonzero(qpsk_detect(equalized) != waveform.payload_bits))
        branches.append(
            {
                "symbols": equalized,
                "bit_errors": errors,
                "ber": errors / waveform.payload_bits.size,
                "evm_rms": rms_evm(equalized, waveform.payload_symbols),
                "channel": h_simo[branch],
            }
        )
    return {
        "cfo_hz": cfo_hz,
        "coarse_cfo_hz": coarse_cfo_hz,
        "residual_cfo_hz": residual_cfo_hz,
        "mimo_channel": h,
        "mimo_capacity_bps_hz": capacity,
        "siso_capacity_bps_hz": siso,
        "singular_values": singular_values,
        "condition_number": float(singular_values[0] / max(singular_values[1], 1e-12)),
        "timing_shift_samples": timing_shift,
        "simo_channel": h_simo,
        "payload_symbols_rx": payload_symbols_rx,
        "branches": branches,
        "combiners": methods,
        "timing_search": search,
        "timing_correlation": np.asarray(correlations),
    }


def analyze_hardware_capture(
    rx0: np.ndarray,
    rx1: np.ndarray,
    noise0: np.ndarray,
    noise1: np.ndarray,
    waveform: Lab6Waveform,
) -> dict[str, object]:
    starts, metric = detect_period_starts(rx0, rx1, waveform)
    noise_variances = np.array(
        [
            np.mean(np.abs(noise0 - np.mean(noise0)) ** 2),
            np.mean(np.abs(noise1 - np.mean(noise1)) ** 2),
        ],
        dtype=float,
    )
    frames = []
    for start in starts:
        if start < 0 or start + PERIOD_SAMPLES > rx0.size:
            continue
        frames.append(
            analyze_hardware_frame(
                rx0[start : start + PERIOD_SAMPLES],
                rx1[start : start + PERIOD_SAMPLES],
                waveform,
                noise_variances,
            )
        )
    analyzed = frames[WARMUP_PERIODS:] if len(frames) > WARMUP_PERIODS else frames
    return {
        "starts": starts,
        "metric": metric,
        "noise_variances": noise_variances,
        "frames": frames,
        "analyzed_frames": analyzed,
    }


def load_complex64(path: Path, expected_samples: int | None = None) -> np.ndarray:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    values = np.fromfile(path, dtype=np.complex64)
    if expected_samples is not None and values.size != expected_samples:
        raise ValueError(f"{path.name} has {values.size} samples, expected {expected_samples}.")
    return values
