
from __future__ import annotations

import numpy as np


def _max_len_bits(degree: int) -> np.ndarray:
    """Generate one maximal-length Fibonacci-LFSR period from an all-one seed."""
    n = int(degree)
    tap_map = {7: (0, 1), 10: (0, 3)}
    if n not in tap_map:
        raise ValueError("this lab defines degree-7 and degree-10 sequences")
    state = (1 << n) - 1
    bits = np.empty((1 << n) - 1, dtype=np.uint8)
    for index in range(bits.size):
        bits[index] = state & 1
        feedback = 0
        for tap in tap_map[n]:
            feedback ^= (state >> tap) & 1
        state = (state >> 1) | (feedback << (n - 1))
    if state != (1 << n) - 1:
        raise RuntimeError("LFSR did not return to its seed after one period")
    return bits


def _fft_convolve_valid(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Complex FFT convolution restricted to positions where the kernel fits."""
    x = np.asarray(values, dtype=np.complex128).reshape(-1)
    h = np.asarray(kernel, dtype=np.complex128).reshape(-1)
    if h.size == 0 or x.size < h.size:
        raise ValueError("kernel must be nonempty and fit within values")
    full_size = x.size + h.size - 1
    fft_size = 1 << (full_size - 1).bit_length()
    full = np.fft.ifft(np.fft.fft(x, fft_size) * np.fft.fft(h, fft_size))[:full_size]
    return full[h.size - 1 : x.size]


def _peak_valid_correlation(
    values: np.ndarray, template: np.ndarray, *, chunk_starts: int = 1_000_000
) -> tuple[int, float]:
    """Find the largest valid correlation without one capture-sized FFT."""
    x = np.asarray(values, dtype=np.complex128).reshape(-1)
    t = np.asarray(template, dtype=np.complex128).reshape(-1)
    if x.size < t.size:
        raise ValueError("template must fit within values")
    kernel = np.conj(t[::-1])
    candidate_count = x.size - t.size + 1
    best_start = 0
    best_magnitude = -np.inf
    for first in range(0, candidate_count, int(chunk_starts)):
        count = min(int(chunk_starts), candidate_count - first)
        window = x[first : first + count + t.size - 1]
        correlation = _fft_convolve_valid(window, kernel)[:count]
        local = int(np.argmax(np.abs(correlation)))
        magnitude = float(abs(correlation[local]))
        if magnitude > best_magnitude:
            best_start = first + local
            best_magnitude = magnitude
    return best_start, best_magnitude


def gray_decode(values: np.ndarray) -> np.ndarray:
    """Decode nonnegative binary-reflected Gray integers."""
    gray = np.asarray(values, dtype=np.int64)
    decoded = gray.copy()
    shifted = gray.copy()
    while np.any(shifted):
        shifted >>= 1
        decoded ^= shifted
    return decoded


def psk_constellation(order: int) -> np.ndarray:
    """Return a unit-energy, cyclic Gray-labeled PSK constellation."""
    M = int(order)
    if M < 2 or M & (M - 1):
        raise ValueError("order must be a power of two")
    phase_index = gray_decode(np.arange(M))
    return np.exp(1j * 2 * np.pi * phase_index / M)


def nearest_labels(samples: np.ndarray, constellation: np.ndarray) -> np.ndarray:
    """Minimum-distance hard decisions."""
    r = np.asarray(samples, dtype=np.complex128).reshape(-1)
    c = np.asarray(constellation, dtype=np.complex128).reshape(-1)
    return np.argmin(np.abs(r[:, None] - c[None, :]) ** 2, axis=1)


def labels_to_bits(labels: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    """Convert integer labels to an MSB-first flat bit array."""
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    shifts = np.arange(int(bits_per_symbol) - 1, -1, -1)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def rrc_taps(beta: float, samples_per_symbol: int, span_symbols: int) -> np.ndarray:
    """Generate a unit-energy odd-length root-raised-cosine FIR."""
    sps = int(samples_per_symbol)
    span = int(span_symbols)
    if not 0 <= beta <= 1 or sps < 2 or span < 2 or (span * sps) % 2:
        raise ValueError("invalid RRC parameters")
    t = np.arange(-span * sps / 2, span * sps / 2 + 1) / sps
    h = np.empty(t.size, dtype=float)
    for index, value in enumerate(t):
        if np.isclose(value, 0.0):
            h[index] = 1 + beta * (4 / np.pi - 1)
        elif beta > 0 and np.isclose(abs(value), 1 / (4 * beta)):
            h[index] = (beta / np.sqrt(2)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            numerator = (
                np.sin(np.pi * value * (1 - beta))
                + 4 * beta * value * np.cos(np.pi * value * (1 + beta))
            )
            denominator = np.pi * value * (1 - (4 * beta * value) ** 2)
            h[index] = numerator / denominator
    return h / np.sqrt(np.sum(h**2))


def pulse_shape(symbols: np.ndarray, taps: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    """Zero insert and filter a complex symbol array."""
    values = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    sps = int(samples_per_symbol)
    upsampled = np.zeros((values.size - 1) * sps + 1, dtype=np.complex128)
    upsampled[::sps] = values
    return np.convolve(upsampled, taps)


def timing_trial(
    order: int,
    ebn0_db: float,
    timing_error_samples: int,
    *,
    payload_symbols: int = 5000,
    samples_per_symbol: int = 8,
    seed: int = 404,
) -> dict[str, object]:
    """Simulate RRC PSK and recover the integer sampling phase from a preamble."""
    M = int(order)
    sps = int(samples_per_symbol)
    rng = np.random.default_rng(seed + M + 100 * int(timing_error_samples))
    constellation = psk_constellation(M)
    labels = rng.integers(0, M, int(payload_symbols), dtype=np.int64)
    payload = constellation[labels]
    preamble_bits = _max_len_bits(7)
    preamble = 2 * preamble_bits.astype(float) - 1
    all_symbols = np.concatenate((preamble, payload))
    taps = rrc_taps(0.35, sps, 10)
    tx = pulse_shape(all_symbols, taps, sps)
    delay = int(timing_error_samples)
    if delay < 0 or delay >= sps:
        raise ValueError("timing error must be in [0, samples_per_symbol)")
    delayed = np.concatenate((np.zeros(delay, dtype=complex), tx))
    bits_per_symbol = int(np.log2(M))
    ebn0 = 10 ** (float(ebn0_db) / 10)
    N0 = 1 / (bits_per_symbol * ebn0)
    noise = np.sqrt(N0 / 2) * (
        rng.standard_normal(delayed.size) + 1j * rng.standard_normal(delayed.size)
    )
    matched = np.convolve(delayed + noise, taps)
    cascade_delay = 2 * ((taps.size - 1) // 2)

    phase_metrics = np.empty(sps)
    for phase in range(sps):
        indices = cascade_delay + phase + np.arange(preamble.size) * sps
        received = matched[indices]
        phase_metrics[phase] = abs(np.vdot(preamble, received)) ** 2 / (
            np.vdot(preamble, preamble).real * np.vdot(received, received).real
        )
    estimated_phase = int(np.argmax(phase_metrics))

    payload_offset = preamble.size * sps
    before_indices = cascade_delay + payload_offset + np.arange(labels.size) * sps
    after_indices = cascade_delay + estimated_phase + payload_offset + np.arange(labels.size) * sps
    before = matched[before_indices]
    after = matched[after_indices]
    before_labels = nearest_labels(before, constellation)
    after_labels = nearest_labels(after, constellation)
    tx_bits = labels_to_bits(labels, bits_per_symbol)
    return {
        "order": M,
        "labels": labels,
        "constellation": constellation,
        "before": before,
        "after": after,
        "before_ber": float(np.mean(labels_to_bits(before_labels, bits_per_symbol) != tx_bits)),
        "after_ber": float(np.mean(labels_to_bits(after_labels, bits_per_symbol) != tx_bits)),
        "estimated_phase": estimated_phase,
        "true_phase": delay,
        "phase_metrics": phase_metrics,
        "rrc_taps": taps,
        "cascade_delay": cascade_delay,
    }


NFFT = 64
CP_LEN = 16
ACTIVE_SUBCARRIERS = np.r_[np.arange(-26, 0), np.arange(1, 27)]
PILOT_SUBCARRIERS = np.array([-21, -7, 7, 21])
DATA_SUBCARRIERS = np.array([k for k in ACTIVE_SUBCARRIERS if k not in PILOT_SUBCARRIERS])


def _qpsk_map(bits: np.ndarray) -> np.ndarray:
    pairs = np.asarray(bits, dtype=np.uint8).reshape(-1, 2)
    return ((1 - 2 * pairs[:, 0].astype(float)) + 1j * (1 - 2 * pairs[:, 1].astype(float))) / np.sqrt(2)


def _qpsk_bits(symbols: np.ndarray) -> np.ndarray:
    values = np.asarray(symbols)
    return np.column_stack((values.real < 0, values.imag < 0)).astype(np.uint8).reshape(-1)


def ofdm_phase_sweep(
    phase_degrees: tuple[int, ...] = (0, 30, 60, 90),
    *,
    frames: int = 1000,
    ebn0_db: float = 10.0,
    message: str = "hello world",
    seed: int = 4402,
) -> dict[str, object]:
    """Run a 64-point, 52-active-subcarrier 802.11-like OFDM phase study."""
    rng = np.random.default_rng(seed)
    message_bits = np.unpackbits(np.frombuffer(message.encode("ascii"), dtype=np.uint8))
    bits_per_frame = 2 * DATA_SUBCARRIERS.size
    frame_bits = np.resize(message_bits, bits_per_frame)
    bits = np.tile(frame_bits, (int(frames), 1))
    data_symbols = _qpsk_map(bits.reshape(-1)).reshape(int(frames), -1)
    grid = np.zeros((int(frames), NFFT), dtype=complex)
    grid[:, DATA_SUBCARRIERS % NFFT] = data_symbols
    grid[:, PILOT_SUBCARRIERS % NFFT] = 1
    body = np.fft.ifft(grid, axis=1) * np.sqrt(NFFT)
    tx = np.concatenate((body[:, -CP_LEN:], body), axis=1)
    N0 = 1 / (2 * 10 ** (float(ebn0_db) / 10))
    base_noise = np.sqrt(N0 / 2) * (
        rng.standard_normal(tx.shape) + 1j * rng.standard_normal(tx.shape)
    )
    rows: list[dict[str, object]] = []
    for phase_deg in phase_degrees:
        received = tx * np.exp(1j * np.deg2rad(phase_deg)) + base_noise
        frequency = np.fft.fft(received[:, CP_LEN:], axis=1) / np.sqrt(NFFT)
        raw_data = frequency[:, DATA_SUBCARRIERS % NFFT]
        raw_bits = _qpsk_bits(raw_data.reshape(-1)).reshape(int(frames), -1)
        pilot_values = frequency[:, PILOT_SUBCARRIERS % NFFT]
        phase_hat = np.angle(np.sum(pilot_values, axis=1))
        corrected = frequency * np.exp(-1j * phase_hat[:, None])
        corrected_data = corrected[:, DATA_SUBCARRIERS % NFFT]
        corrected_bits = _qpsk_bits(corrected_data.reshape(-1)).reshape(int(frames), -1)
        raw_errors = raw_bits != bits
        corrected_errors = corrected_bits != bits
        rows.append(
            {
                "phase_degrees": int(phase_deg),
                "ber": float(np.mean(raw_errors)),
                "fer": float(np.mean(np.any(raw_errors, axis=1))),
                "corrected_ber": float(np.mean(corrected_errors)),
                "corrected_fer": float(np.mean(np.any(corrected_errors, axis=1))),
                "estimated_phase_mean_degrees": float(np.rad2deg(np.angle(np.mean(np.exp(1j * phase_hat))))),
                "received_data": raw_data[:20].reshape(-1),
                "corrected_data": corrected_data[:20].reshape(-1),
            }
        )
    return {
        "message": message,
        "frames": int(frames),
        "ebn0_db": float(ebn0_db),
        "bits_per_frame": bits_per_frame,
        "rows": rows,
    }

def ofdm_grid(
    phase_degrees: tuple[int, ...] = (0, 30, 60, 90),
    *,
    frames: int = 1000,
    ebn0_db: float = 10.0,
    message: str = "hello world",
    seed: int = 4402,
) -> dict[str, object]:
    """Run a 64-point, 52-active-subcarrier 802.11-like OFDM phase study."""
    rng = np.random.default_rng(seed)
    message_bits = np.unpackbits(np.frombuffer(message.encode("ascii"), dtype=np.uint8))
    bits_per_frame = 2 * DATA_SUBCARRIERS.size
    frame_bits = np.resize(message_bits, bits_per_frame)
    bits = np.tile(frame_bits, (int(frames), 1))
    data_symbols = _qpsk_map(bits.reshape(-1)).reshape(int(frames), -1)
    grid = np.zeros((int(frames), NFFT), dtype=complex)
    grid[:, DATA_SUBCARRIERS % NFFT] = data_symbols
    grid[:, PILOT_SUBCARRIERS % NFFT] = 1

    return grid


def pn_sequence(degree: int = 10) -> np.ndarray:
    """Return one maximal-length BPSK sequence from SciPy's GLFSR."""
    bits = _max_len_bits(int(degree))
    chips = 2 * bits.astype(float) - 1
    expected = 2 ** int(degree) - 1
    if chips.size != expected:
        raise RuntimeError("unexpected maximal-length sequence")
    return chips.astype(np.complex128)


def build_pn_burst(
    *,
    sample_rate_sps: float = 1_250_000.0,
    baseband_frequency_hz: float = 200_000.0,
    acquisition_lead_samples: int = 2_500_000,
    guard_samples: int = 4096,
    peak: float = 0.08,
) -> dict[str, object]:
    """Build one PN period after a quiet lead-in and finite edge guards."""
    chips = pn_sequence(10)
    baseband = np.concatenate(
        (
            np.zeros(int(acquisition_lead_samples), dtype=complex),
            np.zeros(int(guard_samples), dtype=complex),
            peak * chips,
            np.zeros(int(guard_samples), dtype=complex),
        )
    )
    n = np.arange(baseband.size)
    shifted = baseband * np.exp(1j * 2 * np.pi * baseband_frequency_hz * n / sample_rate_sps)
    return {
        "chips": chips,
        "baseband": baseband,
        "shifted": shifted.astype(np.complex64),
        "acquisition_lead_samples": int(acquisition_lead_samples),
        "guard_samples": int(guard_samples),
        "sample_rate_sps": float(sample_rate_sps),
        "baseband_frequency_hz": float(baseband_frequency_hz),
    }


def locate_and_estimate_pn(
    capture: np.ndarray,
    *,
    sample_rate_sps: float = 1_250_000.0,
    baseband_frequency_hz: float = 200_000.0,
) -> dict[str, object]:
    """Locate the PN sequence, estimate residual CFO, and estimate scalar h."""
    r = np.asarray(capture, dtype=np.complex128).reshape(-1)
    n = np.arange(r.size)
    downconverted = r * np.exp(-1j * 2 * np.pi * baseband_frequency_hz * n / sample_rate_sps)
    training = pn_sequence(10)
    start, correlation_peak = _peak_valid_correlation(downconverted, training)
    first_lag = max(0, start - 2500)
    last_lag = min(r.size - training.size, start + 2500)
    local_values = downconverted[first_lag : last_lag + training.size]
    correlation = _fft_convolve_valid(local_values, np.conj(training[::-1]))
    correlation_lags = first_lag + np.arange(correlation.size)
    received = downconverted[start : start + training.size]
    product = received * np.conj(training)
    phase = np.unwrap(np.angle(product))
    sample_index = np.arange(training.size)
    weights = np.maximum(np.abs(product), np.finfo(float).eps)
    slope, intercept = np.polyfit(sample_index, phase, 1, w=weights)
    cfo_hz = float(slope * sample_rate_sps / (2 * np.pi))
    corrected = received * np.exp(-1j * 2 * np.pi * cfo_hz * sample_index / sample_rate_sps)
    channel = complex(np.vdot(training, corrected) / np.vdot(training, training))
    equalized = corrected / channel
    evm = float(np.sqrt(np.mean(np.abs(equalized - training) ** 2) / np.mean(np.abs(training) ** 2)))
    decisions = np.where(equalized.real >= 0, 1.0, -1.0)
    chip_error_rate = float(np.mean(decisions != training.real))
    return {
        "start_sample": start,
        "correlation_peak": correlation_peak,
        "correlation": correlation,
        "correlation_lags": correlation_lags,
        "received": received,
        "product_phase": phase,
        "phase_fit": slope * sample_index + intercept,
        "cfo_hz": cfo_hz,
        "corrected": corrected,
        "channel": channel,
        "equalized": equalized,
        "evm": evm,
        "chip_error_rate": chip_error_rate,
    }
