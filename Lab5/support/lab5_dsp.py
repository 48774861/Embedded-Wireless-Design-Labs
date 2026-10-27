

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt
import zlib

import numpy as np


FS = 1_000_000.0
NFFT = 64
NCP = 16
N_PAYLOAD_SYMBOLS = 20
FRAME_PERIOD = 8192
LEADING_GUARD = 128
TRAILING_GUARD = 128
SHORT_LEN = 16
N_SHORT_REPETITIONS = 4
PREAMBLE_LEN = SHORT_LEN * N_SHORT_REPETITIONS
N_TRAINING = 2
TX_PEAK_LIMIT = 0.10
FFT_WINDOW_ADVANCE = 4
PAYLOAD_SEED = 0x5A17

ACTIVE_SIGNED = np.r_[np.arange(-26, 0), np.arange(1, 27)]
PILOT_SIGNED = np.array([-21, -7, 7, 21])
DATA_SIGNED = np.array([k for k in ACTIVE_SIGNED if k not in set(PILOT_SIGNED)])
ACTIVE_IDX = ACTIVE_SIGNED % NFFT
PILOT_IDX = PILOT_SIGNED % NFFT
DATA_IDX = DATA_SIGNED % NFFT

# The first three pilots are +1 and the fourth is -1, followed by a
# per-symbol polarity.  This explicit sequence avoids hidden state.
PILOT_BASE = np.array([1.0, 1.0, 1.0, -1.0], dtype=np.complex64)
PILOT_POLARITY = np.array(
    [1, 1, 1, -1, -1, 1, 1, -1, 1, -1,
     1, 1, -1, 1, -1, -1, 1, 1, -1, 1],
    dtype=np.float32,
)

# One deterministic, constant-envelope 16-sample sequence, repeated four
# times.  The entries are QPSK values with unit magnitude.
_SHORT_SIGNS = np.array(
    [
        1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j,
        1 + 1j, -1 + 1j, 1 - 1j, -1 - 1j,
        -1 + 1j, 1 + 1j, -1 - 1j, 1 - 1j,
        1 + 1j, -1 - 1j, -1 + 1j, 1 - 1j,
    ],
    dtype=np.complex64,
)
SHORT_SECTION = _SHORT_SIGNS / np.sqrt(np.float32(2.0))
SHORT_PREAMBLE = np.tile(SHORT_SECTION, N_SHORT_REPETITIONS)


def signed_to_index(k: np.ndarray | int) -> np.ndarray | int:
    """Map signed subcarrier numbers to unshifted NumPy FFT indices."""
    return np.asarray(k) % NFFT if not np.isscalar(k) else int(k) % NFFT


def _lfsr_bpsk(length: int, state: int = 0x5D) -> np.ndarray:
    """Generate a fixed BPSK sequence using x^7 + x^4 + 1.

    The output bit is the least-significant bit.  The feedback bit is the XOR
    of bit positions 0 and 3; the register shifts right.  State must be nonzero.
    """
    if not 0 < state < 128:
        raise ValueError("the 7-bit LFSR state must be in [1, 127]")
    bits = np.empty(length, dtype=np.uint8)
    reg = int(state)
    for n in range(length):
        bits[n] = reg & 1
        feedback = ((reg >> 0) ^ (reg >> 3)) & 1
        reg = (reg >> 1) | (feedback << 6)
    return (1.0 - 2.0 * bits.astype(np.float32)).astype(np.complex64)


TRAINING_ACTIVE = _lfsr_bpsk(len(ACTIVE_IDX))


def qpsk_map(bits: np.ndarray) -> np.ndarray:
    """Map [b0,b1] using 00 NE, 01 NW, 11 SW, 10 SE."""
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if b.size % 2:
        raise ValueError("QPSK input bit count must be even")
    if np.any(b > 1):
        raise ValueError("bits must contain only 0 and 1")
    pairs = b.reshape(-1, 2)
    real = 1.0 - 2.0 * pairs[:, 1]
    imag = 1.0 - 2.0 * pairs[:, 0]
    return ((real + 1j * imag) / np.sqrt(2.0)).astype(np.complex64)


def qpsk_detect(symbols: np.ndarray) -> np.ndarray:
    """Hard-decision inverse of qpsk_map; zeros lie on the positive side."""
    z = np.asarray(symbols).reshape(-1)
    bits = np.empty(2 * z.size, dtype=np.uint8)
    bits[0::2] = np.imag(z) < 0
    bits[1::2] = np.real(z) < 0
    return bits


def int_to_bits(value: int, width: int) -> np.ndarray:
    if value < 0 or value >= 2**width:
        raise ValueError(f"value does not fit in {width} bits")
    shifts = np.arange(width - 1, -1, -1)
    return ((value >> shifts) & 1).astype(np.uint8)


def crc32_bits(data_bits: np.ndarray) -> np.ndarray:
    """IEEE CRC-32, serialized most-significant bit first."""
    b = np.asarray(data_bits, dtype=np.uint8).reshape(-1)
    if b.size % 8:
        raise ValueError("CRC input length must be an integer number of bytes")
    packed = np.packbits(b, bitorder="big").tobytes()
    return int_to_bits(zlib.crc32(packed) & 0xFFFFFFFF, 32)


def make_payload_bits(sequence_number: int = 0) -> np.ndarray:
    """Return 16-bit sequence, 1872 deterministic test bits, and CRC-32."""
    seq = int_to_bits(sequence_number % 2**16, 16)
    rng = np.random.Generator(np.random.PCG64(PAYLOAD_SEED + sequence_number))
    message = rng.integers(0, 2, 1872, dtype=np.uint8)
    protected = np.r_[seq, message]
    payload = np.r_[protected, crc32_bits(protected)]
    assert payload.size == 1920
    return payload


def check_crc(payload_bits: np.ndarray) -> bool:
    b = np.asarray(payload_bits, dtype=np.uint8).reshape(-1)
    return b.size == 1920 and np.array_equal(crc32_bits(b[:-32]), b[-32:])


def pilot_values(symbol_number: int) -> np.ndarray:
    return PILOT_BASE * PILOT_POLARITY[symbol_number % len(PILOT_POLARITY)]


def build_payload_grid(bits: np.ndarray) -> np.ndarray:
    b = np.asarray(bits, dtype=np.uint8).reshape(-1)
    expected = N_PAYLOAD_SYMBOLS * len(DATA_IDX) * 2
    if b.size != expected:
        raise ValueError(f"expected {expected} payload bits, received {b.size}")
    data = qpsk_map(b).reshape(N_PAYLOAD_SYMBOLS, len(DATA_IDX))
    grid = np.zeros((N_PAYLOAD_SYMBOLS, NFFT), dtype=np.complex64)
    grid[:, DATA_IDX] = data
    for m in range(N_PAYLOAD_SYMBOLS):
        grid[m, PILOT_IDX] = pilot_values(m)
    return grid


def build_training_grid() -> np.ndarray:
    grid = np.zeros((N_TRAINING, NFFT), dtype=np.complex64)
    grid[:, ACTIVE_IDX] = TRAINING_ACTIVE[None, :]
    return grid


def add_cp(useful: np.ndarray, ncp: int = NCP) -> np.ndarray:
    x = np.asarray(useful)
    if x.shape[-1] != NFFT:
        raise ValueError("last dimension must equal NFFT")
    return np.concatenate((x[..., -ncp:], x), axis=-1)


def remove_cp(cp_symbols: np.ndarray, ncp: int = NCP) -> np.ndarray:
    x = np.asarray(cp_symbols)
    if x.shape[-1] != NFFT + ncp:
        raise ValueError("last dimension does not match NFFT + NCP")
    return x[..., ncp:ncp + NFFT]


def ofdm_modulate(grid: np.ndarray, ncp: int = NCP) -> np.ndarray:
    useful = np.fft.ifft(np.asarray(grid), n=NFFT, axis=-1, norm="ortho")
    return add_cp(useful, ncp=ncp)


def ofdm_demodulate(cp_symbols: np.ndarray, ncp: int = NCP) -> np.ndarray:
    useful = remove_cp(cp_symbols, ncp=ncp)
    return np.fft.fft(useful, n=NFFT, axis=-1, norm="ortho")


@dataclass(frozen=True)
class ReferenceFrame:
    sequence_number: int
    ncp: int
    payload_bits: np.ndarray
    payload_grid: np.ndarray
    training_grid: np.ndarray
    training_waveform: np.ndarray
    logical_period: np.ndarray
    tx_period: np.ndarray
    tx_scale: float
    papr_db: float

    @property
    def training_start(self) -> int:
        return LEADING_GUARD + PREAMBLE_LEN

    @property
    def payload_start(self) -> int:
        return self.training_start + N_TRAINING * (NFFT + self.ncp)

    @property
    def active_frame_length(self) -> int:
        return (
            LEADING_GUARD + PREAMBLE_LEN
            + (N_TRAINING + N_PAYLOAD_SYMBOLS) * (NFFT + self.ncp)
            + TRAILING_GUARD
        )


def build_reference_frame(sequence_number: int = 0, ncp: int = NCP) -> ReferenceFrame:
    payload_bits = make_payload_bits(sequence_number)
    payload_grid = build_payload_grid(payload_bits)
    training_grid = build_training_grid()
    training_cp = ofdm_modulate(training_grid, ncp=ncp)
    payload_cp = ofdm_modulate(payload_grid, ncp=ncp)
    training_waveform = training_cp.reshape(-1)

    active = np.concatenate(
        [
            np.zeros(LEADING_GUARD, dtype=np.complex128),
            SHORT_PREAMBLE.astype(np.complex128),
            training_waveform,
            payload_cp.reshape(-1),
            np.zeros(TRAILING_GUARD, dtype=np.complex128),
        ]
    )
    if active.size > FRAME_PERIOD:
        raise ValueError("active frame exceeds frame period")
    logical = np.zeros(FRAME_PERIOD, dtype=np.complex128)
    logical[:active.size] = active
    peak = float(np.max(np.abs(logical)))
    scale = TX_PEAK_LIMIT / peak if peak > TX_PEAK_LIMIT else 1.0
    tx = (scale * logical).astype(np.complex64)
    nonzero = np.abs(logical) > 0
    power = np.mean(np.abs(logical[nonzero]) ** 2)
    papr = 10.0 * np.log10(np.max(np.abs(logical[nonzero]) ** 2) / power)

    frame = ReferenceFrame(
        sequence_number=sequence_number,
        ncp=ncp,
        payload_bits=payload_bits,
        payload_grid=payload_grid,
        training_grid=training_grid,
        training_waveform=training_waveform,
        logical_period=logical,
        tx_period=tx,
        tx_scale=scale,
        papr_db=float(papr),
    )
    expected_length = 2080 + (N_TRAINING + N_PAYLOAD_SYMBOLS) * (ncp - NCP)
    assert frame.active_frame_length == expected_length
    assert frame.logical_period.size == FRAME_PERIOD
    assert np.max(np.abs(frame.tx_period)) <= TX_PEAK_LIMIT + 1e-7
    return frame


def _extract_cp_symbols(
    samples: np.ndarray, start: int, count: int, ncp: int = NCP
) -> np.ndarray:
    width = NFFT + ncp
    stop = start + count * width
    if start < 0 or stop > len(samples):
        raise ValueError("capture does not contain all requested OFDM symbols")
    return np.asarray(samples[start:stop]).reshape(count, width)


def evm_rms(estimated: np.ndarray, reference: np.ndarray) -> float:
    e = np.asarray(estimated) - np.asarray(reference)
    denom = np.sum(np.abs(reference) ** 2)
    return float(np.sqrt(np.sum(np.abs(e) ** 2) / denom))


def bit_error_rate(estimated: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(estimated, dtype=np.uint8).reshape(-1)
    b = np.asarray(reference, dtype=np.uint8).reshape(-1)
    if a.size != b.size:
        raise ValueError("bit arrays must have the same length")
    return float(np.count_nonzero(a != b) / b.size)


def estimate_channel(received_training_grid: np.ndarray) -> np.ndarray:
    y = np.asarray(received_training_grid)
    if y.shape != (N_TRAINING, NFFT):
        raise ValueError(f"training grid must have shape {(N_TRAINING, NFFT)}")
    h = np.zeros(NFFT, dtype=np.complex128)
    h[ACTIVE_IDX] = np.mean(
        y[:, ACTIVE_IDX] / TRAINING_ACTIVE[None, :], axis=0
    )
    return h


def equalize_zf(received_grid: np.ndarray, h_est: np.ndarray, floor: float = 1e-3) -> np.ndarray:
    y = np.asarray(received_grid)
    h = np.asarray(h_est)
    mag = np.abs(h)
    phase = np.exp(1j * np.angle(h + 1e-30))
    h_safe = np.where(mag < floor, floor * phase, h)
    return y / h_safe[None, :]


def equalize_mmse(
    received_grid: np.ndarray,
    h_est: np.ndarray,
    noise_variance: float,
    symbol_energy: float = 1.0,
) -> np.ndarray:
    h = np.asarray(h_est)
    weight = np.conj(h) / (np.abs(h) ** 2 + noise_variance / symbol_energy)
    return np.asarray(received_grid) * weight[None, :]


def pilot_cpe_correct(equalized_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(equalized_grid).copy()
    phases = np.empty(x.shape[0], dtype=float)
    for m in range(x.shape[0]):
        c = pilot_values(m)
        phases[m] = np.angle(np.sum(np.conj(c) * x[m, PILOT_IDX]))
        x[m, ACTIVE_IDX] *= np.exp(-1j * phases[m])
    return x, phases


def decode_at_training_start(
    corrected_samples: np.ndarray,
    training_start: int,
    reference: ReferenceFrame,
    *,
    equalizer: str = "zf",
    noise_variance: float | None = None,
) -> dict:
    """Decode one CFO-corrected frame from the start of its first training CP."""
    ncp = reference.ncp
    train_cp = _extract_cp_symbols(corrected_samples, training_start, N_TRAINING, ncp)
    payload_start = training_start + N_TRAINING * (NFFT + ncp)
    payload_cp = _extract_cp_symbols(
        corrected_samples, payload_start, N_PAYLOAD_SYMBOLS, ncp
    )
    y_train = ofdm_demodulate(train_cp, ncp)
    y_payload = ofdm_demodulate(payload_cp, ncp)
    h_est = estimate_channel(y_train)
    if equalizer == "zf":
        # Hardware channel coefficients include the arbitrary OTA link scale.
        # A fixed absolute floor would therefore suppress valid low-amplitude
        # captures.  Set the floor relative to the active-carrier median.
        relative_floor = max(
            1e-12, 0.05 * float(np.median(np.abs(h_est[ACTIVE_IDX])))
        )
        equalized = equalize_zf(y_payload, h_est, floor=relative_floor)
    elif equalizer == "mmse":
        if noise_variance is None:
            raise ValueError("MMSE requires noise_variance")
        equalized = equalize_mmse(y_payload, h_est, noise_variance)
    else:
        raise ValueError("equalizer must be 'zf' or 'mmse'")
    corrected_grid, cpe = pilot_cpe_correct(equalized)
    data_symbols = corrected_grid[:, DATA_IDX]
    decoded_bits = qpsk_detect(data_symbols)
    reference_symbols = reference.payload_grid[:, DATA_IDX]
    return {
        "received_training_grid": y_train,
        "received_payload_grid": y_payload,
        "channel_estimate": h_est,
        "equalized_grid": equalized,
        "corrected_grid": corrected_grid,
        "cpe_rad": cpe,
        "decoded_bits": decoded_bits,
        "crc_ok": check_crc(decoded_bits),
        "ber": bit_error_rate(decoded_bits, reference.payload_bits),
        "evm_rms": evm_rms(data_symbols, reference_symbols),
        "evm_db": 20.0 * np.log10(max(evm_rms(data_symbols, reference_symbols), 1e-15)),
    }


def repetition_metric(samples: np.ndarray, length: int = SHORT_LEN) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return M[d], P[d], R[d] for the Lab 5 four-repeat preamble."""
    r = np.asarray(samples).reshape(-1)
    window = 3 * length
    if r.size < window + length:
        raise ValueError("capture is too short for the repetition metric")
    products = np.conj(r[:-length]) * r[length:]
    delayed_power = np.abs(r[length:]) ** 2

    def moving_sum(x: np.ndarray, width: int) -> np.ndarray:
        cs = np.concatenate((np.zeros(1, dtype=x.dtype), np.cumsum(x)))
        return cs[width:] - cs[:-width]

    p = moving_sum(products, window)
    energy = moving_sum(delayed_power, window).real
    epsilon = 1e-12 * max(float(np.max(energy) ** 2), 1.0)
    metric = np.abs(p) ** 2 / (energy**2 + epsilon)
    return metric, p, energy


def detect_preambles(
    samples: np.ndarray,
    threshold: float = 0.55,
    min_separation: int = FRAME_PERIOD // 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect preamble candidates with thresholding and nonmaximum suppression."""
    metric, p, energy = repetition_metric(samples)
    # A normalized correlation alone is unreliable in near-zero guard regions:
    # two tiny numbers can form a large ratio.  Require a modest fraction of
    # the maximum 48-sample delayed-window energy as a second gate.
    energy_gate = 0.10 * float(np.max(energy))
    above = np.flatnonzero((metric >= threshold) & (energy >= energy_gate))
    if above.size == 0:
        return np.empty(0, dtype=int), metric, p, energy

    # Find the maximum in each contiguous above-threshold island.
    split = np.flatnonzero(np.diff(above) > 1) + 1
    islands = np.split(above, split)
    candidates = np.array([g[np.argmax(metric[g])] for g in islands], dtype=int)

    # Keep the strongest candidate in each frame-sized neighborhood.
    kept: list[int] = []
    for idx in candidates[np.argsort(metric[candidates])[::-1]]:
        if all(abs(int(idx) - old) >= min_separation for old in kept):
            kept.append(int(idx))
    return np.array(sorted(kept), dtype=int), metric, p, energy


def estimate_cfo(p_value: complex, fs: float = FS, length: int = SHORT_LEN) -> float:
    return float(fs * np.angle(p_value) / (2.0 * np.pi * length))


def correct_cfo(samples: np.ndarray, cfo_hz: float, fs: float = FS) -> np.ndarray:
    n = np.arange(len(samples), dtype=float)
    return np.asarray(samples) * np.exp(-1j * 2.0 * np.pi * cfo_hz * n / fs)


def fine_training_start(
    corrected_samples: np.ndarray,
    coarse_preamble_start: int,
    reference: ReferenceFrame,
    radius: int = 32,
) -> tuple[int, np.ndarray, np.ndarray]:
    """Cross-correlate around the expected start of the first training CP."""
    expected = coarse_preamble_start + PREAMBLE_LEN
    starts = np.arange(max(0, expected - radius), expected + radius + 1)
    ref = reference.training_waveform
    ref_norm = np.linalg.norm(ref)
    corr = np.zeros(starts.size, dtype=float)
    for i, start in enumerate(starts):
        segment = np.asarray(corrected_samples[start:start + ref.size])
        if segment.size == ref.size:
            denom = ref_norm * np.linalg.norm(segment)
            corr[i] = abs(np.vdot(ref, segment)) / denom if denom > 0 else 0.0
    best = int(starts[np.argmax(corr)])
    return best, starts, corr


def recover_frame(
    samples: np.ndarray,
    coarse_preamble_start: int,
    p_value: complex,
    reference: ReferenceFrame,
) -> dict:
    coarse_cfo_hz = estimate_cfo(p_value)
    corrected = correct_cfo(samples, coarse_cfo_hz)
    training_start, timing_starts, timing_corr = fine_training_start(
        corrected, coarse_preamble_start, reference
    )
    # Fine timing gives the actual preamble boundary.  Re-estimate CFO while
    # omitting eight samples at the beginning of the repeated preamble.  This
    # avoids the channel turn-on transient yet keeps every delayed sample
    # inside the 64-sample preamble.
    preamble_start = training_start - PREAMBLE_LEN
    trim = 8
    first = np.asarray(samples[preamble_start + trim:preamble_start + 48])
    second = np.asarray(samples[preamble_start + trim + SHORT_LEN:preamble_start + 64])
    refined_p = np.vdot(first, second)
    cfo_hz = estimate_cfo(refined_p)
    corrected = correct_cfo(samples, cfo_hz)
    training_start, timing_starts, timing_corr = fine_training_start(
        corrected, preamble_start, reference
    )
    # Put the FFT window inside the CP rather than at the correlation peak.
    # This fixed advance leaves margin for a delayed strongest path while the
    # fine timing result itself remains the nominal training boundary.
    fft_window_start = training_start - FFT_WINDOW_ADVANCE
    decoded = decode_at_training_start(corrected, fft_window_start, reference)
    decoded.update(
        {
            "coarse_preamble_start": int(coarse_preamble_start),
            "training_start": training_start,
            "fft_window_start": fft_window_start,
            "frame_start": training_start - LEADING_GUARD - PREAMBLE_LEN,
            "coarse_cfo_hz": coarse_cfo_hz,
            "cfo_hz": cfo_hz,
            "timing_search_starts": timing_starts,
            "timing_correlation": timing_corr,
        }
    )
    return decoded


def recover_capture(
    samples: np.ndarray,
    reference: ReferenceFrame,
    threshold: float = 0.55,
) -> tuple[list[dict], dict]:
    candidates, metric, p, energy = detect_preambles(samples, threshold=threshold)
    recovered: list[dict] = []
    for d in candidates:
        try:
            recovered.append(recover_frame(samples, int(d), p[d], reference))
        except ValueError:
            # A partial final frame is not a decoder failure.
            continue
    diagnostics = {
        "preamble_candidates": candidates,
        "repetition_metric": metric,
        "repetition_correlation": p,
        "repetition_energy": energy,
        "threshold": threshold,
    }
    return recovered, diagnostics


def add_complex_awgn(
    samples: np.ndarray, noise_power: float, rng: np.random.Generator
) -> np.ndarray:
    sigma = np.sqrt(noise_power / 2.0)
    noise = sigma * (rng.standard_normal(len(samples)) + 1j * rng.standard_normal(len(samples)))
    return np.asarray(samples) + noise


def apply_cfo_and_phase(
    samples: np.ndarray, cfo_hz: float, phase_rad: float, fs: float = FS
) -> np.ndarray:
    n = np.arange(len(samples), dtype=float)
    return np.asarray(samples) * np.exp(1j * (2.0 * np.pi * cfo_hz * n / fs + phase_rad))


def short_channel() -> np.ndarray:
    h = np.array([1, 0, 0.45 * np.exp(0.4j), 0, 0, 0.20 * np.exp(-0.9j)])
    return h / np.linalg.norm(h)


def long_channel() -> np.ndarray:
    h = np.zeros(21, dtype=np.complex128)
    h[0] = 1.0
    h[20] = 0.65 * np.exp(0.8j)
    return h / np.linalg.norm(h)


def theoretical_qpsk_ber(ebn0_db: float) -> float:
    gamma_b = 10.0 ** (ebn0_db / 10.0)
    return 0.5 * erfc(sqrt(gamma_b))


def awgn_sweep(ebn0_values=(0, 5, 10, 15, 20, 25), frames_per_point: int = 100) -> list[dict]:
    """Task 2.1 reference sweep using exact timing and channel knowledge."""
    results: list[dict] = []
    for ebn0_db in ebn0_values:
        rng = np.random.default_rng(9000 + int(ebn0_db))
        errors = 0
        total = 0
        error_energy = 0.0
        reference_energy = 0.0
        n0 = 1.0 / (2.0 * 10.0 ** (ebn0_db / 10.0))
        for frame_number in range(frames_per_point):
            ref = build_reference_frame(frame_number)
            payload_cp = ofdm_modulate(ref.payload_grid, ref.ncp)
            noisy = add_complex_awgn(payload_cp.reshape(-1), n0, rng)
            y = ofdm_demodulate(noisy.reshape(N_PAYLOAD_SYMBOLS, NFFT + ref.ncp), ref.ncp)
            data = y[:, DATA_IDX]
            detected = qpsk_detect(data)
            errors += int(np.count_nonzero(detected != ref.payload_bits))
            total += ref.payload_bits.size
            e = data - ref.payload_grid[:, DATA_IDX]
            error_energy += float(np.sum(np.abs(e) ** 2))
            reference_energy += float(np.sum(np.abs(ref.payload_grid[:, DATA_IDX]) ** 2))
        evm = sqrt(error_energy / reference_energy)
        results.append(
            {
                "ebn0_db": float(ebn0_db),
                "errors": errors,
                "bits": total,
                "ber": errors / total,
                "theory": theoretical_qpsk_ber(float(ebn0_db)),
                "evm_rms": evm,
                "evm_db": 20.0 * np.log10(evm),
            }
        )
    return results


def simulate_impaired_capture(
    reference: ReferenceFrame,
    *,
    delay: int = 777,
    cfo_hz: float = 7500.0,
    phase_rad: float = 0.7,
    snr_db: float = 20.0,
    channel: np.ndarray | None = None,
    seed: int = 7301,
) -> tuple[np.ndarray, dict]:
    """Create a deterministic Task 3 test capture using direct NumPy math."""
    h = short_channel() if channel is None else np.asarray(channel)
    channel_output = np.convolve(reference.logical_period, h)
    delayed = np.r_[np.zeros(delay, dtype=np.complex128), channel_output]
    impaired = apply_cfo_and_phase(delayed, cfo_hz, phase_rad)
    start = delay + reference.training_start
    stop = delay + reference.payload_start + N_PAYLOAD_SYMBOLS * (NFFT + reference.ncp)
    signal_power = float(np.mean(np.abs(impaired[start:stop]) ** 2))
    noise_power = signal_power * 10.0 ** (-snr_db / 10.0)
    rng = np.random.default_rng(seed)
    capture = add_complex_awgn(impaired, noise_power, rng)
    truth = {
        "delay": delay,
        "frame_start": delay,
        "preamble_start": delay + LEADING_GUARD,
        "training_start": delay + reference.training_start,
        "cfo_hz": cfo_hz,
        "phase_rad": phase_rad,
        "snr_db": snr_db,
        "noise_power": noise_power,
        "channel": h,
    }
    return capture.astype(np.complex64), truth


def nmse(estimated: np.ndarray, reference: np.ndarray, indices=ACTIVE_IDX) -> float:
    e = np.asarray(estimated)[indices] - np.asarray(reference)[indices]
    denom = np.sum(np.abs(np.asarray(reference)[indices]) ** 2)
    return float(np.sum(np.abs(e) ** 2) / denom)


def ideal_round_trip(reference: ReferenceFrame) -> dict:
    payload_cp = ofdm_modulate(reference.payload_grid, reference.ncp)
    recovered_grid = ofdm_demodulate(payload_cp, reference.ncp)
    recovered_data = recovered_grid[:, DATA_IDX]
    detected = qpsk_detect(recovered_data)
    evm = evm_rms(recovered_data, reference.payload_grid[:, DATA_IDX])
    return {
        "ber": bit_error_rate(detected, reference.payload_bits),
        "bit_errors": int(np.count_nonzero(detected != reference.payload_bits)),
        "evm_rms": evm,
        "max_grid_error": float(np.max(np.abs(recovered_grid - reference.payload_grid))),
        "max_null": float(np.max(np.abs(recovered_grid[:, np.setdiff1d(np.arange(NFFT), ACTIVE_IDX)]))),
    }


def print_quick_check() -> None:
    ref = build_reference_frame(0)
    ideal = ideal_round_trip(ref)
    capture, truth = simulate_impaired_capture(ref)
    recovered, diagnostics = recover_capture(capture, ref)
    print("Lab 5 instructor reference quick check")
    print(f"active frame samples : {ref.active_frame_length}")
    print(f"frame period samples : {len(ref.logical_period)}")
    print(f"TX scale             : {ref.tx_scale:.9f}")
    print(f"TX peak              : {np.max(np.abs(ref.tx_period)):.9f}")
    print(f"PAPR                 : {ref.papr_db:.3f} dB")
    print(f"ideal bit errors     : {ideal['bit_errors']}")
    print(f"ideal EVM            : {ideal['evm_rms']:.3e}")
    print(f"detector candidates  : {diagnostics['preamble_candidates'].tolist()}")
    if recovered:
        result = recovered[0]
        print(f"timing error         : {result['training_start'] - truth['training_start']} samples")
        print(f"CFO estimate         : {result['cfo_hz']:.3f} Hz")
        print(f"CFO error            : {result['cfo_hz'] - truth['cfo_hz']:.3f} Hz")
        print(f"impaired BER         : {result['ber']:.6g}")
        print(f"impaired EVM         : {result['evm_rms']:.6g}")
        print(f"CRC                   : {'PASS' if result['crc_ok'] else 'FAIL'}")
    else:
        print("impaired recovery    : FAILED")


if __name__ == "__main__":
    print_quick_check()
