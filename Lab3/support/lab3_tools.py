"""Provided support code for the ECE 5377/7377 Lab 3 notebooks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from support.lab3_dsp import (
    bits_to_integers,
    error_rates,
    integers_to_bits,
    minimum_distance_detect,
    principal_alias,
    psk_constellation,
    pulse_shape,
    rrc_taps,
)


TINY = np.finfo(float).tiny


def find_lab_root() -> Path:
    """Find a Lab 3 directory when a notebook is opened from JupyterLab."""
    for base in (Path.cwd(), *Path.cwd().parents):
        if (base / "support" / "lab3_tools.py").is_file():
            return base
        if (base / "python" / "lab3_tools.py").is_file():
            return base
    raise FileNotFoundError(
        "Could not find support/lab3_tools.py. Open the notebook inside the Lab 3 folder."
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    return path


def centered_welch(
    samples: np.ndarray, sample_rate_sps: float, nperseg: int = 8192
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(samples)
    segment_length = min(int(nperseg), values.size)
    frequency, psd = signal.welch(
        values,
        fs=float(sample_rate_sps),
        window="hann",
        nperseg=segment_length,
        noverlap=segment_length // 2,
        return_onesided=False,
        scaling="density",
    )
    return np.fft.fftshift(frequency), np.fft.fftshift(psd)


def simulate_awgn_point(
    constellation: np.ndarray,
    ebn0_db: float,
    rng: np.random.Generator,
    detector: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    target_symbol_errors: int = 200,
    maximum_symbols: int = 2_000_000,
    block_symbols: int = 50_000,
) -> dict[str, float | int]:
    """Run one statistically bounded complex-AWGN Monte Carlo point."""
    constellation = np.asarray(constellation, dtype=np.complex128)
    order = int(constellation.size)
    bits_per_symbol = int(math.log2(order))
    esn0_db = float(ebn0_db + 10.0 * np.log10(bits_per_symbol))
    gamma_s = 10.0 ** (esn0_db / 10.0)
    n0 = 1.0 / gamma_s
    total_symbols = 0
    symbol_errors = 0
    bit_errors = 0

    while total_symbols < maximum_symbols and symbol_errors < target_symbol_errors:
        count = min(block_symbols, maximum_symbols - total_symbols)
        labels = rng.integers(0, order, count)
        transmitted = constellation[labels]
        noise = np.sqrt(n0 / 2.0) * (
            rng.standard_normal(count) + 1j * rng.standard_normal(count)
        )
        detected = detector(transmitted + noise, constellation)
        symbol_errors += int(np.count_nonzero(detected != labels))
        bit_errors += int(
            np.count_nonzero(
                integers_to_bits(detected, bits_per_symbol)
                != integers_to_bits(labels, bits_per_symbol)
            )
        )
        total_symbols += count

    return {
        "ebn0_db": float(ebn0_db),
        "esn0_db": esn0_db,
        "symbols": total_symbols,
        "symbol_errors": symbol_errors,
        "bit_errors": bit_errors,
        "ser": symbol_errors / total_symbols,
        "ber": bit_errors / (total_symbols * bits_per_symbol),
        "zero_error_95_percent_upper_bound": 3.0 / total_symbols,
    }


def plot_error_rate_results(
    results: dict[int, list[dict[str, float | int]]],
    theory: Callable[[int, np.ndarray], np.ndarray],
    title: str,
    output_dir: Path,
    stem: str,
    minimum_errors_to_plot: int = 20,
) -> Path:
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for order, points in results.items():
        ebn0 = np.asarray([point["ebn0_db"] for point in points], dtype=float)
        errors = np.asarray([point["symbol_errors"] for point in points], dtype=int)
        ser = np.asarray([point["ser"] for point in points], dtype=float)
        dense = np.linspace(ebn0[0], ebn0[-1], 401)
        line = ax.semilogy(dense, theory(order, dense), linewidth=1.8, label=f"{order} theory")[0]
        reliable = errors >= minimum_errors_to_plot
        ax.semilogy(
            ebn0[reliable],
            ser[reliable],
            "o",
            color=line.get_color(),
            markersize=5,
            label=f"{order} simulation",
        )
    ax.set(
        title=title,
        xlabel=r"$E_b/N_0$ (dB)",
        ylabel="Symbol error rate",
        ylim=(1e-6, 1.0),
    )
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8, loc="lower left")
    return save_figure(fig, output_dir, stem)


def plot_constellation_snapshots(
    snapshots: list[tuple[str, np.ndarray, np.ndarray]], output_dir: Path
) -> Path:
    fig, axes = plt.subplots(1, len(snapshots), figsize=(10.2, 3.5), constrained_layout=True)
    for axis, (title, received, reference) in zip(np.atleast_1d(axes), snapshots):
        axis.scatter(np.real(received), np.imag(received), s=4, alpha=0.25)
        axis.scatter(
            np.real(reference),
            np.imag(reference),
            marker="x",
            s=55,
            linewidths=1.8,
            color="black",
        )
        axis.set(title=title, xlabel="I", ylabel="Q", aspect="equal")
        axis.grid(True, alpha=0.25)
    return save_figure(fig, output_dir, "task1_noisy_constellations")


def plot_pam_chain(
    symbols: np.ndarray,
    upsampled: np.ndarray,
    transmitted: np.ndarray,
    received: np.ndarray,
    matched: np.ndarray,
    decision_indices: np.ndarray,
    decisions: np.ndarray,
    samples_per_symbol: int,
    output_dir: Path,
) -> Path:
    shown_symbols = 24
    shown_samples = shown_symbols * samples_per_symbol
    fig, axes = plt.subplots(3, 2, figsize=(10.0, 8.2), constrained_layout=True)
    axes[0, 0].stem(np.arange(shown_symbols), symbols[:shown_symbols], basefmt=" ")
    axes[0, 0].set(title="Normalized 4-PAM symbols", xlabel="Symbol index", ylabel="Amplitude")
    axes[0, 1].stem(np.arange(shown_samples), upsampled[:shown_samples], basefmt=" ")
    axes[0, 1].set(title="Zero insertion, L = 4", xlabel="Sample index", ylabel="Amplitude")
    axes[1, 0].plot(transmitted[:shown_samples])
    axes[1, 0].set(title="RRC pulse-shaped waveform", xlabel="Sample index", ylabel="Amplitude")
    axes[1, 1].plot(received[:shown_samples])
    axes[1, 1].set(title="Channel output with AWGN", xlabel="Sample index", ylabel="Amplitude")
    view_start = int(decision_indices[0])
    view_stop = view_start + shown_samples
    axes[2, 0].plot(np.arange(view_start, view_stop), matched[view_start:view_stop])
    shown_indices = decision_indices[:shown_symbols]
    axes[2, 0].scatter(shown_indices, matched[shown_indices], s=18, color="tab:red")
    axes[2, 0].set(title="Matched-filter output and decisions", xlabel="Sample index", ylabel="Amplitude")
    axes[2, 1].stem(np.arange(shown_symbols), decisions[:shown_symbols], basefmt=" ", label="received")
    axes[2, 1].scatter(np.arange(shown_symbols), symbols[:shown_symbols], marker="x", color="tab:red", label="sent")
    axes[2, 1].set(title="Downsampled decisions", xlabel="Symbol index", ylabel="Amplitude")
    axes[2, 1].legend(loc="upper right")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    return save_figure(fig, output_dir, "task2_pam_signal_chain")


def plot_pam_eye(
    symbols: np.ndarray,
    matched: np.ndarray,
    decision_indices: np.ndarray,
    decisions: np.ndarray,
    levels: np.ndarray,
    samples_per_symbol: int,
    ser: float,
    output_dir: Path,
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), constrained_layout=True)
    eye_time = np.arange(-samples_per_symbol, samples_per_symbol + 1) / samples_per_symbol
    for symbol_index in range(20, 220):
        center = int(decision_indices[symbol_index])
        trace = matched[center - samples_per_symbol : center + samples_per_symbol + 1]
        axes[0].plot(eye_time, trace, color="tab:blue", alpha=0.08)
    axes[0].axvline(0.0, color="black", linewidth=1.0)
    axes[0].set(title="Matched-filter eye", xlabel="Time (symbols)", ylabel="Amplitude")
    axes[1].scatter(symbols[:1200], decisions[:1200], s=5, alpha=0.25)
    axes[1].plot(levels, levels, "kx", markersize=8, markeredgewidth=1.8)
    axes[1].set(title=f"Decision samples, SER = {ser:.4f}", xlabel="Transmitted symbol", ylabel="Received sample")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    return save_figure(fig, output_dir, "task2_pam_eye_and_decisions")


def repeated_message_bits(message: str, bit_count: int) -> np.ndarray:
    source = np.unpackbits(np.frombuffer(message.encode("ascii"), dtype=np.uint8))
    return np.resize(source, int(bit_count)).astype(np.uint8)


def build_psk_frame(
    order: int,
    waveform_dir: Path,
    *,
    sample_rate_sps: float = 1_250_000.0,
    symbol_rate_baud: float = 125_000.0,
    rolloff: float = 0.35,
    span_symbols: int = 10,
    baseband_shift_hz: float = 200_000.0,
    message: str = "ECE7377 LAB3 PYTHON RADIO | ",
    guard_symbols: int = 24,
    preamble_symbols: int = 127,
    payload_symbols: int = 512,
    file_periods: int = 8,
    digital_peak: float = 0.10,
) -> dict[str, object]:
    """Build one deterministic finite PSK reference file used by GRC."""
    samples_per_symbol = int(round(sample_rate_sps / symbol_rate_baud))
    bits_per_symbol = int(math.log2(order))
    payload_bits = repeated_message_bits(message, payload_symbols * bits_per_symbol)
    payload_labels = bits_to_integers(payload_bits, bits_per_symbol)
    constellation = psk_constellation(order)
    payload = constellation[payload_labels]
    rng = np.random.default_rng(7377)
    preamble = (2.0 * rng.integers(0, 2, preamble_symbols) - 1.0).astype(complex)
    frame_symbols = np.concatenate(
        (
            np.zeros(guard_symbols, dtype=complex),
            preamble,
            payload,
            np.zeros(guard_symbols, dtype=complex),
        )
    )
    taps = rrc_taps(rolloff, samples_per_symbol, span_symbols)
    _, shaped = pulse_shape(frame_symbols, taps, samples_per_symbol)
    scale = digital_peak / np.max(np.abs(shaped))
    baseband = scale * shaped
    index = np.arange(baseband.size)
    shifted = baseband * np.exp(
        1j * 2.0 * np.pi * baseband_shift_hz * index / sample_rate_sps
    )
    period = shifted.astype(np.complex64)
    finite_file = np.tile(period, file_periods).astype(np.complex64)
    waveform_dir = Path(waveform_dir)
    waveform_dir.mkdir(parents=True, exist_ok=True)
    waveform_path = waveform_dir / f"task3_m{order}_tx_c64.dat"
    finite_file.tofile(waveform_path)
    return {
        "order": order,
        "sample_rate_sps": sample_rate_sps,
        "symbol_rate_baud": symbol_rate_baud,
        "samples_per_symbol": samples_per_symbol,
        "rolloff": rolloff,
        "span_symbols": span_symbols,
        "baseband_shift_hz": baseband_shift_hz,
        "message": message,
        "guard_symbols": guard_symbols,
        "preamble_symbols": preamble_symbols,
        "payload_symbols": payload_symbols,
        "payload_labels": payload_labels,
        "frame_symbols": frame_symbols,
        "taps": taps,
        "scale": float(scale),
        "period": period,
        "finite_file": finite_file,
        "waveform_path": waveform_path,
    }


def plot_task3_waveform(reference: dict[str, object], output_dir: Path) -> Path:
    taps = np.asarray(reference["taps"])
    cascade = np.convolve(taps, taps)
    sps = int(reference["samples_per_symbol"])
    center = (cascade.size - 1) // 2
    symbol_lags = np.arange(-5, 6)
    frequency, psd = centered_welch(
        np.asarray(reference["period"]), float(reference["sample_rate_sps"]), 4096
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 6.6), constrained_layout=True)
    axes[0, 0].plot(np.arange(taps.size) - (taps.size - 1) / 2, taps)
    axes[0, 0].set(title="Transmit RRC pulse", xlabel="Sample offset", ylabel="Amplitude")
    axes[0, 1].stem(symbol_lags, cascade[center + symbol_lags * sps], basefmt=" ")
    axes[0, 1].set(title="TX plus RX raised-cosine samples", xlabel="Symbol offset", ylabel="Amplitude")
    axes[1, 0].plot(np.abs(np.asarray(reference["period"])))
    axes[1, 0].set(title="One finite QPSK frame", xlabel="Sample index", ylabel="Magnitude")
    axes[1, 1].plot(frequency / 1e3, 10.0 * np.log10(np.maximum(psd, TINY)))
    axes[1, 1].axvline(float(reference["baseband_shift_hz"]) / 1e3, color="tab:red", linestyle="--")
    axes[1, 1].set(title="Shifted frame PSD", xlabel="Baseband frequency (kHz)", ylabel="PSD (normalized dB/Hz)")
    for axis in axes.flat:
        axis.grid(True, alpha=0.25)
    return save_figure(fig, output_dir, "task3_frame_and_rrc")


def normalized_power(samples: np.ndarray) -> float:
    values = np.asarray(samples)
    centered = values - np.mean(values)
    return float(np.mean(np.abs(centered) ** 2))


def detect_strongest_burst(
    samples: np.ndarray,
    sample_rate_sps: float,
    noise_power: float,
    block_seconds: float = 0.002,
) -> tuple[int, int, np.ndarray]:
    block_samples = round(block_seconds * sample_rate_sps)
    block_count = samples.size // block_samples
    blocks = np.asarray(samples[: block_count * block_samples]).reshape(block_count, block_samples)
    powers = np.mean(
        np.abs(blocks - np.mean(blocks, axis=1, keepdims=True)) ** 2,
        axis=1,
    )
    threshold = max(float(np.median(powers)), noise_power) * 10 ** (1.5 / 10)
    active = powers > threshold
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    if starts.size == 0:
        # A transmitter started first can occupy the complete finite receive
        # record.  In that valid case there is no noise-only block from which
        # the threshold detector can form rising and falling edges.
        return 0, block_count * block_samples, powers
    scores = np.asarray(
        [np.sum(np.maximum(powers[a:b] - threshold, 0)) for a, b in zip(starts, stops)]
    )
    selected = int(np.argmax(scores))
    margin_blocks = 4
    start = max(int(starts[selected]) - margin_blocks, 0) * block_samples
    stop = min(int(stops[selected]) + margin_blocks, block_count) * block_samples
    return start, stop, powers


def estimate_repeated_period_cfo(
    samples: np.ndarray, period_samples: int, sample_rate_sps: float
) -> float:
    product = np.asarray(samples[period_samples:]) * np.conj(
        np.asarray(samples[:-period_samples])
    )
    phase_per_period = np.angle(np.sum(product, dtype=np.complex128))
    return float(
        phase_per_period * sample_rate_sps / (2.0 * np.pi * period_samples)
    )


def _decode_corrected_frame(
    corrected: np.ndarray, lag: int, reference: dict[str, object]
) -> dict[str, object]:
    period = np.asarray(reference["period"], dtype=np.complex128)
    frame = corrected[lag : lag + period.size]
    channel = np.vdot(period, frame) / np.vdot(period, period)
    equalized = frame / channel
    sample_rate = float(reference["sample_rate_sps"])
    index = np.arange(period.size, dtype=float)
    baseband = equalized * np.exp(
        -1j
        * 2.0
        * np.pi
        * float(reference["baseband_shift_hz"])
        * index
        / sample_rate
    )
    taps = np.asarray(reference["taps"], dtype=float)
    matched = np.convolve(baseband, taps, mode="full")
    sps = int(reference["samples_per_symbol"])
    recovered = matched[taps.size - 1 + np.arange(len(reference["frame_symbols"])) * sps]
    recovered /= float(reference["scale"])
    payload_start = int(reference["guard_symbols"]) + int(reference["preamble_symbols"])
    payload_stop = payload_start + int(reference["payload_symbols"])
    payload = recovered[payload_start:payload_stop]
    labels = np.asarray(reference["payload_labels"], dtype=np.int64)
    constellation = psk_constellation(int(reference["order"]))
    symbol_channel = np.vdot(constellation[labels], payload) / np.vdot(
        constellation[labels], constellation[labels]
    )
    payload_equalized = payload / symbol_channel
    detected = minimum_distance_detect(payload_equalized, constellation)
    ser, ber = error_rates(detected, labels, int(reference["order"]))
    evm = float(
        np.sqrt(
            np.mean(np.abs(payload_equalized - constellation[labels]) ** 2)
            / np.mean(np.abs(constellation[labels]) ** 2)
        )
    )
    width = int(math.log2(int(reference["order"])))
    bits = integers_to_bits(detected, width)
    decoded = np.packbits(bits[: 8 * (bits.size // 8)]).tobytes().decode(
        "ascii", errors="replace"
    )
    return {
        "lag": int(lag),
        "ser": ser,
        "ber": ber,
        "rms_evm": evm,
        "decoded_prefix": decoded[: len(str(reference["message"]))],
        "payload_equalized": payload_equalized,
    }


def analyze_psk_capture(
    capture_path: Path,
    noise_path: Path,
    reference: dict[str, object],
    maximum_frames: int = 24,
) -> dict[str, object]:
    """Analyze one real finite N310 record using the known transmitted period."""
    samples = np.memmap(capture_path, dtype=np.complex64, mode="r")
    noise = np.memmap(noise_path, dtype=np.complex64, mode="r")
    sample_rate = float(reference["sample_rate_sps"])
    noise_power = normalized_power(noise)
    start, stop, block_power = detect_strongest_burst(samples, sample_rate, noise_power)
    period = np.asarray(reference["period"], dtype=np.complex128)
    cfo_hz = estimate_repeated_period_cfo(samples, period.size, sample_rate)
    index = np.arange(samples.size, dtype=float)
    corrected = np.asarray(samples, dtype=np.complex128) * np.exp(
        -1j * 2.0 * np.pi * cfo_hz * index / sample_rate
    )
    correlation = signal.fftconvolve(corrected, np.conj(period[::-1]), mode="valid")
    peaks, _ = signal.find_peaks(
        np.abs(correlation),
        distance=round(0.95 * period.size),
        prominence=0.04 * np.max(np.abs(correlation)),
    )
    if peaks.size == 0:
        raise RuntimeError("Reference correlation found no frame")
    strong = peaks[np.abs(correlation[peaks]) >= 0.45 * np.max(np.abs(correlation[peaks]))]
    if strong.size == 0:
        strong = peaks[np.argsort(np.abs(correlation[peaks]))[-1:]]
    requested = min(maximum_frames, strong.size)
    selected = np.sort(strong[np.argsort(np.abs(correlation[strong]))[-requested:]])
    frames = [_decode_corrected_frame(corrected, int(lag), reference) for lag in selected]
    active_power = normalized_power(samples[start:stop])
    signal_power = max(active_power - noise_power, TINY)
    return {
        "capture_path": str(Path(capture_path)),
        "sample_count": int(samples.size),
        "file_bytes": int(Path(capture_path).stat().st_size),
        "burst_start": int(start),
        "burst_stop": int(stop),
        "frame_count": len(frames),
        "cfo_hz": cfo_hz,
        "noise_power": noise_power,
        "active_power": active_power,
        "signal_power": signal_power,
        "snr_db": float(10.0 * np.log10(signal_power / noise_power)),
        "maximum_magnitude": float(np.max(np.abs(samples))),
        "evm_percent": 100.0 * float(np.median([item["rms_evm"] for item in frames])),
        "ser": float(np.mean([item["ser"] for item in frames])),
        "ber": float(np.mean([item["ber"] for item in frames])),
        "decoded_prefixes": [item["decoded_prefix"] for item in frames],
        "payload_equalized": np.concatenate([item["payload_equalized"] for item in frames]),
        "correlation_lags": selected,
        "correlation_peak": float(np.max(np.abs(correlation))),
        "block_peak_db": float(10.0 * np.log10(np.max(block_power))),
    }


def make_synthetic_psk_capture(
    reference: dict[str, object],
    output_dir: Path,
    *,
    true_lag: int = 5000,
    frame_count: int = 6,
    true_cfo_hz: float = 85.0,
    snr_db: float = 24.0,
) -> tuple[Path, Path, dict[str, object]]:
    """Write a deterministic repeated-frame capture and noise-only record."""
    period = np.asarray(reference["period"], dtype=np.complex128)
    sample_rate = float(reference["sample_rate_sps"])
    rng = np.random.default_rng(909)
    signal_samples = np.tile(period, frame_count)
    channel = 0.006 * np.exp(1j * 0.72)
    active = channel * signal_samples
    active_power = float(np.mean(np.abs(active) ** 2))
    noise_power = active_power / 10.0 ** (snr_db / 10.0)
    capture_count = true_lag + active.size + 5000
    noise = np.sqrt(noise_power / 2.0) * (
        rng.standard_normal(capture_count) + 1j * rng.standard_normal(capture_count)
    )
    capture = noise.copy()
    index = true_lag + np.arange(active.size)
    capture[true_lag : true_lag + active.size] += active * np.exp(
        1j * 2.0 * np.pi * true_cfo_hz * index / sample_rate
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_path = output_dir / "task3_synthetic_capture_c64.dat"
    noise_path = output_dir / "task3_synthetic_noise_c64.dat"
    capture.astype(np.complex64).tofile(capture_path)
    noise.astype(np.complex64).tofile(noise_path)
    truth = {
        "true_lag": true_lag,
        "frame_count": frame_count,
        "true_cfo_hz": true_cfo_hz,
        "snr_db": snr_db,
    }
    return capture_path, noise_path, truth


def plot_synthetic_psk_result(
    capture_path: Path,
    result: dict[str, object],
    reference: dict[str, object],
    output_dir: Path,
) -> Path:
    capture = np.memmap(capture_path, dtype=np.complex64, mode="r")
    payload = np.asarray(result["payload_equalized"])
    ideal = psk_constellation(int(reference["order"]))
    lags = np.asarray(result["correlation_lags"])
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.4), constrained_layout=True)
    axes[0].plot(np.abs(capture), linewidth=0.8)
    for lag in lags:
        axes[0].axvline(lag, color="tab:red", linewidth=0.8)
    axes[0].set(title="Synthetic finite capture", xlabel="Sample index", ylabel="Magnitude")
    axes[1].stem(np.arange(lags.size), lags, basefmt=" ")
    axes[1].set(title="Recovered frame starts", xlabel="Frame index", ylabel="Sample lag")
    axes[2].scatter(payload.real, payload.imag, s=4, alpha=0.18)
    axes[2].scatter(ideal.real, ideal.imag, marker="x", s=55, color="black")
    axes[2].set(title="Corrected QPSK payload", xlabel="I", ylabel="Q", aspect="equal")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    return save_figure(fig, output_dir, "task3_synthetic_reference_validation")


def plot_task3_measurements(
    records: dict[str, dict[str, object]], output_dir: Path
) -> tuple[Path, Path]:
    keys = list(records)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0), constrained_layout=True)
    for axis, key in zip(axes.flat, keys):
        record = records[key]
        payload = np.asarray(record["payload_equalized"])
        order = 4 if "m4" in key else 8
        ideal = psk_constellation(order)
        axis.scatter(payload.real, payload.imag, s=2, alpha=0.10)
        axis.scatter(ideal.real, ideal.imag, marker="x", s=55, color="black")
        axis.set(
            title=f"{key}: SNR {record['snr_db']:.1f} dB, EVM {record['evm_percent']:.1f}%",
            xlabel="I",
            ylabel="Q",
            aspect="equal",
        )
        axis.grid(True, alpha=0.25)
    constellation_path = save_figure(fig, output_dir, "task3_real_constellations")

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5), constrained_layout=True)
    for order, label in ((4, "QPSK"), (8, "8-PSK")):
        selected = [(key, value) for key, value in records.items() if f"m{order}_" in key]
        gains = np.asarray([int(key.split("gain")[1]) for key, _ in selected])
        order_index = np.argsort(gains)
        values = [selected[index][1] for index in order_index]
        gains = gains[order_index]
        axes[0].plot(gains, [value["snr_db"] for value in values], "o-", label=label)
        axes[1].plot(gains, [value["evm_percent"] for value in values], "o-", label=label)
        axes[2].semilogy(gains, np.maximum([value["ber"] for value in values], 1e-5), "o-", label=label)
    axes[0].set(title="Measured SNR", xlabel="TX gain (dB)", ylabel="SNR (dB)")
    axes[1].set(title="Measured RMS EVM", xlabel="TX gain (dB)", ylabel="EVM (%)")
    axes[2].set(title="Uncoded BER", xlabel="TX gain (dB)", ylabel="BER")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    metrics_path = save_figure(fig, output_dir, "task3_real_metrics")
    return constellation_path, metrics_path


def generate_tone_file(
    output_path: Path,
    sample_rate_sps: float,
    tone_hz: float,
    duration_s: float,
    amplitude: float,
) -> np.ndarray:
    count = round(sample_rate_sps * duration_s)
    index = np.arange(count)
    waveform = (
        amplitude * np.exp(1j * 2.0 * np.pi * tone_hz * index / sample_rate_sps)
    ).astype(np.complex64)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    waveform.tofile(output_path)
    return waveform


def analyze_tone_capture(
    capture_path: Path,
    sample_rate_sps: float,
    tone_hz: float,
) -> dict[str, object]:
    samples = np.memmap(capture_path, dtype=np.complex64, mode="r")
    block = round(0.002 * sample_rate_sps)
    blocks = np.asarray(samples[: (samples.size // block) * block]).reshape(-1, block)
    powers = np.mean(np.abs(blocks - np.mean(blocks, axis=1, keepdims=True)) ** 2, axis=1)
    threshold = np.median(powers) * 10 ** (3.0 / 10.0)
    active = powers > threshold
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    if starts.size == 0:
        # The reliable classroom sequence starts the finite transmitter first,
        # so a tone can be present throughout the complete receive record.
        start = 0
        stop = int(samples.size)
    else:
        lengths = stops - starts
        choice = int(np.argmax(lengths))
        start = int(starts[choice] * block)
        stop = int(stops[choice] * block)
    margin = min(round(0.01 * sample_rate_sps), max((stop - start) // 10, 1))
    segment = np.asarray(samples[start + margin : stop - margin])
    records = {
        "Original 1.25 MS/s": (segment, sample_rate_sps),
        "Raw factor 2: 625 kS/s": (segment[::2], sample_rate_sps / 2),
        "Filtered ratio 7/10: 875 kS/s": (signal.resample_poly(segment, 7, 10), sample_rate_sps * 7 / 10),
        "Raw factor 20: 62.5 kS/s": (segment[::20], sample_rate_sps / 20),
        "Filtered factor 20: 62.5 kS/s": (signal.resample_poly(segment, 1, 20), sample_rate_sps / 20),
    }
    peaks = {}
    powers_db = {}
    for name, (values, rate) in records.items():
        frequency, psd = centered_welch(values, rate)
        peaks[name] = float(frequency[np.argmax(psd)])
        powers_db[name] = float(10 * np.log10(normalized_power(values)))
    return {
        "capture_path": str(Path(capture_path)),
        "sample_count": int(samples.size),
        "file_bytes": int(Path(capture_path).stat().st_size),
        "burst_start": start,
        "burst_stop": stop,
        "tone_hz": tone_hz,
        "records": records,
        "peaks_hz": peaks,
        "powers_db": powers_db,
        "expected_factor20_alias_hz": principal_alias(tone_hz, sample_rate_sps / 20),
    }


def plot_task4_analysis(result: dict[str, object], output_dir: Path) -> tuple[Path, Path]:
    displayed = [
        "Original 1.25 MS/s",
        "Raw factor 2: 625 kS/s",
        "Filtered ratio 7/10: 875 kS/s",
        "Raw factor 20: 62.5 kS/s",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.7), constrained_layout=True)
    for axis, name in zip(axes.flat, displayed):
        values, rate = result["records"][name]
        frequency, psd = centered_welch(values, rate)
        peak = result["peaks_hz"][name]
        axis.plot(frequency / 1e3, 10.0 * np.log10(np.maximum(psd, TINY)))
        axis.axvline(peak / 1e3, color="tab:red", linestyle="--", linewidth=0.9)
        axis.set(title=name, xlabel="Baseband frequency (kHz)", ylabel="PSD (normalized dB/Hz)")
        axis.grid(True, alpha=0.25)
    spectra = save_figure(fig, output_dir, "task4_real_multirate_spectra")

    raw, rate = result["records"]["Raw factor 20: 62.5 kS/s"]
    filtered, _ = result["records"]["Filtered factor 20: 62.5 kS/s"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), constrained_layout=True)
    shown = min(60, raw.size)
    time_s = np.arange(shown) / rate
    axes[0].plot(time_s * 1e3, raw[:shown].real, label="raw I")
    axes[0].plot(time_s * 1e3, raw[:shown].imag, label="raw Q", alpha=0.8)
    axes[0].set(title=f"Raw factor 20: {result['peaks_hz']['Raw factor 20: 62.5 kS/s']/1e3:.2f} kHz", xlabel="Time (ms)", ylabel="Amplitude")
    axes[0].legend()
    for name, values in (("raw downsampling", raw), ("anti-aliased resampling", filtered)):
        frequency, psd = centered_welch(values, rate)
        axes[1].plot(frequency / 1e3, 10.0 * np.log10(np.maximum(psd, TINY)), label=name)
    axes[1].set(title="62.5 kS/s raw versus filtered", xlabel="Baseband frequency (kHz)", ylabel="PSD (normalized dB/Hz)")
    axes[1].legend()
    for axis in axes:
        axis.grid(True, alpha=0.25)
    alias = save_figure(fig, output_dir, "task4_real_aliasing")
    return spectra, alias


def json_ready(value: object) -> object:
    """Convert NumPy-containing result dictionaries before saving summaries."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value
