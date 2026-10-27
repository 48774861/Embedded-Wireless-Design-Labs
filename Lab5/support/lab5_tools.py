#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab5_dsp as lab


BLUE = "#1f4e79"
ORANGE = "#d97706"
GREEN = "#238636"
RED = "#b42318"
PURPLE = "#7e22ce"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-dir", type=Path)
    return parser.parse_args()


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def prepare_hardware_experiment(
    waveform_dir: Path,
    capture_dir: Path,
    tx_gains_db: tuple[int, int, int],
    rx_gain_db: float,
    sample_rate_sps: float = 1_000_000.0,
    center_frequency_hz: float = 2_450_000_000.0,
    repeat_count: int = 13,
    acquisition_lead_samples: int = 2_000_000,
    capture_duration_s: float = 15.0,
) -> dict[str, object]:
    """Write the frozen finite TX file and the hardware-analysis metadata."""
    gains = tuple(int(value) for value in tx_gains_db)
    if len(gains) != 3 or len(set(gains)) != 3 or tuple(sorted(gains)) != gains:
        raise ValueError("Enter three distinct TX gains in increasing order.")
    if not np.isfinite(rx_gain_db):
        raise ValueError("Enter the instructor-assigned RX gain.")

    waveform_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    reference = lab.build_reference_frame(0)
    period = np.asarray(reference.tx_period, dtype=np.complex64)
    guarded = np.concatenate(
        (
            np.zeros(acquisition_lead_samples, dtype=np.complex64),
            np.tile(period, repeat_count),
        )
    )
    waveform_path = waveform_dir / "task4_ofdm_13period_c64.dat"
    guarded.tofile(waveform_path)

    summary = {
        "waveform": {
            "tx_gains_db": list(gains),
            "measurement_periods": repeat_count - 1,
            "warmup_periods": 1,
            "repeat_count": repeat_count,
            "capture_duration_s": capture_duration_s,
            "capture_samples": int(round(capture_duration_s * sample_rate_sps)),
            "sample_rate_sps": sample_rate_sps,
            "center_frequency_hz": center_frequency_hz,
            "rx_gain_db": float(rx_gain_db),
            "period_samples": int(period.size),
            "active_frame_samples": int(reference.active_frame_length),
            "acquisition_lead_samples": acquisition_lead_samples,
            "finite_tx_samples": int(guarded.size),
            "digital_peak": float(np.max(np.abs(guarded))),
            "period_sha256": hashlib.sha256(period.tobytes()).hexdigest(),
            "file_sha256": hashlib.sha256(guarded.tobytes()).hexdigest(),
        }
    }
    summary_path = capture_dir / "lab5_measurement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "waveform_path": str(waveform_path),
        "waveform_bytes": waveform_path.stat().st_size,
        "summary_path": str(summary_path),
        "summary": summary,
    }


def scatter(axis: plt.Axes, values: np.ndarray, title: str) -> None:
    z = np.asarray(values).reshape(-1)
    axis.scatter(z.real, z.imag, s=5, alpha=0.35, color=BLUE, edgecolors="none")
    ideal = lab.qpsk_map(np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8))
    axis.scatter(
        ideal.real, ideal.imag, s=45, facecolors="none", edgecolors=RED, linewidths=1.2
    )
    axis.axhline(0, color="0.7", lw=0.7)
    axis.axvline(0, color="0.7", lw=0.7)
    axis.set(xlabel="In-phase", ylabel="Quadrature", title=title)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)


def channel_case(reference, channel, snr_db, seed=2205):
    channel_output = np.convolve(reference.logical_period, channel)
    start = reference.training_start
    stop = reference.payload_start + lab.N_PAYLOAD_SYMBOLS * (lab.NFFT + reference.ncp)
    signal_power = float(np.mean(np.abs(channel_output[start:stop]) ** 2))
    noise_power = signal_power * 10.0 ** (-snr_db / 10.0)
    noisy = lab.add_complex_awgn(
        channel_output, noise_power, np.random.default_rng(seed)
    )
    result = lab.decode_at_training_start(noisy, start, reference)
    return noisy, result, noise_power


def generate_simulation(output_dir: Path) -> dict[str, object]:
    reference = lab.build_reference_frame(0)
    ideal = lab.ideal_round_trip(reference)

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), constrained_layout=True)
    active = reference.logical_period[: reference.active_frame_length]
    axes[0].plot(np.abs(active), lw=0.75, color=BLUE)
    regions = [
        (0, 128, "guard", "#d9e2f3"),
        (128, 192, "short preamble", "#fde68a"),
        (192, 352, "training", "#bbf7d0"),
        (352, 1952, "20 payload symbols", "#bfdbfe"),
        (1952, 2080, "guard", "#d9e2f3"),
    ]
    for left, right, label, color in regions:
        axes[0].axvspan(left, right, color=color, alpha=0.35, lw=0)
        axes[0].text((left + right) / 2, 1.05, label, ha="center", va="bottom", fontsize=8)
    axes[0].set(
        xlim=(0, reference.active_frame_length),
        xlabel="Sample index",
        ylabel="Magnitude",
        title="Deterministic Lab 5 active frame",
    )
    axes[0].grid(alpha=0.2)
    classification = np.zeros((lab.N_PAYLOAD_SYMBOLS, lab.NFFT), dtype=int)
    classification[:, lab.DATA_IDX] = 1
    classification[:, lab.PILOT_IDX] = 2
    image = axes[1].imshow(
        classification,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=matplotlib.colors.ListedColormap(["#e5e7eb", "#60a5fa", "#f59e0b"]),
    )
    colorbar = fig.colorbar(image, ax=axes[1], ticks=[0, 1, 2], pad=0.02)
    colorbar.ax.set_yticklabels(["null", "data", "pilot"])
    axes[1].set(
        xlabel="Unshifted NumPy FFT-array index",
        ylabel="Payload OFDM symbol",
        title="20 x 64 resource-grid allocation",
    )
    save_figure(fig, output_dir, "task1_frame_and_resource_grid")

    first_cp = lab.ofdm_modulate(reference.payload_grid[:1], reference.ncp)[0]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    signed_order = np.fft.fftshift(np.arange(lab.NFFT))
    signed_axis = np.arange(-lab.NFFT // 2, lab.NFFT // 2)
    axes[0].stem(
        signed_axis,
        np.abs(reference.payload_grid[0, signed_order]),
        basefmt=" ",
        linefmt=BLUE,
        markerfmt="o",
    )
    axes[0].set(xlabel="Signed subcarrier", ylabel="Magnitude", title="One grid row")
    axes[1].plot(first_cp[lab.NCP:].real, label="I", color=BLUE)
    axes[1].plot(first_cp[lab.NCP:].imag, label="Q", color=ORANGE)
    axes[1].set(xlabel="Useful-sample index", ylabel="Amplitude", title="64-sample IFFT output")
    axes[1].legend()
    axes[2].plot(first_cp.real, label="I", color=BLUE)
    axes[2].plot(first_cp.imag, label="Q", color=ORANGE)
    axes[2].axvspan(0, lab.NCP - 1, color="#fde68a", alpha=0.4, label="cyclic prefix")
    axes[2].set(xlabel="CP-extended sample index", ylabel="Amplitude", title="80-sample transmitted symbol")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    save_figure(fig, output_dir, "task1_ofdm_symbol")

    awgn_rows = lab.awgn_sweep(frames_per_point=100)
    ebn0 = np.array([row["ebn0_db"] for row in awgn_rows])
    ber = np.array([row["ber"] for row in awgn_rows])
    theory = np.array([row["theory"] for row in awgn_rows])
    evm = np.array([row["evm_rms"] for row in awgn_rows])
    resolution = 1.0 / float(awgn_rows[0]["bits"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), constrained_layout=True)
    axes[0].semilogy(ebn0, theory, color=BLUE, label="QPSK theory")
    axes[0].semilogy(ebn0, np.maximum(ber, resolution), "o", color=ORANGE, label="100-frame experiment")
    for x_value, observed, y_value in zip(ebn0, ber, np.maximum(ber, resolution)):
        if observed == 0:
            axes[0].annotate("0 errors", (x_value, y_value), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7)
    axes[0].set(xlabel="$E_b/N_0$ (dB)", ylabel="BER", title="QPSK AWGN validation", ylim=(1e-6, 2e-1))
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].plot(ebn0, 20.0 * np.log10(evm), "o-", color=GREEN)
    axes[1].set(xlabel="$E_b/N_0$ (dB)", ylabel="RMS EVM (dB)", title="Post-FFT data-carrier EVM")
    axes[1].grid(alpha=0.25)
    save_figure(fig, output_dir, "task2_awgn_ber_evm")

    short_h = lab.short_channel()
    noisy_short, short_result, _ = channel_case(reference, short_h, 20.0)
    true_h = np.fft.fft(short_h, lab.NFFT)
    payload_cp = lab._extract_cp_symbols(
        noisy_short, reference.payload_start, lab.N_PAYLOAD_SYMBOLS, reference.ncp
    )
    raw_grid = lab.ofdm_demodulate(payload_cp, reference.ncp)
    raw_data = raw_grid[:, lab.DATA_IDX]
    true_equalized = raw_data / true_h[lab.DATA_IDX]
    estimated_equalized = short_result["corrected_grid"][:, lab.DATA_IDX]
    short_nmse = lab.nmse(short_result["channel_estimate"], true_h)
    short_rows = {
        "raw_ber": lab.bit_error_rate(lab.qpsk_detect(raw_data), reference.payload_bits),
        "raw_evm_rms": lab.evm_rms(raw_data, reference.payload_grid[:, lab.DATA_IDX]),
        "true_h_ber": lab.bit_error_rate(lab.qpsk_detect(true_equalized), reference.payload_bits),
        "true_h_evm_rms": lab.evm_rms(true_equalized, reference.payload_grid[:, lab.DATA_IDX]),
        "estimated_h_ber": short_result["ber"],
        "estimated_h_evm_rms": short_result["evm_rms"],
        "channel_nmse": short_nmse,
    }
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.4), constrained_layout=True)
    axes[0, 0].plot(lab.ACTIVE_SIGNED, np.abs(true_h[lab.ACTIVE_IDX]), color=BLUE, label="true")
    axes[0, 0].plot(lab.ACTIVE_SIGNED, np.abs(short_result["channel_estimate"][lab.ACTIVE_IDX]), "o", ms=2.5, color=ORANGE, label="two-symbol LS")
    axes[0, 0].set(xlabel="Signed subcarrier", ylabel="$|H[k]|$", title="Channel magnitude")
    axes[0, 0].legend()
    axes[0, 1].plot(lab.ACTIVE_SIGNED, np.unwrap(np.angle(true_h[lab.ACTIVE_IDX])), color=BLUE)
    axes[0, 1].plot(lab.ACTIVE_SIGNED, np.unwrap(np.angle(short_result["channel_estimate"][lab.ACTIVE_IDX])), "o", ms=2.5, color=ORANGE)
    axes[0, 1].set(xlabel="Signed subcarrier", ylabel="Phase (rad)", title="Channel phase")
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02,
        0.96,
        f"20 dB sample-domain SNR\n\n"
        f"No EQ: BER {short_rows['raw_ber']:.4g}, EVM {short_rows['raw_evm_rms']:.3f}\n"
        f"True H: BER {short_rows['true_h_ber']:.4g}, EVM {short_rows['true_h_evm_rms']:.3f}\n"
        f"Estimated H: BER {short_rows['estimated_h_ber']:.4g}, EVM {short_rows['estimated_h_evm_rms']:.3f}\n"
        f"Channel NMSE: {10*np.log10(short_nmse):.2f} dB",
        va="top",
        fontsize=10,
    )
    scatter(axes[1, 0], raw_data, "Before equalization")
    scatter(axes[1, 1], true_equalized, "True-channel equalization")
    scatter(axes[1, 2], estimated_equalized, "Training-based equalization")
    for axis in axes[0, :2]:
        axis.grid(alpha=0.25)
    save_figure(fig, output_dir, "task2_multipath_equalization")

    cp_rows = []
    for ncp in (16, 24):
        cp_reference = lab.build_reference_frame(0, ncp=ncp)
        _, cp_result, _ = channel_case(cp_reference, lab.long_channel(), 20.0, seed=3024)
        cp_rows.append({"ncp": ncp, "ber": cp_result["ber"], "evm_rms": cp_result["evm_rms"]})
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), constrained_layout=True)
    labels = [f"CP = {row['ncp']}" for row in cp_rows]
    axes[0].bar(labels, [row["ber"] for row in cp_rows], color=[RED, GREEN], width=0.62)
    axes[0].set(ylabel="BER", title="20-sample delayed path")
    axes[1].bar(labels, [row["evm_rms"] for row in cp_rows], color=[RED, GREEN], width=0.62)
    axes[1].set(ylabel="RMS EVM", title="One-tap FDE result")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "task2_cp_comparison")

    simulated_capture, truth = lab.simulate_impaired_capture(reference, snr_db=20.0)
    recovered, diagnostics = lab.recover_capture(simulated_capture, reference)
    acquired = recovered[0]
    metric = diagnostics["repetition_metric"]
    left = truth["preamble_start"] - 160
    right = truth["preamble_start"] + 220
    fig, axes = plt.subplots(2, 1, figsize=(10.0, 5.5), constrained_layout=True)
    axes[0].plot(np.arange(left, right), metric[left:right], color=BLUE)
    axes[0].axhline(diagnostics["threshold"], color=RED, ls="--", label="fixed threshold")
    axes[0].axvline(truth["preamble_start"], color=GREEN, ls=":", label="true preamble")
    axes[0].set(xlabel="Capture sample", ylabel="$M[d]$", title="Repeated-preamble coarse acquisition")
    axes[0].legend()
    axes[1].plot(acquired["timing_search_starts"], acquired["timing_correlation"], color=ORANGE)
    axes[1].axvline(truth["training_start"], color=GREEN, ls=":", label="true CP boundary")
    axes[1].axvline(acquired["fft_window_start"], color=PURPLE, ls="--", label="FFT-window start")
    axes[1].set(xlabel="Candidate first-training CP sample", ylabel="Normalized correlation", title="Fine timing and fixed four-sample CP advance")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure(fig, output_dir, "task3_synthetic_acquisition")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    scatter(axes[0], acquired["received_payload_grid"][:, lab.DATA_IDX], "Raw FFT carriers")
    scatter(axes[1], acquired["equalized_grid"][:, lab.DATA_IDX], "After ZF")
    scatter(axes[2], acquired["corrected_grid"][:, lab.DATA_IDX], "After pilot CPE")
    save_figure(fig, output_dir, "task3_synthetic_constellations")

    fig, ax = plt.subplots(figsize=(8.0, 3.0), constrained_layout=True)
    ax.plot(np.arange(1, 21), np.unwrap(acquired["cpe_rad"]), "o-", color=BLUE)
    ax.set(xlabel="Payload OFDM symbol", ylabel="Estimated CPE (rad)", title="Four-pilot residual phase tracking")
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(alpha=0.25)
    save_figure(fig, output_dir, "task3_pilot_cpe")

    chooser = np.random.default_rng(5503)
    monte_rows = []
    for _ in range(100):
        delay = int(chooser.integers(300, 2001))
        cfo_hz = float(chooser.uniform(-20_000, 20_000))
        phase_rad = float(chooser.uniform(-np.pi, np.pi))
        noise_seed = int(chooser.integers(0, 2**31 - 1))
        capture, case_truth = lab.simulate_impaired_capture(
            reference,
            delay=delay,
            cfo_hz=cfo_hz,
            phase_rad=phase_rad,
            snr_db=20.0,
            seed=noise_seed,
        )
        case_recovered, _ = lab.recover_capture(capture, reference)
        if not case_recovered:
            monte_rows.append({"detected": False})
            continue
        result = min(
            case_recovered,
            key=lambda item: abs(item["training_start"] - case_truth["training_start"]),
        )
        monte_rows.append(
            {
                "detected": True,
                "timing_error_samples": int(result["training_start"] - case_truth["training_start"]),
                "cfo_error_hz": float(result["cfo_hz"] - case_truth["cfo_hz"]),
                "ber": float(result["ber"]),
                "evm_rms": float(result["evm_rms"]),
                "crc_ok": bool(result["crc_ok"]),
            }
        )
    successful = [row for row in monte_rows if row["detected"]]
    monte_summary = {
        "trials": len(monte_rows),
        "detected": len(successful),
        "detection_rate": len(successful) / len(monte_rows),
        "timing_within_one_rate": float(np.mean([abs(row["timing_error_samples"]) <= 1 for row in successful])),
        "cfo_rmse_hz": float(np.sqrt(np.mean([row["cfo_error_hz"] ** 2 for row in successful]))),
        "ber": float(np.mean([row["ber"] for row in successful])),
        "packet_error_rate": float(1.0 - np.mean([row["crc_ok"] for row in successful])),
        "mean_evm_rms": float(np.mean([row["evm_rms"] for row in successful])),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5), constrained_layout=True)
    axes[0].hist([row["cfo_error_hz"] for row in successful], bins=15, color=BLUE, alpha=0.85)
    axes[0].set(xlabel="CFO error (Hz)", ylabel="Trials", title=f"CFO RMSE = {monte_summary['cfo_rmse_hz']:.1f} Hz")
    axes[1].hist(100.0 * np.array([row["evm_rms"] for row in successful]), bins=15, color=GREEN, alpha=0.85)
    axes[1].set(xlabel="RMS EVM (%)", ylabel="Trials", title=f"PER = {100*monte_summary['packet_error_rate']:.1f}%")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    save_figure(fig, output_dir, "task3_monte_carlo")

    return {
        "reference": {
            "active_frame_samples": reference.active_frame_length,
            "period_samples": int(reference.logical_period.size),
            "tx_scale": reference.tx_scale,
            "digital_peak": float(np.max(np.abs(reference.tx_period))),
            "papr_db": reference.papr_db,
        },
        "ideal": ideal,
        "awgn": awgn_rows,
        "short_channel": short_rows,
        "cp_comparison": cp_rows,
        "synthetic_acquisition": {
            "true_cfo_hz": truth["cfo_hz"],
            "estimated_cfo_hz": acquired["cfo_hz"],
            "cfo_error_hz": acquired["cfo_hz"] - truth["cfo_hz"],
            "timing_error_samples": acquired["training_start"] - truth["training_start"],
            "ber": acquired["ber"],
            "evm_rms": acquired["evm_rms"],
            "crc_ok": acquired["crc_ok"],
        },
        "monte_carlo": monte_summary,
    }


def analyze_hardware(hardware_dir: Path, output_dir: Path) -> dict[str, object]:
    summary = json.loads((hardware_dir / "lab5_measurement_summary.json").read_text())
    reference = lab.build_reference_frame(0)
    noise_path = hardware_dir / "lab5_noise_c64.dat"
    if not noise_path.is_file():
        raise FileNotFoundError(f"Missing transmitter-off noise file: {noise_path}")
    noise = np.fromfile(noise_path, dtype=np.complex64)
    if noise.size != 250_000:
        raise ValueError(f"Noise file must contain 250000 complex64 samples, found {noise.size}.")
    noise_power = float(np.mean(np.abs(noise - np.mean(noise)) ** 2))
    expected_measurement_frames = int(summary["waveform"]["measurement_periods"])
    expected_capture_samples = int(summary["waveform"]["capture_samples"])
    gain_results: dict[int, dict[str, object]] = {}

    gains = tuple(int(value) for value in summary["waveform"]["tx_gains_db"])
    for gain in gains:
        capture_path = hardware_dir / f"lab5_ofdm_tx{gain}_c64.dat"
        if not capture_path.is_file():
            raise FileNotFoundError(f"Missing gain capture: {capture_path}")
        capture = np.fromfile(capture_path, dtype=np.complex64)
        if capture.size != expected_capture_samples:
            raise ValueError(
                f"{capture_path.name} must contain {expected_capture_samples} complex64 "
                f"samples, found {capture.size}."
            )
        recovered, diagnostics = lab.recover_capture(capture, reference)
        analyzed = recovered[1:] if len(recovered) > expected_measurement_frames else recovered
        frame_rows = []
        for result in analyzed:
            bit_errors = int(np.count_nonzero(result["decoded_bits"] != reference.payload_bits))
            channel_power = float(np.mean(np.abs(result["channel_estimate"][lab.ACTIVE_IDX]) ** 2))
            frame_rows.append(
                {
                    "frame_start": int(result["frame_start"]),
                    "training_start": int(result["training_start"]),
                    "fft_window_start": int(result["fft_window_start"]),
                    "fine_correlation": float(np.max(result["timing_correlation"])),
                    "cfo_hz": float(result["cfo_hz"]),
                    "bit_errors": bit_errors,
                    "ber": float(result["ber"]),
                    "evm_rms": float(result["evm_rms"]),
                    "crc_ok": bool(result["crc_ok"]),
                    "channel_power": channel_power,
                }
            )
        if frame_rows:
            aggregate = {
                "detected_frames_including_warmup": len(recovered),
                "analyzed_frames": len(frame_rows),
                "detection_rate": min(1.0, len(frame_rows) / expected_measurement_frames),
                "bit_errors": int(sum(row["bit_errors"] for row in frame_rows)),
                "ber": float(np.mean([row["ber"] for row in frame_rows])),
                "packet_error_rate": float(1.0 - np.mean([row["crc_ok"] for row in frame_rows])),
                "evm_rms": float(np.sqrt(np.mean([row["evm_rms"] ** 2 for row in frame_rows]))),
                "cfo_mean_hz": float(np.mean([row["cfo_hz"] for row in frame_rows])),
                "cfo_std_hz": float(np.std([row["cfo_hz"] for row in frame_rows])),
                "fine_correlation_mean": float(np.mean([row["fine_correlation"] for row in frame_rows])),
                "channel_power": float(np.mean([row["channel_power"] for row in frame_rows])),
                "snr_estimate_db": float(10.0 * np.log10(np.mean([row["channel_power"] for row in frame_rows]) / noise_power)),
                "capture_peak": float(np.max(np.abs(capture))),
                "clipped_sample_count": int(np.count_nonzero(np.abs(capture) >= 0.99)),
            }
        else:
            aggregate = {
                "detected_frames_including_warmup": 0,
                "analyzed_frames": 0,
                "detection_rate": 0.0,
                "bit_errors": None,
                "ber": None,
                "packet_error_rate": None,
                "evm_rms": None,
                "cfo_mean_hz": None,
                "cfo_std_hz": None,
                "fine_correlation_mean": None,
                "channel_power": None,
                "snr_estimate_db": None,
                "capture_peak": float(np.max(np.abs(capture))),
                "clipped_sample_count": int(np.count_nonzero(np.abs(capture) >= 0.99)),
            }
        gain_results[gain] = {
            "capture": capture,
            "recovered": recovered,
            "analyzed": analyzed,
            "diagnostics": diagnostics,
            "frames": frame_rows,
            "aggregate": aggregate,
        }

    baseline_gain = gains[len(gains) // 2]
    baseline = gain_results[baseline_gain]
    capture = baseline["capture"]
    diagnostics = baseline["diagnostics"]
    recovered = baseline["recovered"]
    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), constrained_layout=True)
    step = 50
    axes[0].plot(np.arange(0, capture.size, step), np.abs(capture[::step]), lw=0.55, color=BLUE)
    for result in recovered:
        axes[0].axvline(result["frame_start"], color=ORANGE, alpha=0.45, lw=0.8)
    axes[0].set(xlabel="Capture sample", ylabel="Magnitude", title=f"Uncropped TX {baseline_gain} dB capture and detected frames")
    metric = diagnostics["repetition_metric"]
    metric_left = max(0, int(recovered[0]["coarse_preamble_start"]) - 4_000)
    metric_right = min(metric.size, int(recovered[-1]["coarse_preamble_start"]) + 4_000)
    metric_indices = np.arange(metric_left, metric_right)
    axes[1].plot(metric_indices, metric[metric_left:metric_right], lw=0.55, color=GREEN)
    axes[1].axhline(diagnostics["threshold"], color=RED, ls="--", label="fixed threshold")
    axes[1].set(xlabel="Metric sample", ylabel="$M[d]$", title="Repeated-preamble metric over the uncropped capture")
    axes[1].legend()
    first = baseline["analyzed"][0]
    axes[2].plot(first["timing_search_starts"], first["timing_correlation"], color=PURPLE)
    axes[2].axvline(first["training_start"], color=GREEN, ls=":", label="fine boundary")
    axes[2].axvline(first["fft_window_start"], color=ORANGE, ls="--", label="FFT-window start")
    axes[2].set(xlabel="Candidate first-training CP sample", ylabel="Normalized correlation", title="Measured fine timing and CP backoff")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    save_figure(fig, output_dir, "task4_hardware_acquisition")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    raw_hardware = first["received_payload_grid"][:, lab.DATA_IDX]
    raw_display = raw_hardware / np.sqrt(np.mean(np.abs(raw_hardware) ** 2))
    scatter(axes[0], raw_display, "Raw FFT carriers (display-normalized)")
    scatter(axes[1], first["equalized_grid"][:, lab.DATA_IDX], "After per-carrier ZF")
    scatter(axes[2], first["corrected_grid"][:, lab.DATA_IDX], "After four-pilot CPE")
    save_figure(fig, output_dir, "task4_hardware_constellations")

    fig, axes = plt.subplots(2, 1, figsize=(10.0, 6.2), constrained_layout=True)
    for index, result in enumerate(baseline["analyzed"][:3]):
        h = result["channel_estimate"][lab.ACTIVE_IDX]
        axes[0].plot(lab.ACTIVE_SIGNED, 20.0 * np.log10(np.maximum(np.abs(h), 1e-12)), marker="o", ms=2.5, label=f"frame {index + 1}")
        axes[1].plot(lab.ACTIVE_SIGNED, np.unwrap(np.angle(h)), marker="o", ms=2.5, label=f"frame {index + 1}")
    axes[0].set(xlabel="Signed subcarrier", ylabel=r"$20\log_{10}|\hat H[k]|$ (dB)", title="Measured channel magnitude across three frames")
    axes[1].set(xlabel="Signed subcarrier", ylabel="Unwrapped phase (rad)", title="Measured channel phase across three frames")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=3)
    save_figure(fig, output_dir, "task4_hardware_channel_estimates")

    corrected = first["corrected_grid"][:, lab.DATA_IDX]
    reference_data = reference.payload_grid[:, lab.DATA_IDX]
    carrier_error = np.sqrt(np.mean(np.abs(corrected - reference_data) ** 2, axis=0))
    fig, ax = plt.subplots(figsize=(9.0, 3.2), constrained_layout=True)
    ax.bar(lab.DATA_SIGNED, 100.0 * carrier_error, color=BLUE, width=0.8)
    ax.set(xlabel="Signed data subcarrier", ylabel="RMS error (%)", title=f"Measured per-data-subcarrier error, TX {baseline_gain} dB baseline")
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "task4_hardware_subcarrier_error")

    gains = np.array(gains)
    aggregates = [gain_results[int(gain)]["aggregate"] for gain in gains]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    axes[0].plot(gains, [100 * row["detection_rate"] for row in aggregates], "o-", label="detected", color=BLUE)
    axes[0].plot(gains, [100 * (1 - row["packet_error_rate"]) if row["packet_error_rate"] is not None else np.nan for row in aggregates], "s--", label="CRC pass", color=GREEN)
    axes[0].set(xlabel="TX gain (dB)", ylabel="Rate (%)", title="Acquisition and packets", ylim=(-3, 105))
    axes[0].legend()
    axes[1].plot(gains, [row["snr_estimate_db"] for row in aggregates], "o-", color=ORANGE)
    axes[1].set(xlabel="TX gain (dB)", ylabel="Estimated SNR (dB)", title="Training/noise estimate")
    axes[2].semilogy(gains, [row["ber"] if row["ber"] and row["ber"] > 0 else 0.5 / (max(1, row["analyzed_frames"]) * 1920) for row in aggregates], "o-", color=RED, label="BER")
    axes[2].plot(gains, [row["evm_rms"] for row in aggregates], "s--", color=PURPLE, label="RMS EVM")
    axes[2].set(xlabel="TX gain (dB)", ylabel="BER / fractional EVM", title="Recovered payload quality")
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure(fig, output_dir, "task4_hardware_gain_metrics")

    serializable_results = {}
    for gain, result in gain_results.items():
        serializable_results[f"tx{gain}"] = {
            "aggregate": result["aggregate"],
            "frames": result["frames"],
        }
    return {
        "hardware_directory": str(hardware_dir),
        "noise_power": noise_power,
        "expected_measurement_frames": expected_measurement_frames,
        "streaming_metadata": summary,
        "gain_results": serializable_results,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {"simulation": generate_simulation(args.output_dir)}
    if args.hardware_dir is not None:
        results["hardware"] = analyze_hardware(args.hardware_dir, args.output_dir)
    (args.output_dir / "lab5_validation_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
