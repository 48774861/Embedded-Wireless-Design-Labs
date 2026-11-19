#!/usr/bin/env python3
"""Generate Lab 6 waveforms, figures, and hardware analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lab6_dsp as lab


BLUE = "#1f4e79"
ORANGE = "#d97706"
GREEN = "#238636"
RED = "#b42318"
PURPLE = "#7e22ce"


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
    tx_subdevice: str = "A:0 A:1",
    rx_subdevice: str = "A:0 A:1",
    tx_antenna: str = "TX/RX",
    rx_antenna: str = "RX2",
) -> dict[str, object]:
    gains = tuple(int(value) for value in tx_gains_db)
    if len(gains) != 3 or tuple(sorted(gains)) != gains or len(set(gains)) != 3:
        raise ValueError("Enter three distinct TX gains in increasing order.")
    if not np.isfinite(rx_gain_db):
        raise ValueError("Enter the instructor-assigned RX gain.")
    radio_strings = {
        "TX subdevice": tx_subdevice,
        "RX subdevice": rx_subdevice,
        "TX antenna": tx_antenna,
        "RX antenna": rx_antenna,
    }
    for label, value in radio_strings.items():
        if not isinstance(value, str) or not value.strip() or value.strip() == "...":
            raise ValueError(f"Enter the instructor-assigned {label} string.")
    waveform_dir = Path(waveform_dir)
    capture_dir = Path(capture_dir)
    waveform_dir.mkdir(parents=True, exist_ok=True)
    capture_dir.mkdir(parents=True, exist_ok=True)
    waveform = lab.build_waveform()
    tx0 = lab.make_transport(waveform.tx0_period)
    tx1 = lab.make_transport(waveform.tx1_period)
    tx0_path = waveform_dir / "lab6_tx_ch0_c64.dat"
    tx1_path = waveform_dir / "lab6_tx_ch1_c64.dat"
    tx0.tofile(tx0_path)
    tx1.tofile(tx1_path)
    summary = {
        "waveform": lab.waveform_metadata(waveform),
        "experiment": {
            "tx_gains_db": list(gains),
            "rx_gain_db": float(rx_gain_db),
            "noise_samples_per_channel": 250_000,
            "tx_subdevice": tx_subdevice.strip(),
            "rx_subdevice": rx_subdevice.strip(),
            "tx_antenna": tx_antenna.strip(),
            "rx_antenna": rx_antenna.strip(),
        },
    }
    summary_path = capture_dir / "lab6_measurement_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "tx0_path": str(tx0_path),
        "tx1_path": str(tx1_path),
        "bytes_per_tx_file": tx0_path.stat().st_size,
        "summary_path": str(summary_path),
        "summary": summary,
    }


def generate_synthetic_hardware(hardware_dir: Path) -> dict[str, object]:
    """Create deterministic two-channel captures for validating the full analysis path."""
    hardware_dir = Path(hardware_dir)
    prepared = prepare_hardware_experiment(hardware_dir.parent / "waveforms", hardware_dir, (40, 45, 50), 60.0)
    waveform = lab.build_waveform()
    tx = np.stack((lab.make_transport(waveform.tx0_period), lab.make_transport(waveform.tx1_period)))
    channel = np.array(
        [[0.76 + 0.18j, 0.23 - 0.31j], [0.34 + 0.42j, 0.69 - 0.12j]],
        dtype=np.complex128,
    )
    rng = np.random.default_rng(7606)
    delay = 347
    cfo_hz = 173.0
    noise_sigma = 2.5e-3
    noise = noise_sigma * (
        rng.standard_normal((2, 250_000)) + 1j * rng.standard_normal((2, 250_000))
    ) / np.sqrt(2.0)
    for branch in range(2):
        noise[branch].astype(np.complex64).tofile(hardware_dir / f"lab6_noise_ch{branch}_c64.dat")
    for gain in (40, 45, 50):
        scale = 0.42 * 10.0 ** ((gain - 45.0) / 20.0)
        clean = scale * channel @ tx
        capture = noise_sigma * (
            rng.standard_normal((2, lab.CAPTURE_SAMPLES))
            + 1j * rng.standard_normal((2, lab.CAPTURE_SAMPLES))
        ) / np.sqrt(2.0)
        stop = min(lab.CAPTURE_SAMPLES, delay + clean.shape[1])
        count = stop - delay
        phase = np.exp(1j * 2.0 * np.pi * cfo_hz * np.arange(count) / lab.FS)
        capture[:, delay:stop] += clean[:, :count] * phase[None, :]
        for branch in range(2):
            capture[branch].astype(np.complex64).tofile(
                hardware_dir / f"lab6_rx_tx{gain}_ch{branch}_c64.dat"
            )
    truth = {
        "channel": _complex_matrix(channel),
        "delay_samples": delay,
        "cfo_hz": cfo_hz,
        "noise_sigma": noise_sigma,
        "prepared": prepared,
    }
    (hardware_dir / "synthetic_truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    return truth


def _plot_constellation(axis: plt.Axes, values: np.ndarray, title: str) -> None:
    z = np.asarray(values).reshape(-1)
    axis.scatter(z.real, z.imag, s=5, alpha=0.35, color=BLUE, edgecolors="none")
    ideal = lab.qpsk_map(np.array([0, 0, 0, 1, 1, 1, 1, 0], dtype=np.uint8))
    axis.scatter(ideal.real, ideal.imag, s=45, facecolors="none", edgecolors=RED, linewidths=1.2)
    axis.axhline(0, color="0.7", lw=0.7)
    axis.axvline(0, color="0.7", lw=0.7)
    axis.set(xlabel="In-phase", ylabel="Quadrature", title=title)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.2)


def plot_diversity_outputs(outputs: dict[str, np.ndarray], output_path: Path) -> Path:
    """Plot already-computed diversity outputs so notebooks stay focused on combining math."""
    required = ("sc", "egc", "mrc")
    if any(name not in outputs for name in required):
        raise ValueError("outputs must contain sc, egc, and mrc")
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.7), constrained_layout=True)
    for axis, name, title in zip(
        axes,
        required,
        ("Selection combining", "Equal-gain combining", "Maximum-ratio combining"),
    ):
        _plot_constellation(axis, np.asarray(outputs[name]), title)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def generate_simulation(output_dir: Path) -> dict[str, object]:
    output_dir = Path(output_dir)
    mimo = lab.simulate_mimo_capacity()
    diversity = lab.simulate_diversity()
    waveform = lab.build_waveform()

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.1), constrained_layout=True)
    axes[0].plot(mimo["snr_db"], mimo["mimo_capacity_bps_hz"], "o-", color=BLUE, lw=2, label="2x2 MIMO")
    labels = ("h11", "h12", "h21", "h22")
    for index, label in enumerate(labels):
        axes[0].plot(mimo["snr_db"], mimo["siso_capacity_bps_hz"][:, index], "o--", lw=1, label=f"SISO {label}")
    axes[0].set(xlabel="Total SNR (dB)", ylabel="Spectral efficiency (bit/s/Hz)", title="Equal-total-power capacity")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].semilogy(mimo["snr_db"], mimo["channel_nmse"], "o-", color=ORANGE, label="LS channel NMSE")
    axes[1].set(xlabel="Pilot SNR (dB)", ylabel="Normalized MSE", title="2x2 LS channel estimation")
    axes[1].grid(alpha=0.25, which="both")
    save_figure(fig, output_dir, "task1_mimo_capacity_and_estimation")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1), constrained_layout=True)
    for axis, method, title in zip(axes, ("sc", "egc", "mrc"), ("Selection combining", "Equal-gain combining", "Maximum-ratio combining")):
        for li, branches in enumerate(diversity["branch_counts"]):
            observed = diversity["ber"][method][li]
            floor = 0.5 / diversity["bits_per_point"]
            axis.semilogy(diversity["snr_db"], np.maximum(observed, floor), "o-", label=f"L={branches}")
        axis.set(xlabel="Average branch SNR (dB)", ylabel="BER", title=title, ylim=(1e-6, 1))
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    save_figure(fig, output_dir, "task1_diversity_by_method")

    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    li = len(diversity["branch_counts"]) - 1
    for method, color, label in (("sc", BLUE, "SC"), ("egc", ORANGE, "EGC"), ("mrc", GREEN, "MRC")):
        observed = diversity["ber"][method][li]
        floor = 0.5 / diversity["bits_per_point"]
        ax.semilogy(diversity["snr_db"], np.maximum(observed, floor), "o-", color=color, label=label)
    ax.set(xlabel="Average branch SNR (dB)", ylabel="BER", title="Four-branch Rayleigh diversity", ylim=(1e-6, 1))
    ax.grid(alpha=0.25, which="both")
    ax.legend()
    save_figure(fig, output_dir, "task1_diversity_comparison")

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.8), constrained_layout=True)
    t_ms = 1e3 * np.arange(lab.PERIOD_SAMPLES) / lab.FS
    axes[0].plot(t_ms, np.abs(waveform.tx0_period), lw=0.8, color=BLUE, label="TX channel 0")
    axes[0].plot(t_ms, np.abs(waveform.tx1_period), lw=0.8, color=ORANGE, label="TX channel 1")
    axes[0].set(xlabel="Time in one period (ms)", ylabel="Magnitude", title="Lab 6 orthogonal sounding and SIMO schedule")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    segment = waveform.tx0_period[lab.VERIFY]
    nfft = 16_384
    spectrum0 = np.fft.fftshift(np.fft.fft(waveform.tx0_period[lab.VERIFY], nfft))
    spectrum1 = np.fft.fftshift(np.fft.fft(waveform.tx1_period[lab.VERIFY], nfft))
    frequency = np.fft.fftshift(np.fft.fftfreq(nfft, 1.0 / lab.FS)) / 1e3
    axes[1].plot(frequency, 20 * np.log10(np.maximum(np.abs(spectrum0), 1e-12)), color=BLUE, label="TX0 verify tone")
    axes[1].plot(frequency, 20 * np.log10(np.maximum(np.abs(spectrum1), 1e-12)), color=ORANGE, label="TX1 verify tone")
    axes[1].set(xlabel="Baseband frequency (kHz)", ylabel="Relative magnitude (dB)", title="Dual-chain verification tones")
    axes[1].set_xlim(-250, 250)
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    save_figure(fig, output_dir, "task2_waveform_schedule")

    return {
        "mimo": {
            "snr_db": mimo["snr_db"].tolist(),
            "mimo_capacity_bps_hz": mimo["mimo_capacity_bps_hz"].tolist(),
            "siso_capacity_bps_hz": mimo["siso_capacity_bps_hz"].tolist(),
            "channel_nmse": mimo["channel_nmse"].tolist(),
            "mean_singular_value_ratio": mimo["mean_singular_value_ratio"].tolist(),
            "realizations": mimo["realizations"],
        },
        "diversity": {
            "snr_db": diversity["snr_db"].tolist(),
            "branch_counts": diversity["branch_counts"].tolist(),
            "bits_per_point": diversity["bits_per_point"],
            "ber": {key: value.tolist() for key, value in diversity["ber"].items()},
            "errors": {key: value.tolist() for key, value in diversity["errors"].items()},
        },
        "waveform": lab.waveform_metadata(waveform),
    }


def _complex_matrix(value: np.ndarray) -> dict[str, list[list[float]]]:
    value = np.asarray(value)
    return {"real": value.real.tolist(), "imag": value.imag.tolist()}


def _aggregate_gain(result: dict[str, object], waveform: lab.Lab6Waveform) -> dict[str, object]:
    frames = result["analyzed_frames"]
    if not frames:
        return {
            "detected_frames_including_warmup": len(result["frames"]),
            "analyzed_frames": 0,
            "detection_rate": 0.0,
        }
    h = np.stack([frame["mimo_channel"] for frame in frames])
    capacities = np.array([frame["mimo_capacity_bps_hz"] for frame in frames])
    siso = np.stack([frame["siso_capacity_bps_hz"] for frame in frames])
    cfo = np.array([frame["cfo_hz"] for frame in frames])
    condition = np.array([frame["condition_number"] for frame in frames])
    combiner = {}
    for method in ("sc", "egc", "mrc"):
        bit_errors = int(sum(frame["combiners"][method]["bit_errors"] for frame in frames))
        total_bits = len(frames) * waveform.payload_bits.size
        combiner[method] = {
            "bit_errors": bit_errors,
            "ber": bit_errors / total_bits,
            "evm_rms": float(np.sqrt(np.mean([frame["combiners"][method]["evm_rms"] ** 2 for frame in frames]))),
        }
    branch_evm = [
        float(np.sqrt(np.mean([frame["branches"][branch]["evm_rms"] ** 2 for frame in frames])))
        for branch in range(2)
    ]
    return {
        "detected_frames_including_warmup": len(result["frames"]),
        "analyzed_frames": len(frames),
        "detection_rate": min(1.0, len(frames) / (lab.REPEAT_COUNT - lab.WARMUP_PERIODS)),
        "mimo_channel_mean": _complex_matrix(np.mean(h, axis=0)),
        "mimo_channel_magnitude_mean": np.mean(np.abs(h), axis=0).tolist(),
        "mimo_channel_reference": _complex_matrix(h[0]),
        "mimo_capacity_reference_bps_hz": float(capacities[0]),
        "mimo_capacity_mean_bps_hz": float(np.mean(capacities)),
        "mimo_capacity_std_bps_hz": float(np.std(capacities)),
        "siso_capacity_mean_bps_hz": np.mean(siso, axis=0).tolist(),
        "condition_number_mean": float(np.mean(condition)),
        "cfo_mean_hz": float(np.mean(cfo)),
        "cfo_std_hz": float(np.std(cfo)),
        "branch_evm_rms": branch_evm,
        "combining": combiner,
    }


def analyze_hardware(hardware_dir: Path, output_dir: Path) -> dict[str, object]:
    hardware_dir = Path(hardware_dir)
    output_dir = Path(output_dir)
    summary = json.loads((hardware_dir / "lab6_measurement_summary.json").read_text())
    gains = tuple(int(value) for value in summary["experiment"]["tx_gains_db"])
    waveform = lab.build_waveform()
    noise0 = lab.load_complex64(hardware_dir / "lab6_noise_ch0_c64.dat")
    noise1 = lab.load_complex64(hardware_dir / "lab6_noise_ch1_c64.dat")
    if noise0.size < 250_000 or noise1.size < 250_000:
        raise ValueError(
            "Each transmitter-off noise record must contain at least 250000 complex64 samples."
        )
    # The standard finite receiver records six seconds. Use the first 0.25 s
    # for the noise estimate, as specified in the student manual, while also
    # accepting a dedicated 0.25 s noise capture.
    noise0 = noise0[:250_000]
    noise1 = noise1[:250_000]
    gain_results: dict[int, dict[str, object]] = {}
    full_results: dict[int, dict[str, object]] = {}
    for gain in gains:
        rx0 = lab.load_complex64(hardware_dir / f"lab6_rx_tx{gain}_ch0_c64.dat", lab.CAPTURE_SAMPLES)
        rx1 = lab.load_complex64(hardware_dir / f"lab6_rx_tx{gain}_ch1_c64.dat", lab.CAPTURE_SAMPLES)
        result = lab.analyze_hardware_capture(rx0, rx1, noise0, noise1, waveform)
        aggregate = _aggregate_gain(result, waveform)
        aggregate["capture_peak"] = [float(np.max(np.abs(rx0))), float(np.max(np.abs(rx1)))]
        aggregate["clipped_sample_count"] = [int(np.count_nonzero(np.abs(rx0) >= 0.99)), int(np.count_nonzero(np.abs(rx1) >= 0.99))]
        gain_results[gain] = aggregate
        full_results[gain] = {"rx0": rx0, "rx1": rx1, "result": result}

    baseline_gain = gains[len(gains) // 2]
    baseline = full_results[baseline_gain]
    baseline_result = baseline["result"]
    first = baseline_result["analyzed_frames"][0]
    starts = baseline_result["starts"]

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.0), constrained_layout=True)
    step = 50
    sample_axis = np.arange(0, baseline["rx0"].size, step)
    axes[0].plot(sample_axis, np.abs(baseline["rx0"][::step]), lw=0.5, color=BLUE, label="RX channel 0")
    axes[0].plot(sample_axis, np.abs(baseline["rx1"][::step]), lw=0.5, color=ORANGE, alpha=0.75, label="RX channel 1")
    for start in starts:
        axes[0].axvline(start, color=GREEN, lw=0.6, alpha=0.5)
    axes[0].set(xlabel="Capture sample", ylabel="Magnitude", title=f"Uncropped dual-channel TX {baseline_gain} dB capture")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    metric = baseline_result["metric"]
    axes[1].plot(np.arange(metric.size) * 64, metric / np.max(metric), color=PURPLE, lw=0.6)
    axes[1].set(xlabel="Candidate period start (samples)", ylabel="Normalized envelope correlation", title="Known schedule acquisition metric")
    axes[1].grid(alpha=0.2)
    axes[2].plot(first["timing_search"], first["timing_correlation"] / np.max(first["timing_correlation"]), color=GREEN)
    axes[2].axvline(lab.SIMO_TRAINING.start + first["timing_shift_samples"], color=RED, ls="--", label="selected start")
    axes[2].set(xlabel="SIMO training start within period", ylabel="Normalized correlation", title="Fine timing from the known QPSK training field")
    axes[2].legend()
    axes[2].grid(alpha=0.2)
    save_figure(fig, output_dir, "task2_hardware_acquisition")

    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.8), constrained_layout=True)
    for index, gain in enumerate(gains):
        frame = full_results[gain]["result"]["analyzed_frames"][0]
        magnitude = np.abs(frame["mimo_channel"])
        image = axes[index].imshow(magnitude, cmap="viridis", vmin=0)
        for row in range(2):
            for col in range(2):
                axes[index].text(col, row, f"{magnitude[row,col]:.4f}", ha="center", va="center", color="white")
        axes[index].set(xticks=(0, 1), yticks=(0, 1), xticklabels=("TX0", "TX1"), yticklabels=("RX0", "RX1"), title=f"TX gain {gain} dB")
        fig.colorbar(image, ax=axes[index], fraction=0.046)
    save_figure(fig, output_dir, "task2_hardware_channel_matrices")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), constrained_layout=True)
    axes[0].plot(gains, [gain_results[g]["mimo_capacity_mean_bps_hz"] for g in gains], "o-", color=BLUE, lw=2, label="2x2 MIMO")
    for index, label in enumerate(("h11", "h12", "h21", "h22")):
        axes[0].plot(gains, [gain_results[g]["siso_capacity_mean_bps_hz"][index] for g in gains], "o--", lw=1, label=f"SISO {label}")
    axes[0].set(xlabel="TX gain (dB)", ylabel="Spectral efficiency (bit/s/Hz)", title="Measured capacity")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)
    axes[1].plot(gains, [gain_results[g]["condition_number_mean"] for g in gains], "o-", color=ORANGE)
    axes[1].set(xlabel="TX gain (dB)", ylabel="Mean condition number", title="Measured channel conditioning")
    axes[1].grid(alpha=0.25)
    save_figure(fig, output_dir, "task2_hardware_capacity")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    _plot_constellation(axes[0], first["branches"][0]["symbols"], "RX branch 0")
    _plot_constellation(axes[1], first["branches"][1]["symbols"], "RX branch 1")
    _plot_constellation(axes[2], first["combiners"]["mrc"]["symbols"], "MRC output")
    save_figure(fig, output_dir, "task2_hardware_diversity_constellations")

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0), constrained_layout=True)
    for method, color, label in (("sc", BLUE, "SC"), ("egc", ORANGE, "EGC"), ("mrc", GREEN, "MRC")):
        axes[0].semilogy(gains, [max(gain_results[g]["combining"][method]["ber"], 0.5 / (gain_results[g]["analyzed_frames"] * lab.SIMO_PAYLOAD_BITS)) for g in gains], "o-", color=color, label=label)
        axes[1].plot(gains, [gain_results[g]["combining"][method]["evm_rms"] for g in gains], "o-", color=color, label=label)
    axes[0].set(xlabel="TX gain (dB)", ylabel="BER", title="Measured diversity BER")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend()
    axes[1].set(xlabel="TX gain (dB)", ylabel="RMS EVM", title="Measured diversity quality")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    save_figure(fig, output_dir, "task2_hardware_diversity_metrics")

    return {
        "hardware_directory": str(hardware_dir),
        "noise_variance": [
            float(np.mean(np.abs(noise0 - np.mean(noise0)) ** 2)),
            float(np.mean(np.abs(noise1 - np.mean(noise1)) ** 2)),
        ],
        "gains": {f"tx{gain}": gain_results[gain] for gain in gains},
        "metadata": summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--hardware-dir", type=Path)
    parser.add_argument("--synthesize-hardware", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthesize_hardware:
        if args.hardware_dir is None:
            raise ValueError("--synthesize-hardware requires --hardware-dir")
        generate_synthetic_hardware(args.hardware_dir)
    results: dict[str, object] = {"simulation": generate_simulation(args.output_dir)}
    if args.hardware_dir is not None:
        results["hardware"] = analyze_hardware(args.hardware_dir, args.output_dir)
    (args.output_dir / "lab6_validation_results.json").write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
