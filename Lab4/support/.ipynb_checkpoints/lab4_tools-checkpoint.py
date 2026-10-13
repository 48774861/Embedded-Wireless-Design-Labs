
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lab4_dsp import (
    DATA_SUBCARRIERS,
    build_pn_burst,
    locate_and_estimate_pn,
    ofdm_phase_sweep,
    pn_sequence,
    timing_trial,
    ofdm_grid
)


BLUE = "#185b9d"
ORANGE = "#d06b20"
RED = "#aa2c3b"
GREEN = "#2f7d59"
PURPLE = "#6f4b9b"


def _save(figure: plt.Figure, output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return path


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(value, output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2) + "\n")
    return path


def run_timing_study(
    orders=(4, 16, 64),
    ebn0_values=(8, 12, 15, 18),
    timing_errors=(0, 1, 2, 3),
    *,
    initial_ebn0_db=15,
    initial_timing_error=1,
):
    single = {
        int(order): timing_trial(int(order), initial_ebn0_db, initial_timing_error)
        for order in orders
    }
    before = np.empty((len(orders), len(ebn0_values), len(timing_errors)))
    after = np.empty_like(before)
    for oi, order in enumerate(orders):
        for si, ebn0_db in enumerate(ebn0_values):
            for ei, error in enumerate(timing_errors):
                result = timing_trial(int(order), float(ebn0_db), int(error), seed=1400 + 10 * si)
                before[oi, si, ei] = result["before_ber"]
                after[oi, si, ei] = result["after_ber"]
    return {
        "orders": tuple(int(value) for value in orders),
        "ebn0_values": tuple(float(value) for value in ebn0_values),
        "timing_errors": tuple(int(value) for value in timing_errors),
        "single": single,
        "before": before,
        "after": after,
    }


def plot_timing_study(study, figure_dir: Path | str):
    figure_dir = Path(figure_dir)
    figure, axes = plt.subplots(3, 2, figsize=(8.5, 10), constrained_layout=True)
    for row, order in enumerate(study["orders"]):
        result = study["single"][order]
        for column, key in enumerate(("before", "after")):
            points = np.asarray(result[key])[:1800]
            axis = axes[row, column]
            axis.scatter(points.real, points.imag, s=3, alpha=0.25, color=BLUE)
            constellation = np.asarray(result["constellation"])
            axis.scatter(constellation.real, constellation.imag, s=42, marker="x", color=RED)
            phase = 0 if key == "before" else result["estimated_phase"]
            axis.set(
                title=f"{order}-PSK, phase {phase}, BER={result[f'{key}_ber']:.4g}",
                xlabel="In phase",
                ylabel="Quadrature",
            )
            axis.set_aspect("equal", adjustable="box")
            axis.grid(alpha=0.25)
    constellation_path = _save(figure, figure_dir / "task1_symbol_sync_constellations.png")

    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for oi, order in enumerate(study["orders"]):
        for si, ebn0_db in enumerate(study["ebn0_values"]):
            axes[oi].plot(
                study["timing_errors"], study["before"][oi, si], "o-", label=f"{ebn0_db:g} dB"
            )
        axes[oi].set(
            title=f"{order}-PSK before correction",
            xlabel="Timing error (samples)",
            ylabel="BER",
        )
        axes[oi].set_yscale("symlog", linthresh=1e-5)
        axes[oi].grid(alpha=0.25)
        axes[oi].legend(title=r"$E_b/N_0$")
    sweep_path = _save(figure, figure_dir / "task1_timing_ber_sweep.png")
    return constellation_path, sweep_path


def plot_ofdm_phase_study(result, figure_dir: Path | str):
    figure_dir = Path(figure_dir)
    rows = result["rows"]
    figure, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    for column, row in enumerate(rows):
        for axis, key, prefix in (
            (axes[0, column], "received_data", "Uncorrected"),
            (axes[1, column], "corrected_data", "Pilot corrected"),
        ):
            points = np.asarray(row[key])
            axis.scatter(points.real, points.imag, s=4, alpha=0.25, color=BLUE)
            axis.axhline(0, color="0.4", lw=0.7)
            axis.axvline(0, color="0.4", lw=0.7)
            axis.set(
                title=f"{prefix}, {row['phase_degrees']} degrees",
                xlabel="In phase",
                ylabel="Quadrature",
                xlim=(-1.6, 1.6),
                ylim=(-1.6, 1.6),
            )
            axis.set_aspect("equal", adjustable="box")
    constellation_path = _save(figure, figure_dir / "task2_ofdm_phase_constellations.png")

    phases = np.array([row["phase_degrees"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    axes[0].plot(phases, [row["ber"] for row in rows], "o-", color=RED, label="uncorrected")
    axes[0].plot(
        phases, [row["corrected_ber"] for row in rows], "s--", color=GREEN, label="pilot corrected"
    )
    axes[1].plot(phases, [row["fer"] for row in rows], "o-", color=RED, label="uncorrected")
    axes[1].plot(
        phases, [row["corrected_fer"] for row in rows], "s--", color=GREEN, label="pilot corrected"
    )
    axes[0].set(xlabel="Applied phase offset (degrees)", ylabel="BER", title="Bit errors")
    axes[1].set(xlabel="Applied phase offset (degrees)", ylabel="FER", title="Frame errors")
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(alpha=0.25)
        axis.legend()
    rates_path = _save(figure, figure_dir / "task2_ofdm_phase_error_rates.png")
    return constellation_path, rates_path


def generate_pn_waveform(output: Path | str):
    reference = build_pn_burst()
    waveform = np.asarray(reference["shifted"], dtype=np.complex64)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform.tofile(path)
    return reference, path


def make_synthetic_pn_capture(
    *,
    capture_samples=90_000,
    delay_samples=22_000,
    cfo_hz=-245.0,
    channel=0.0023 * np.exp(1j * 0.8),
    snr_db=18.0,
    seed=4403,
):
    reference = build_pn_burst(acquisition_lead_samples=0)
    waveform = np.asarray(reference["shifted"], dtype=np.complex128)
    capture = np.zeros(int(capture_samples), dtype=np.complex128)
    n = np.arange(waveform.size)
    impaired = channel * waveform * np.exp(
        1j * 2 * np.pi * cfo_hz * n / reference["sample_rate_sps"]
    )
    capture[delay_samples : delay_samples + waveform.size] += impaired
    active = impaired[
        reference["guard_samples"] : reference["guard_samples"] + reference["chips"].size
    ]
    active_power = np.mean(np.abs(active) ** 2)
    noise_power = active_power / 10 ** (snr_db / 10)
    rng = np.random.default_rng(seed)
    capture += np.sqrt(noise_power / 2) * (
        rng.standard_normal(capture.size) + 1j * rng.standard_normal(capture.size)
    )
    result = locate_and_estimate_pn(capture)
    truth = {
        "start_sample": delay_samples + reference["guard_samples"],
        "cfo_hz": float(cfo_hz),
        "channel": complex(channel),
        "noise_power": float(noise_power),
    }
    return capture, result, truth


def plot_synthetic_pn(capture, result, truth, figure_dir: Path | str):
    figure_dir = Path(figure_dir)
    figure, axes = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axes[0].plot(np.abs(capture), lw=0.5, color=BLUE)
    axes[0].axvline(truth["start_sample"], color=GREEN, ls="--", label="true PN start")
    axes[0].axvline(result["start_sample"], color=RED, ls=":", label="estimated PN start")
    axes[0].set(xlabel="Capture sample", ylabel="Magnitude", title="Complete synthetic capture")
    axes[0].legend()
    correlation = np.abs(np.asarray(result["correlation"]))
    axes[1].plot(result["correlation_lags"], correlation / np.max(correlation), color=ORANGE)
    axes[1].axvline(result["start_sample"], color=RED, ls=":")
    axes[1].set(xlabel="Candidate PN start", ylabel="Normalized magnitude", title="Cross-correlation")
    axes[2].plot(result["product_phase"], lw=0.8, color=BLUE, label="unwrapped phase")
    axes[2].plot(result["phase_fit"], color=RED, ls="--", label=f"fit, CFO={result['cfo_hz']:.2f} Hz")
    axes[2].set(xlabel="PN chip", ylabel="Phase (rad)", title="Residual CFO fit")
    axes[2].legend()
    sync_path = _save(figure, figure_dir / "task3_synthetic_channel_estimation.png")

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    received = np.asarray(result["received"])
    equalized = np.asarray(result["equalized"])
    axes[0].scatter(received.real, received.imag, s=4, alpha=0.3, color=BLUE)
    axes[0].set(title="Received PN chips", xlabel="In phase", ylabel="Quadrature")
    axes[1].scatter(equalized.real, equalized.imag, s=4, alpha=0.3, color=GREEN)
    axes[1].scatter([-1, 1], [0, 0], marker="x", color=RED, s=70)
    axes[1].set(
        title=f"After CFO and scalar equalization, EVM={100*result['evm']:.2f}%",
        xlabel="In phase",
        ylabel="Quadrature",
    )
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    equalized_path = _save(figure, figure_dir / "task3_synthetic_equalization.png")
    return sync_path, equalized_path


def plot_pn_sequence(figure_dir: Path | str):
    figure_dir = Path(figure_dir)
    chips = pn_sequence(10)
    circular = np.fft.ifft(np.abs(np.fft.fft(chips)) ** 2).real / chips.size
    circular = np.fft.fftshift(circular)
    lags = np.arange(-(chips.size // 2), chips.size // 2 + 1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    axes[0].step(np.arange(64), chips.real[:64], where="post", color=BLUE)
    axes[0].set(xlabel="Chip index", ylabel="Amplitude", title="First 64 degree-10 PN chips", ylim=(-1.25, 1.25))
    axes[1].plot(lags, circular, color=ORANGE)
    axes[1].set(
        xlabel="Circular lag (chips)",
        ylabel="Normalized correlation",
        title="PN circular autocorrelation",
        xlim=(-80, 80),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    return _save(figure, figure_dir / "task3_pn_sequence.png")


def analyze_pn_capture(
    capture_path: Path | str,
    *,
    noise_path: Path | str | None = None,
    squelched_path: Path | str | None = None,
):
    capture_path = Path(capture_path)
    capture = np.memmap(capture_path, dtype=np.complex64, mode="r")
    result = locate_and_estimate_pn(capture)
    start = int(result["start_sample"])
    active = np.asarray(capture[start : start + 1023], dtype=np.complex128)
    active_power = float(np.mean(np.abs(active) ** 2))
    noise_power = np.nan
    if noise_path is not None:
        noise = np.memmap(Path(noise_path), dtype=np.complex64, mode="r")
        noise_power = float(np.mean(np.abs(noise) ** 2))
    signal_power = max(active_power - noise_power, np.finfo(float).tiny) if np.isfinite(noise_power) else np.nan
    snr_db = 10 * np.log10(signal_power / noise_power) if np.isfinite(noise_power) else np.nan
    correlation = np.abs(np.asarray(result["correlation"]))
    peak_to_median_db = float(20 * np.log10(np.max(correlation) / np.median(correlation)))
    squelched_nonzero_fraction = np.nan
    if squelched_path is not None and Path(squelched_path).is_file():
        squelched = np.memmap(Path(squelched_path), dtype=np.complex64, mode="r")
        squelched_nonzero_fraction = float(np.count_nonzero(squelched) / squelched.size)
    channel = complex(result["channel"])
    return {
        "capture_path": capture_path,
        "squelched_path": Path(squelched_path) if squelched_path is not None else None,
        "file_bytes": capture_path.stat().st_size,
        "sample_count": int(capture.size),
        "start_sample": start,
        "correlation_peak": float(result["correlation_peak"]),
        "peak_to_median_db": peak_to_median_db,
        "cfo_hz": float(result["cfo_hz"]),
        "channel": channel,
        "channel_magnitude": float(abs(channel)),
        "channel_phase_degrees": float(np.rad2deg(np.angle(channel))),
        "evm": float(result["evm"]),
        "chip_error_rate": float(result["chip_error_rate"]),
        "noise_power": noise_power,
        "noise_power_db": float(10 * np.log10(noise_power)) if np.isfinite(noise_power) else np.nan,
        "active_power": active_power,
        "active_power_db": float(10 * np.log10(active_power)),
        "snr_db": float(snr_db),
        "maximum_magnitude": float(np.max(np.abs(capture))),
        "squelched_nonzero_fraction": squelched_nonzero_fraction,
        "correlation": result["correlation"],
        "correlation_lags": result["correlation_lags"],
        "received": result["received"],
        "corrected": result["corrected"],
        "equalized": result["equalized"],
        "product_phase": result["product_phase"],
        "phase_fit": result["phase_fit"],
    }


def plot_real_pn_measurement(record, figure_dir: Path | str, *, prefix="task3_real"):
    figure_dir = Path(figure_dir)
    capture = np.memmap(record["capture_path"], dtype=np.complex64, mode="r")
    stride = max(1, capture.size // 30_000)
    full_index = np.arange(0, capture.size, stride)
    start = record["start_sample"]
    first = max(0, start - 6000)
    last = min(capture.size, start + 7000)
    local_index = np.arange(first, last)

    figure, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes[0, 0].plot(full_index, np.abs(capture[::stride]), lw=0.5, color=BLUE)
    axes[0, 0].axvline(start, color=RED, ls=":")
    axes[0, 0].set(xlabel="Capture sample", ylabel="Magnitude", title="Complete raw capture")
    axes[0, 1].plot(local_index, np.abs(capture[first:last]), lw=0.65, color=BLUE, label="raw")
    if record["squelched_path"] is not None and Path(record["squelched_path"]).is_file():
        squelched = np.memmap(record["squelched_path"], dtype=np.complex64, mode="r")
        axes[0, 1].plot(local_index, np.abs(squelched[first:last]), lw=0.65, color=ORANGE, label="squelched")
        axes[0, 1].legend()
    axes[0, 1].axvline(start, color=RED, ls=":")
    axes[0, 1].set(xlabel="Capture sample", ylabel="Magnitude", title="PN neighborhood")
    correlation = np.abs(np.asarray(record["correlation"]))
    axes[1, 0].plot(record["correlation_lags"], correlation / np.max(correlation), color=ORANGE)
    axes[1, 0].axvline(start, color=RED, ls=":")
    axes[1, 0].set(xlabel="Candidate PN start", ylabel="Normalized magnitude", title="Cross-correlation")
    axes[1, 1].plot(record["product_phase"], lw=0.8, color=BLUE, label="unwrapped phase")
    axes[1, 1].plot(record["phase_fit"], color=RED, ls="--", label=f"CFO={record['cfo_hz']:.2f} Hz")
    axes[1, 1].set(xlabel="PN chip", ylabel="Phase (rad)", title="Residual CFO fit")
    axes[1, 1].legend()
    measurement_path = _save(figure, figure_dir / f"{prefix}_acquisition.png")

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    received = np.asarray(record["received"])
    equalized = np.asarray(record["equalized"])
    axes[0].scatter(received.real, received.imag, s=4, alpha=0.3, color=BLUE)
    axes[0].set(title="Received PN chips", xlabel="In phase", ylabel="Quadrature")
    axes[1].scatter(equalized.real, equalized.imag, s=4, alpha=0.3, color=GREEN)
    axes[1].scatter([-1, 1], [0, 0], marker="x", color=RED, s=70)
    axes[1].set(
        title=f"Equalized, EVM={100*record['evm']:.2f}%, CER={record['chip_error_rate']:.4f}",
        xlabel="In phase",
        ylabel="Quadrature",
    )
    for axis in axes:
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.25)
    equalized_path = _save(figure, figure_dir / f"{prefix}_equalization.png")
    return measurement_path, equalized_path


def analyze_gain_sweep(capture_dir: Path | str, gains=(25, 32, 40), runs=(1, 2, 3, 4)):
    capture_dir = Path(capture_dir)
    noise_path = capture_dir / "task3_noise_c64.dat"
    records = []
    for gain in gains:
        for run in runs:
            stem = f"task3_gain{gain}_run{run}"
            record = analyze_pn_capture(
                capture_dir / f"{stem}_raw_c64.dat",
                noise_path=noise_path,
                squelched_path=capture_dir / f"{stem}_squelched_c64.dat",
            )
            record["tx_gain_db"] = int(gain)
            record["run"] = int(run)
            records.append(record)
    summaries = []
    for gain in gains:
        group = [record for record in records if record["tx_gain_db"] == gain]
        channels = np.asarray([record["channel"] for record in group])
        mean_channel = complex(np.mean(channels))
        mean_magnitude = float(np.mean(np.abs(channels)))
        summaries.append(
            {
                "tx_gain_db": int(gain),
                "channels": channels,
                "complex_mean": mean_channel,
                "magnitude_of_mean": float(abs(mean_channel)),
                "mean_magnitude": mean_magnitude,
                "mean_magnitude_db": float(20 * np.log10(mean_magnitude)),
                "mean_cfo_hz": float(np.mean([record["cfo_hz"] for record in group])),
                "mean_evm": float(np.mean([record["evm"] for record in group])),
                "mean_chip_error_rate": float(np.mean([record["chip_error_rate"] for record in group])),
            }
        )
    x = np.asarray([summary["tx_gain_db"] for summary in summaries], dtype=float)
    y = np.asarray([summary["mean_magnitude_db"] for summary in summaries], dtype=float)
    coefficient = np.polyfit(x, y, 1)
    prediction = np.polyval(coefficient, x)
    residuals = y - prediction
    ss_total = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - float(np.sum(residuals**2)) / ss_total if ss_total > 0 else np.nan
    return {
        "records": records,
        "summaries": summaries,
        "regression_coefficient": coefficient,
        "regression_prediction_db": prediction,
        "regression_residuals_db": residuals,
        "regression_r_squared": r_squared,
    }


def plot_gain_sweep(result, figure_dir: Path | str):
    figure_dir = Path(figure_dir)
    records = result["records"]
    summaries = result["summaries"]
    gains = np.asarray([summary["tx_gain_db"] for summary in summaries])
    means_db = np.asarray([summary["mean_magnitude_db"] for summary in summaries])

    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), constrained_layout=True)
    for summary, color in zip(summaries, (BLUE, ORANGE, GREEN)):
        channels = np.asarray(summary["channels"])
        axes[0, 0].plot(np.arange(1, channels.size + 1), np.abs(channels), "o-", color=color, label=f"{summary['tx_gain_db']} dB")
        axes[0, 1].scatter(channels.real, channels.imag, color=color, label=f"{summary['tx_gain_db']} dB")
    axes[0, 0].set(xlabel="Realization", ylabel=r"$|\hat h_i|$", title="Four estimates per gain")
    axes[0, 0].legend(title="TX gain")
    axes[0, 1].set(xlabel=r"Re$\{\hat h\}$", ylabel=r"Im$\{\hat h\}$", title="Complex estimates")
    axes[0, 1].legend(title="TX gain")
    axes[0, 1].axhline(0, color="0.5", lw=0.6)
    axes[0, 1].axvline(0, color="0.5", lw=0.6)
    coefficient = np.asarray(result["regression_coefficient"])
    dense = np.linspace(gains.min(), gains.max(), 200)
    axes[1, 0].plot(gains, means_db, "o", color=RED, label="measured means")
    axes[1, 0].plot(dense, np.polyval(coefficient, dense), "--", color="0.3", label="linear fit")
    axes[1, 0].set(
        xlabel="TX gain setting (dB)",
        ylabel=r"$20\log_{10}(\overline{|\hat h|})$",
        title=f"Gain regression, $R^2$={result['regression_r_squared']:.4f}",
    )
    axes[1, 0].legend()
    axes[1, 1].plot(gains, 100 * np.asarray([summary["mean_evm"] for summary in summaries]), "o-", color=PURPLE, label="EVM")
    axes[1, 1].plot(gains, 100 * np.asarray([summary["mean_chip_error_rate"] for summary in summaries]), "s--", color=ORANGE, label="chip errors")
    axes[1, 1].set(xlabel="TX gain setting (dB)", ylabel="Percent", title="Receiver quality")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    overview_path = _save(figure, figure_dir / "task4_real_gain_sweep.png")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    for gain, color in zip(gains, (BLUE, ORANGE, GREEN)):
        group = [record for record in records if record["tx_gain_db"] == gain]
        axes[0].plot([record["run"] for record in group], [record["cfo_hz"] for record in group], "o-", color=color, label=f"{gain} dB")
        axes[1].plot([record["run"] for record in group], [record["peak_to_median_db"] for record in group], "o-", color=color, label=f"{gain} dB")
    axes[0].set(xlabel="Realization", ylabel="Estimated CFO (Hz)", title="Residual oscillator offset")
    axes[1].set(xlabel="Realization", ylabel="Peak-to-median correlation (dB)", title="Acquisition evidence")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(title="TX gain")
    diagnostics_path = _save(figure, figure_dir / "task4_real_diagnostics.png")
    return overview_path, diagnostics_path


def compact_record(record):
    omitted = {"correlation", "correlation_lags", "received", "corrected", "equalized", "product_phase", "phase_fit"}
    return {key: value for key, value in record.items() if key not in omitted}


def ofdm_reference(**kwargs):
    return ofdm_phase_sweep(**kwargs)

def get_ofdm_grid(**kwargs):
    return ofdm_grid(**kwargs)

def task2_data_bits_per_frame():
    return int(2 * DATA_SUBCARRIERS.size)
