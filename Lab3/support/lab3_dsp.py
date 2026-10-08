#!/usr/bin/env python3
"""DSP helpers for Lab 3"""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, special


def gray_encode(values: np.ndarray | int) -> np.ndarray:
    integers = np.asarray(values, dtype=np.int64)
    if np.any(integers < 0):
        raise ValueError("Gray-code inputs must be nonnegative")
    return integers ^ (integers >> 1)


def gray_decode(values: np.ndarray | int) -> np.ndarray:
    gray = np.asarray(values, dtype=np.int64)
    if np.any(gray < 0):
        raise ValueError("Gray-code inputs must be nonnegative")
    decoded = gray.copy()
    shifted = gray.copy()
    while np.any(shifted):
        shifted >>= 1
        decoded ^= shifted
    return decoded


def integers_to_bits(values: np.ndarray, width: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64).reshape(-1)
    if width < 1 or np.any(labels < 0) or np.any(labels >= 2**width):
        raise ValueError("labels do not fit the requested bit width")
    shifts = np.arange(width - 1, -1, -1, dtype=np.int64)
    return ((labels[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def bits_to_integers(bits: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if width < 1 or values.size % width or np.any(values > 1):
        raise ValueError("bits must be binary and divisible by width")
    weights = 1 << np.arange(width - 1, -1, -1, dtype=np.int64)
    return (values.reshape(-1, width) @ weights).astype(np.int64)


def psk_constellation(order: int) -> np.ndarray:
    """Return unit-energy PSK points indexed by cyclic Gray labels."""
    if order < 2 or order & (order - 1):
        raise ValueError("PSK order must be a power of two")
    labels = np.arange(order, dtype=np.int64)
    phase_indices = gray_decode(labels)
    return np.exp(1j * 2.0 * np.pi * phase_indices / order)


def square_qam_constellation(order: int) -> np.ndarray:
    """Return unit-average-energy square QAM indexed by Gray bit words."""
    side = math.isqrt(order)
    if order < 4 or side * side != order or side & (side - 1):
        raise ValueError("QAM order must be a square power of two")
    axis_bits = int(math.log2(side))
    labels = np.arange(order, dtype=np.int64)
    i_labels = labels >> axis_bits
    q_labels = labels & (side - 1)
    levels = 2.0 * np.arange(side) - (side - 1)
    points = levels[gray_decode(i_labels)] + 1j * levels[gray_decode(q_labels)]
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def minimum_distance_detect(
    samples: np.ndarray, constellation: np.ndarray, chunk_size: int = 4096
) -> np.ndarray:
    received = np.asarray(samples, dtype=np.complex128).reshape(-1)
    points = np.asarray(constellation, dtype=np.complex128).reshape(-1)
    decisions = np.empty(received.size, dtype=np.int64)
    for start in range(0, received.size, chunk_size):
        stop = min(start + chunk_size, received.size)
        distance = np.abs(received[start:stop, None] - points[None, :]) ** 2
        decisions[start:stop] = np.argmin(distance, axis=1)
    return decisions


def add_complex_awgn(
    samples: np.ndarray,
    esn0_db: float,
    rng: np.random.Generator,
    symbol_energy: float = 1.0,
) -> np.ndarray:
    gamma_s = 10.0 ** (float(esn0_db) / 10.0)
    n0 = float(symbol_energy) / gamma_s
    noise = np.sqrt(n0 / 2.0) * (
        rng.standard_normal(np.asarray(samples).shape)
        + 1j * rng.standard_normal(np.asarray(samples).shape)
    )
    return np.asarray(samples) + noise


def error_rates(
    detected: np.ndarray, reference: np.ndarray, order: int
) -> tuple[float, float]:
    detected_labels = np.asarray(detected, dtype=np.int64).reshape(-1)
    reference_labels = np.asarray(reference, dtype=np.int64).reshape(-1)
    width = int(math.log2(order))
    ser = float(np.mean(detected_labels != reference_labels))
    ber = float(
        np.mean(
            integers_to_bits(detected_labels, width)
            != integers_to_bits(reference_labels, width)
        )
    )
    return ser, ber


def exact_psk_ser(order: int, esn0_db: np.ndarray) -> np.ndarray:
    """Evaluate coherent M-PSK SER by its standard AWGN integral."""
    gamma = 10.0 ** (np.asarray(esn0_db, dtype=float) / 10.0)
    angle = np.pi / order
    upper = (order - 1.0) * np.pi / order
    values = []
    for value in gamma:
        coefficient = value * np.sin(angle) ** 2

        def integrand(theta: float) -> float:
            sine = np.sin(theta)
            if abs(sine) < 1e-14:
                return 0.0
            return float(np.exp(-coefficient / sine**2))

        integral, _ = integrate.quad(integrand, 0.0, upper, epsabs=1e-12)
        values.append(integral / np.pi)
    return np.asarray(values)


def square_qam_ser(order: int, esn0_db: np.ndarray) -> np.ndarray:
    gamma = 10.0 ** (np.asarray(esn0_db, dtype=float) / 10.0)
    q_value = 0.5 * special.erfc(np.sqrt(3.0 * gamma / (order - 1.0)) / np.sqrt(2.0))
    side = np.sqrt(order)
    return 1.0 - (1.0 - 2.0 * (1.0 - 1.0 / side) * q_value) ** 2


def rrc_taps(rolloff: float, samples_per_symbol: int, span_symbols: int) -> np.ndarray:
    beta = float(rolloff)
    sps = int(samples_per_symbol)
    span = int(span_symbols)
    if not 0.0 < beta <= 1.0:
        raise ValueError("rolloff must be in (0, 1]")
    if sps < 2 or span < 2 or (span * sps) % 2:
        raise ValueError("span times samples_per_symbol must be even")

    time = np.arange(-span * sps // 2, span * sps // 2 + 1, dtype=float) / sps
    taps = np.empty_like(time)
    at_zero = np.isclose(time, 0.0)
    at_singular = np.isclose(np.abs(time), 1.0 / (4.0 * beta))
    ordinary = ~(at_zero | at_singular)
    taps[at_zero] = 1.0 + beta * (4.0 / np.pi - 1.0)
    taps[at_singular] = beta / np.sqrt(2.0) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
        + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
    )
    tau = time[ordinary]
    taps[ordinary] = (
        np.sin(np.pi * tau * (1.0 - beta))
        + 4.0 * beta * tau * np.cos(np.pi * tau * (1.0 + beta))
    ) / (np.pi * tau * (1.0 - (4.0 * beta * tau) ** 2))
    return taps / np.sqrt(np.sum(taps**2))


def pulse_shape(
    symbols: np.ndarray, taps: np.ndarray, samples_per_symbol: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    sps = int(samples_per_symbol)
    upsampled = np.zeros((values.size - 1) * sps + 1, dtype=np.complex128)
    upsampled[::sps] = values
    return upsampled, np.convolve(upsampled, np.asarray(taps), mode="full")


def principal_alias(frequency_hz: float, sample_rate_sps: float) -> float:
    return float(
        ((frequency_hz + sample_rate_sps / 2.0) % sample_rate_sps)
        - sample_rate_sps / 2.0
    )
