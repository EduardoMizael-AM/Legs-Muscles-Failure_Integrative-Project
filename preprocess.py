"""Input validation, gap handling, filtering and time-aware segmentation for the raw EMG CSV files.

Changes versus the original version:
* Small timestamp gaps (default <= 30 ms, i.e. a few lost samples) are filled by
  linear interpolation on a uniform grid so they no longer split a recording into
  many segments. Larger gaps remain real discontinuities. Recordings without small
  gaps are returned untouched. Interpolated samples are flagged in `filled_mask`.
* `bandpass_notch` provides a zero-phase band-pass plus mains notch filter that is
  applied per contiguous segment (never across a real discontinuity).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, iirnotch, sosfiltfilt, tf2sos


TIME_COLUMN = "Relative_Time_s"
SIGNAL_COLUMNS = ("Voltage_mV", "ADC_Value")
DEFAULT_MAX_FILL_GAP_S = 0.030


@dataclass(frozen=True)
class Recording:
    """Validated signal and sampling metadata for one source CSV."""

    time_s: np.ndarray
    signal: np.ndarray
    sampling_rate_hz: float
    median_interval_s: float
    gap_count: int                       # discontinuities that remain after filling
    filled_mask: np.ndarray | None = None  # True where a sample was interpolated
    filled_sample_count: int = 0


def fill_small_gaps(
    time_s: np.ndarray, signal: np.ndarray, interval_s: float, max_gap_s: float = DEFAULT_MAX_FILL_GAP_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Insert the missing samples of gaps no longer than `max_gap_s`.

    Original samples keep their exact values; only the lost samples are added, by
    linear interpolation between the neighbouring samples, and each block between
    real discontinuities is re-stamped on a uniform time grid (sub-millisecond
    shift). Returns (time, signal, filled_mask). Recordings with no small gap are
    returned unchanged so clean files stay bit-identical to the original pipeline.
    """
    dt = np.diff(time_s)
    small = (dt > interval_s * 1.01) & (dt <= max_gap_s)
    if not small.any():
        return time_s, signal, np.zeros(len(time_s), dtype=bool)
    cuts = np.flatnonzero(dt > max_gap_s) + 1          # real discontinuities stay
    t_parts, x_parts, m_parts = [], [], []
    for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(time_s)]):
        block_t, block_x = time_s[a:b], signal[a:b]
        missing = np.clip(np.round(np.diff(block_t) / interval_s).astype(int) - 1, 0, None)
        positions = np.arange(len(block_x)) + np.concatenate(([0], np.cumsum(missing)))
        total = len(block_x) + int(missing.sum())
        filled = np.ones(total, dtype=bool)
        filled[positions] = False
        t_parts.append(block_t[0] + np.arange(total) * interval_s)
        x_parts.append(np.interp(np.arange(total), positions, block_x))
        m_parts.append(filled)
    return np.concatenate(t_parts), np.concatenate(x_parts), np.concatenate(m_parts)


def load_recording(
    path: Path, signal_column: str, max_fill_gap_s: float | None = DEFAULT_MAX_FILL_GAP_S,
) -> Recording:
    """Load a source CSV, estimate its sample rate and (optionally) fill small gaps.

    Gaps larger than `max_fill_gap_s` are retained for segmentation; they are never
    interpolated or silently treated as contiguous samples. Pass `None` or 0 to
    reproduce the original behavior (no filling).
    """
    frame = pd.read_csv(path)
    required = {TIME_COLUMN, signal_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing required column(s): {', '.join(sorted(missing))}")

    time_s = pd.to_numeric(frame[TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    signal = pd.to_numeric(frame[signal_column], errors="coerce").to_numpy(dtype=float)
    if len(time_s) < 2:
        raise ValueError(f"{path}: fewer than two samples")
    if not np.isfinite(time_s).all() or not np.isfinite(signal).all():
        raise ValueError(f"{path}: timestamps or signal contain missing/non-finite values")

    # CSV timestamps are decimal seconds; rounding only removes binary-float
    # representation noise, while real 1 ms+ discontinuities remain visible.
    intervals = np.round(np.diff(time_s), decimals=9)
    if np.any(intervals <= 0):
        raise ValueError(f"{path}: timestamps are duplicated or not strictly increasing")
    median_interval_s = float(np.median(intervals))
    if median_interval_s <= 0:
        raise ValueError(f"{path}: could not estimate a positive sample interval")

    original_length = len(time_s)
    filled_mask = np.zeros(original_length, dtype=bool)
    if max_fill_gap_s:
        time_s, signal, filled_mask = fill_small_gaps(time_s, signal, median_interval_s, max_fill_gap_s)
        intervals = np.round(np.diff(time_s), decimals=9)

    # Large gaps and atypical intervals are excluded from windows downstream.
    tolerance_s = max(median_interval_s * 0.01, 1e-9)
    discontinuity = np.abs(intervals - median_interval_s) > tolerance_s
    return Recording(
        time_s=time_s,
        signal=signal,
        sampling_rate_hz=1.0 / median_interval_s,
        median_interval_s=median_interval_s,
        gap_count=int(discontinuity.sum()),
        filled_mask=filled_mask,
        filled_sample_count=int(filled_mask.sum()),
    )


def contiguous_segments(recording: Recording) -> list[tuple[int, int]]:
    """Return [start, stop) sample ranges that contain no timestamp gaps."""
    intervals = np.diff(recording.time_s)
    tolerance_s = max(recording.median_interval_s * 0.01, 1e-9)
    breaks = np.flatnonzero(np.abs(intervals - recording.median_interval_s) > tolerance_s) + 1
    edges = np.concatenate(([0], breaks, [len(recording.time_s)]))
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def bandpass_notch(
    signal: np.ndarray,
    sampling_rate_hz: float,
    segments: list[tuple[int, int]],
    band_hz: tuple[float, float] = (20.0, 450.0),
    notch_hz: float | None = 60.0,
    notch_harmonics: int = 1,
    notch_q: float = 30.0,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth band-pass plus mains notch, applied per contiguous segment.

    Segments too short to filter safely are returned unfiltered; they are also too
    short to produce analysis windows, so they never reach the feature table.
    """
    nyquist = sampling_rate_hz / 2.0
    high = min(band_hz[1], 0.9 * nyquist)
    chain = [butter(order, [band_hz[0], high], btype="bandpass", fs=sampling_rate_hz, output="sos")]
    if notch_hz:
        for k in range(1, notch_harmonics + 1):
            freq = notch_hz * k
            if freq < 0.95 * nyquist:
                b, a = iirnotch(freq, notch_q, fs=sampling_rate_hz)
                chain.append(tf2sos(b, a))
    sos = np.vstack(chain)
    filtered = np.array(signal, dtype=float, copy=True)
    for start, stop in segments:
        if stop - start > 200:
            filtered[start:stop] = sosfiltfilt(sos, filtered[start:stop])
    return filtered
