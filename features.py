"""Create a traceable, unlabeled window dataset from the raw EMG recordings.

This script deliberately does not assign fatigue classes. It keeps the original
(legacy) columns so existing scripts keep working, and adds an improved set of
columns computed after gap filling and band-pass + mains-notch filtering:

* legacy: rms, mav, variance, waveform_length, mdf_hz_raw_unfiltered, mnf_hz_raw_unfiltered
          (raw signal, 4 Hz FFT bins; identical to the original pipeline when run
          with --max-fill-gap-ms 0)
* new:    rms_filtered, mav_filtered, variance_filtered, waveform_length_filtered,
          mdf_hz_filtered, mnf_hz_filtered (zero-padded FFT, ~1 Hz bins, band-limited),
          interpolated_fraction (share of the window that was interpolated)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import bandpass_notch, contiguous_segments, load_recording


FILENAME_PATTERN = re.compile(r"^(R\d+(?:_\d+)?)_(dr|iz)_", re.IGNORECASE)


def time_features(values: np.ndarray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(values**2))),
        "mav": float(np.mean(np.abs(values))),
        "variance": float(np.var(values)),
        "waveform_length": float(np.sum(np.abs(np.diff(values)))),
    }


def spectral_summary(
    values: np.ndarray, sampling_rate_hz: float, nfft: int | None = None,
    band_hz: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Hann-windowed periodogram -> (MDF, MNF). nfft > len(values) zero-pads the FFT."""
    centered = values - np.mean(values)
    tapered = centered * np.hanning(len(centered))
    n = nfft if nfft and nfft > len(tapered) else len(tapered)
    spectrum = np.abs(np.fft.rfft(tapered, n=n)) ** 2
    frequencies = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)
    keep = frequencies > 0                      # omit DC
    if band_hz is not None:
        keep &= (frequencies >= band_hz[0]) & (frequencies <= band_hz[1])
    spectrum, frequencies = spectrum[keep], frequencies[keep]
    total_power = float(np.sum(spectrum))
    if total_power <= 0 or not np.isfinite(total_power):
        return float("nan"), float("nan")
    cumulative = np.cumsum(spectrum)
    mdf = float(frequencies[min(int(np.searchsorted(cumulative, total_power / 2)), len(frequencies) - 1)])
    mnf = float(np.sum(frequencies * spectrum) / total_power)
    return mdf, mnf


def window_features(signal: np.ndarray, sampling_rate_hz: float) -> dict[str, float]:
    """Legacy features: simple time features and unfiltered full-band spectral summaries."""
    values = np.asarray(signal, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or len(values) < 2:
        raise ValueError("A window must be a finite one-dimensional signal with at least two samples")
    mdf, mnf = spectral_summary(values, sampling_rate_hz)
    return {**time_features(values), "mdf_hz_raw_unfiltered": mdf, "mnf_hz_raw_unfiltered": mnf}


def filtered_window_features(
    signal: np.ndarray, sampling_rate_hz: float, nfft: int, band_hz: tuple[float, float],
) -> dict[str, float]:
    """Features from the band-pass/notch filtered signal with a zero-padded, band-limited spectrum."""
    values = np.asarray(signal, dtype=float)
    mdf, mnf = spectral_summary(values, sampling_rate_hz, nfft=nfft, band_hz=band_hz)
    features = {f"{name}_filtered": value for name, value in time_features(values).items()}
    features["mdf_hz_filtered"], features["mnf_hz_filtered"] = mdf, mnf
    return features


def build_dataset(
    data_dir: Path,
    output_path: Path,
    signal_column: str = "Voltage_mV",
    window_duration_s: float = 0.250,
    step_duration_s: float = 0.125,
    overwrite: bool = False,
    max_fill_gap_s: float = 0.030,
    apply_filter: bool = True,
    band_hz: tuple[float, float] = (20.0, 450.0),
    notch_hz: float | None = 60.0,
    nfft: int = 1024,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    if window_duration_s <= 0 or step_duration_s <= 0:
        raise ValueError("Window and step durations must be positive")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Raw-data directory does not exist: {data_dir}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path} (pass --overwrite to replace)")

    files = sorted(path for folder in data_dir.iterdir() if folder.is_dir() for path in folder.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No participant CSV files found under {data_dir}")
    subjects = {name: f"SUBJ{i:02d}" for i, name in enumerate(sorted({p.parent.name for p in files}), start=1)}

    rows: list[dict[str, object]] = []
    fs_values: list[float] = []
    total_gaps = 0
    filled_samples = 0
    skipped_short_segments = 0
    for path in files:
        match = FILENAME_PATTERN.match(path.name)
        if not match:
            raise ValueError(f"Filename does not encode repetition and side: {path.name}")
        repetition, side = match.groups()
        recording = load_recording(path, signal_column, max_fill_gap_s)
        fs = recording.sampling_rate_hz
        fs_values.append(fs)
        total_gaps += recording.gap_count
        filled_samples += recording.filled_sample_count
        window_samples = int(round(window_duration_s * fs))
        step_samples = max(1, int(round(step_duration_s * fs)))
        if window_samples < 2:
            raise ValueError(f"Estimated sampling rate too low for requested window in {path}")
        segments = contiguous_segments(recording)
        filtered_signal = (
            bandpass_notch(recording.signal, fs, segments, band_hz=band_hz, notch_hz=notch_hz)
            if apply_filter else None
        )

        window_index = 0
        for segment_index, (seg_start, seg_stop) in enumerate(segments):
            segment_length = seg_stop - seg_start
            if segment_length < window_samples:
                skipped_short_segments += 1
                continue
            for local_start in range(0, segment_length - window_samples + 1, step_samples):
                start = seg_start + local_start
                stop = start + window_samples
                feat = window_features(recording.signal[start:stop], fs)
                if filtered_signal is not None:
                    feat.update(filtered_window_features(filtered_signal[start:stop], fs, nfft, band_hz))
                row: dict[str, object] = {
                    "subject_id": subjects[path.parent.name],
                    "record_id": path.stem,
                    "source_file": path.relative_to(data_dir.parent).as_posix(),
                    "repetition_id": repetition,
                    "side": "right" if side.lower() == "dr" else "left",
                    "signal_column": signal_column,
                    "segment_id": segment_index,
                    "window_index_in_record": window_index,
                    "start_sample_index": start,
                    "end_sample_index_exclusive": stop,
                    "start_time_s": float(recording.time_s[start]),
                    "end_time_s": float(recording.time_s[stop - 1]),
                    "sampling_rate_hz_estimated": fs,
                    "interpolated_fraction": float(np.mean(recording.filled_mask[start:stop])),
                    **feat,
                    "label_phase": "",
                }
                rows.append(row)
                window_index += 1

    if not rows:
        raise ValueError("No windows were produced; inspect signal lengths and timestamp continuity")
    frame = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    summary: dict[str, float | int] = {
        "source_files": len(files),
        "participants": frame["subject_id"].nunique(),
        "windows": len(frame),
        "timestamp_discontinuities": total_gaps,
        "interpolated_samples": filled_samples,
        "segments_shorter_than_window": skipped_short_segments,
        "estimated_fs_min_hz": min(fs_values),
        "estimated_fs_max_hz": max(fs_values),
    }
    return frame, summary


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=repo_root / "data", help="Directory containing participant folders")
    parser.add_argument(
        "--output", type=Path,
        default=repo_root / "data" / "features_dataset_v2_unlabeled.csv",
        help="New output path; existing files are protected unless --overwrite is passed",
    )
    parser.add_argument("--signal-column", choices=("Voltage_mV", "ADC_Value"), default="Voltage_mV")
    parser.add_argument("--window-duration", type=float, default=0.250, help="Window duration in seconds (default: 0.250)")
    parser.add_argument("--step-duration", type=float, default=0.125, help="Window step in seconds (default: 0.125)")
    parser.add_argument("--max-fill-gap-ms", type=float, default=30.0,
                        help="Interpolate timestamp gaps up to this length (ms); 0 disables filling")
    parser.add_argument("--no-filter", action="store_true", help="Skip the band-pass/notch columns")
    parser.add_argument("--band-low", type=float, default=20.0, help="Band-pass low cutoff (Hz)")
    parser.add_argument("--band-high", type=float, default=450.0, help="Band-pass high cutoff (Hz)")
    parser.add_argument("--notch-hz", type=float, default=60.0, help="Mains notch frequency (Hz); 0 disables")
    parser.add_argument("--nfft", type=int, default=1024, help="FFT length for the filtered spectral features (zero-padding)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing the requested output file")
    args = parser.parse_args()

    _, summary = build_dataset(
        args.data_dir, args.output, args.signal_column,
        args.window_duration, args.step_duration, args.overwrite,
        max_fill_gap_s=args.max_fill_gap_ms / 1000.0,
        apply_filter=not args.no_filter,
        band_hz=(args.band_low, args.band_high),
        notch_hz=args.notch_hz or None,
        nfft=args.nfft,
    )
    print(f"Dataset saved: {args.output}")
    print(f"Signal column: {args.signal_column}")
    print(f"Windows: {summary['windows']} across {summary['participants']} participants and {summary['source_files']} files")
    print(
        f"Remaining timestamp discontinuities: {summary['timestamp_discontinuities']}; "
        f"interpolated samples: {summary['interpolated_samples']}; estimated fs range: "
        f"{summary['estimated_fs_min_hz']:.3f}-{summary['estimated_fs_max_hz']:.3f} Hz"
    )
    print("Labels left blank. Legacy raw columns kept; *_filtered columns hold the band-pass/notch features.")


if __name__ == "__main__":
    main()
