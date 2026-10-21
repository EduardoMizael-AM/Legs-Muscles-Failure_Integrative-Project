"""Create exploratory signal, amplitude, and spectral summaries and figures."""

from __future__ import annotations

import os
from pathlib import Path

# Keep Matplotlib's cache inside the project results folder in restricted runtimes.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "results" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FEATURES_PATH = DATA / "features_dataset_corrected_unlabeled.csv"
METRICS = ("mdf_hz_filtered", "mnf_hz_filtered")


def raw_quality_table() -> tuple[pd.DataFrame, dict[str, float | int]]:
    records = []
    rates = []
    all_zero = 0
    all_count = 0
    global_min = float("inf")
    global_max = float("-inf")
    global_peak: dict[str, object] = {}
    equal_adjacent_total = 0
    adc_global_min = float("inf")
    adc_global_max = float("-inf")
    constant_250_total = 0
    step_events: list[dict[str, object]] = []
    conversion_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for path in sorted(p for folder in DATA.iterdir() if folder.is_dir() for p in folder.glob("*.csv")):
        source = pd.read_csv(path)
        time = pd.to_numeric(source["Relative_Time_s"], errors="raise").to_numpy(dtype=float)
        signal = pd.to_numeric(source["Voltage_mV"], errors="raise").to_numpy(dtype=float)
        adc = pd.to_numeric(source["ADC_Value"], errors="raise").to_numpy(dtype=float)
        intervals = np.round(np.diff(time), 9)
        median_dt = float(np.median(intervals))
        gap_count = int(np.count_nonzero(np.abs(intervals - median_dt) > max(median_dt * 0.01, 1e-9)))
        rate = 1.0 / median_dt
        rates.append(rate)
        all_count += len(signal)
        adc_global_min = min(adc_global_min, float(np.min(adc)))
        adc_global_max = max(adc_global_max, float(np.max(adc)))
        all_zero += int(np.count_nonzero(signal == 0))
        equal = np.diff(signal) == 0
        equal_adjacent_total += int(equal.sum())
        # Longest exact repeated-value run, including the first sample.
        run_starts = np.r_[0, np.flatnonzero(np.diff(signal) != 0) + 1]
        run_ends = np.r_[run_starts[1:], len(signal)]
        longest_run_index = int(np.argmax(run_ends - run_starts))
        longest_run = int(run_ends[longest_run_index] - run_starts[longest_run_index])
        longest_run_value = float(signal[run_starts[longest_run_index]])
        # Number of fully constant 250-sample windows, using a rolling sum of changes.
        changes = (np.diff(signal) != 0).astype(np.int64)
        if len(changes) >= 249:
            change_prefix = np.r_[0, np.cumsum(changes)]
            constant_250 = int(np.count_nonzero(change_prefix[249:] - change_prefix[:-249] == 0))
        else:
            constant_250 = 0
        constant_250_total += constant_250
        abs_steps = np.abs(np.diff(signal))
        if len(abs_steps):
            largest_step_index = int(np.argmax(abs_steps))
            step_events.append({
                "source_file": path.relative_to(ROOT).as_posix(),
                "time_s": float(time[largest_step_index + 1]),
                "step_recorded_units": float(signal[largest_step_index + 1] - signal[largest_step_index]),
                "abs_step_recorded_units": float(abs_steps[largest_step_index]),
            })
        conversion_pairs.append((adc, signal))
        global_min = min(global_min, float(np.min(signal)))
        global_max = max(global_max, float(np.max(signal)))
        peak_index = int(np.argmax(np.abs(signal)))
        peak = float(signal[peak_index])
        if not global_peak or abs(peak) > abs(float(global_peak["value"])):
            global_peak = {
                "source_file": path.relative_to(ROOT).as_posix(),
                "value": peak,
                "time_s": float(time[peak_index]),
                "adc_value": float(adc[peak_index]),
            }
        stem = path.stem
        records.append({
            "source_file": path.relative_to(ROOT).as_posix(),
            "sample_count": len(signal),
            "duration_s_from_timestamps": float(time[-1] - time[0]),
            "estimated_sampling_rate_hz": rate,
            "timestamp_discontinuities": gap_count,
            "nonfinite_signal_samples": int(np.count_nonzero(~np.isfinite(signal))),
            "zero_signal_fraction": float(np.mean(signal == 0)),
            "adjacent_equal_fraction": float(np.mean(equal)),
            "longest_exact_repeat_samples": longest_run,
            "longest_exact_repeat_value_recorded_units": longest_run_value,
            "constant_250_sample_windows": constant_250,
            "unique_adc_values": int(np.unique(adc).size),
            "adc_min": float(np.min(adc)),
            "adc_max": float(np.max(adc)),
            "max_abs_adjacent_step_recorded_units": float(np.max(abs_steps)) if len(abs_steps) else 0.0,
            "q999_abs_adjacent_step_recorded_units": float(np.quantile(abs_steps, 0.999)) if len(abs_steps) else 0.0,
            "signal_min_recorded_units": float(np.min(signal)),
            "signal_max_recorded_units": float(np.max(signal)),
            "signal_q005_recorded_units": float(np.quantile(signal, 0.005)),
            "signal_q995_recorded_units": float(np.quantile(signal, 0.995)),
            "max_abs_signal_recorded_units": abs(peak),
            "time_of_max_abs_s": float(time[peak_index]),
            "adc_at_max_abs": float(adc[peak_index]),
        })
    qc = pd.DataFrame(records)
    all_adc = np.concatenate([pair[0] for pair in conversion_pairs])
    all_voltage = np.concatenate([pair[1] for pair in conversion_pairs])
    fit = np.polyfit(all_adc, all_voltage, 1)
    residual = all_voltage - (fit[0] * all_adc + fit[1])
    steps = pd.DataFrame(step_events).nlargest(10, "abs_step_recorded_units")
    steps.to_csv(RESULTS / "eda_largest_adjacent_steps.csv", index=False)
    summary: dict[str, float | int] = {
        "raw_files": len(qc),
        "raw_samples": all_count,
        "participants": len(list(p for p in DATA.iterdir() if p.is_dir())),
        "files_with_gaps": int((qc["timestamp_discontinuities"] > 0).sum()),
        "timestamp_discontinuities": int(qc["timestamp_discontinuities"].sum()),
        "largest_gap_interval_s": float(
            max((np.round(np.diff(pd.read_csv(ROOT / row.source_file)["Relative_Time_s"].to_numpy(dtype=float)), 9).max()
                 for row in qc.itertuples()), default=np.nan)
        ),
        "estimated_fs_min_hz": float(min(rates)),
        "estimated_fs_max_hz": float(max(rates)),
        "raw_signal_min": global_min,
        "raw_signal_max": global_max,
        "raw_zero_fraction": all_zero / all_count,
        "adjacent_equal_fraction": equal_adjacent_total / max(all_count - len(qc), 1),
        "constant_250_sample_windows": constant_250_total,
        "adc_min": adc_global_min,
        "adc_max": adc_global_max,
        "adc_voltage_linear_slope": float(fit[0]),
        "adc_voltage_linear_intercept": float(fit[1]),
        "adc_voltage_linear_r2": float(1 - np.sum(residual ** 2) / np.sum((all_voltage - np.mean(all_voltage)) ** 2)),
        "adc_voltage_residuala_abs_q99": float(np.quantile(np.abs(residual), 0.99)),
        "adc_voltage_residual_abs_max": float(np.max(np.abs(residual))),
        "longest_exact_repeat_samples": int(qc.longest_exact_repeat_samples.max()),
        "largest_step_recorded_units": float(steps.abs_step_recorded_units.max()),
        "largest_absolute_sample": global_peak,
    }
    return qc, summary


def make_trace_figure(windows: pd.DataFrame) -> None:
    # Deterministic illustrative paired recording, chosen by pseudonymous ID.
    candidates = windows[(windows.subject_id == "SUBJ01") & (windows.repetition_id == "R1")]
    records = candidates.groupby("side")["source_file"].first().to_dict()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)
    for ax, side in zip(axes, ("right", "left")):
        relpath = records[side]
        raw = pd.read_csv(ROOT / relpath)
        time = raw["Relative_Time_s"].to_numpy(dtype=float)
        signal = raw["Voltage_mV"].to_numpy(dtype=float)
        shown = time <= time[0] + 2.0
        ax.plot(time[shown] - time[0], signal[shown], linewidth=0.65, color="#245b78")
        ax.set_title(f"SUBJ01 · R1 · {side}", loc="left", fontsize=10)
        ax.set_ylabel("Voltage_mV column\n(recorded units)")
        ax.grid(True, linewidth=0.35, alpha=0.4)
    axes[-1].set_xlabel("Time from recording start (s)")
    fig.suptitle("Example raw traces (first 2 s; not a rest/reference segment)", fontsize=12)
    fig.savefig(RESULTS / "eda_raw_trace_example.png", dpi=180)
    plt.close(fig)


def make_extreme_excerpt(qc: pd.DataFrame) -> dict[str, object]:
    peak_record = qc.loc[qc.max_abs_signal_recorded_units.idxmax()]
    path = ROOT / peak_record.source_file
    raw = pd.read_csv(path)
    time = raw["Relative_Time_s"].to_numpy(dtype=float)
    signal = raw["Voltage_mV"].to_numpy(dtype=float)
    peak_time = float(peak_record.time_of_max_abs_s)
    shown = (time >= peak_time - 0.05) & (time <= peak_time + 0.05)
    folders = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    subject = f"SUBJ{folders.index(path.parent.name) + 1:02d}"
    fig, ax = plt.subplots(figsize=(9, 3.5), constrained_layout=True)
    ax.plot(time[shown] - peak_time, signal[shown], color="#245b78", linewidth=0.9)
    peak_index = int(np.flatnonzero(time == peak_time)[0])
    ax.scatter([0], [signal[peak_index]], color="#a14f3d", zorder=3, label="Largest absolute sample")
    ax.axvline(0, color="#a14f3d", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("Time relative to peak (s)")
    ax.set_ylabel("Voltage_mV column (recorded units)")
    ax.set_title(f"Largest recorded excursion · {subject} · {path.stem.split('_')[0]} · {path.stem.split('_')[1]}", loc="left", fontsize=11)
    ax.grid(True, linewidth=0.35, alpha=0.4)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / "eda_largest_excursion_excerpt.png", dpi=180)
    plt.close(fig)
    return {
        "source_file": peak_record.source_file,
        "subject_id": subject,
        "record_id": path.stem,
        "value": float(signal[peak_index]),
        "time_s": peak_time,
        "adc_value": float(peak_record.adc_at_max_abs),
    }


def make_amplitude_figure(windows: pd.DataFrame) -> pd.DataFrame:
    subjects = sorted(windows.subject_id.unique())
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    rows = []
    for ax, side in zip(axes, ("right", "left")):
        subset = windows[windows.side == side]
        values = [subset.loc[subset.subject_id == subject, "rms"].to_numpy() for subject in subjects]
        ax.boxplot(values, tick_labels=subjects, showfliers=False, whis=(5, 95))
        ax.set_yscale("log")
        ax.set_ylabel("RMS (Voltage_mV column; log scale)")
        ax.set_title(f"{side.title()} side", loc="left", fontsize=10)
        ax.grid(True, axis="y", linewidth=0.35, alpha=0.4)
        for subject, group in subset.groupby("subject_id"):
            rows.append({
                "subject_id": subject,
                "side": side,
                "window_count": len(group),
                "rms_median_recorded_units": float(group.rms.median()),
                "rms_q25_recorded_units": float(group.rms.quantile(0.25)),
                "rms_q75_recorded_units": float(group.rms.quantile(0.75)),
                "mav_median_recorded_units": float(group.mav.median()),
            })
    axes[-1].set_xlabel("Pseudonymous participant ID")
    axes[-1].tick_params(axis="x", labelrotation=45)
    fig.suptitle("Window RMS by participant and side (boxes: 5th–95th percentiles)", fontsize=12)
    fig.savefig(RESULTS / "eda_rms_by_participant_side.png", dpi=180)
    plt.close(fig)
    return pd.DataFrame(rows)


def make_spectral_figure(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = windows.copy()
    frame["mid_time_s"] = (frame.start_time_s + frame.end_time_s) / 2
    frame["time_bin_s"] = np.floor(frame.mid_time_s).astype(int)
    # First summarize repetitions within participant and time bin so participants
    # with more usable files do not dominate the descriptive group trajectory.
    participant_rows = []
    for metric in METRICS:
        participant_rows.append(
            frame.groupby(["subject_id", "side", "time_bin_s"], as_index=False)[metric]
            .median().assign(metric=metric).rename(columns={metric: "value_hz"})
        )
    per_subject = pd.concat(participant_rows, ignore_index=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    group_rows = []
    colors = {"right": "#245b78", "left": "#a14f3d"}
    for row, metric in enumerate(METRICS):
        for col, side in enumerate(("right", "left")):
            ax = axes[row, col]
            chosen = per_subject[(per_subject.metric == metric) & (per_subject.side == side)]
            for _, subject_data in chosen.groupby("subject_id"):
                ax.plot(subject_data.time_bin_s, subject_data.value_hz, color=colors[side], alpha=0.18, linewidth=0.7)
            stats = chosen.groupby("time_bin_s").value_hz.agg(
                median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75),
                participants="count",
            ).reset_index()
            plotted = stats[stats.participants >= 5]
            ax.fill_between(plotted.time_bin_s, plotted.q25, plotted.q75, color=colors[side], alpha=0.15, linewidth=0)
            ax.plot(plotted.time_bin_s, plotted["median"], color=colors[side], linewidth=1.8)
            ax.set_title(f"{metric.replace('_hz_raw_unfiltered','').upper()} · {side}", loc="left", fontsize=10)
            ax.set_ylabel("Frequency (Hz)")
            ax.grid(True, linewidth=0.35, alpha=0.4)
            for item in stats.to_dict("records"):
                group_rows.append({"metric": metric, "side": side, **item})
    for ax in axes[-1, :]:
        ax.set_xlabel("Time from start of each separate recording (s)")
    fig.suptitle("Unfiltered MDF/MNF within recordings; summary shown where ≥5 participants", fontsize=12)
    fig.savefig(RESULTS / "eda_spectral_trajectories.png", dpi=180)
    plt.close(fig)
    return per_subject, pd.DataFrame(group_rows)


def spectral_slopes(windows: pd.DataFrame) -> pd.DataFrame:
    rows = []
    frame = windows.copy()
    frame["mid_time_s"] = (frame.start_time_s + frame.end_time_s) / 2
    for (subject, side, record), group in frame.groupby(["subject_id", "side", "record_id"]):
        row: dict[str, object] = {"subject_id": subject, "side": side, "record_id": record}
        x = group.mid_time_s.to_numpy(dtype=float)
        for metric in METRICS:
            y = group[metric].to_numpy(dtype=float)
            row[f"{metric}_slope_hz_per_s"] = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def slope_sensitivity(windows: pd.DataFrame) -> pd.DataFrame:
    """Describe how within-file linear slopes change when early windows are omitted."""
    frame = windows.copy()
    frame["mid_time_s"] = (frame.start_time_s + frame.end_time_s) / 2
    rows = []
    for metric in METRICS:
        for excluded_s in (0, 1, 2, 5):
            for side, side_data in frame.groupby("side"):
                record_slopes = []
                for _, group in side_data.groupby("record_id"):
                    x = group.mid_time_s.to_numpy(dtype=float)
                    y = group[metric].to_numpy(dtype=float)
                    keep = x >= x.min() + excluded_s
                    if np.count_nonzero(keep) >= 2:
                        record_slopes.append(float(np.polyfit(x[keep], y[keep], 1)[0]))
                slopes = np.asarray(record_slopes)
                rows.append({
                    "metric": metric,
                    "side": side,
                    "initial_seconds_excluded": excluded_s,
                    "record_count": len(slopes),
                    "median_slope_hz_per_s": float(np.median(slopes)),
                    "negative_slope_fraction": float(np.mean(slopes < 0)),
                })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Run src/features.py first; not found: {FEATURES_PATH}")
    windows = pd.read_csv(FEATURES_PATH)
    expected = {"subject_id", "record_id", "source_file", "side", "start_time_s", "end_time_s", *METRICS}
    missing = expected.difference(windows.columns)
    if missing:
        raise ValueError(f"Corrected dataset is missing required columns: {', '.join(sorted(missing))}")
    if windows.label_phase.fillna("").ne("").any():
        raise ValueError("Expected unlabeled exploratory data; found nonempty label_phase values")
    if not windows[list(METRICS)].apply(np.isfinite).all().all():
        raise ValueError("MDF/MNF columns contain nonfinite values")

    qc, raw_summary = raw_quality_table()
    qc.to_csv(RESULTS / "eda_raw_record_qc.csv", index=False)
    make_trace_figure(windows)
    extreme = make_extreme_excerpt(qc)
    amplitude_summary = make_amplitude_figure(windows)
    amplitude_summary.to_csv(RESULTS / "eda_amplitude_by_participant_side.csv", index=False)
    _, spectral_summary = make_spectral_figure(windows)
    spectral_summary.to_csv(RESULTS / "eda_spectral_trajectory_by_second.csv", index=False)
    initial_spectral = {}
    for side in ("right", "left"):
        first = spectral_summary.query("metric == @METRICS[0] and side == @side and time_bin_s == 0")
        second = spectral_summary.query("metric == @METRICS[0] and side == @side and time_bin_s == 1")
        initial_spectral[side] = (float(first["median"].iloc[0]), float(second["median"].iloc[0]))
    slopes = spectral_slopes(windows)
    slopes.to_csv(RESULTS / "eda_spectral_slopes_by_record.csv", index=False)
    sensitivity = slope_sensitivity(windows)
    sensitivity.to_csv(RESULTS / "eda_spectral_slope_sensitivity.csv", index=False)

    median_mdf_slope = float(slopes["mdf_hz_filtered_slope_hz_per_s"].median())
    mdf_negative_fraction = float((slopes["mdf_hz_filtered_slope_hz_per_s"] < 0).mean())
    mdf_1s = sensitivity[(sensitivity.metric == METRICS[0]) & (sensitivity.initial_seconds_excluded == 1)]
    mnf_1s = sensitivity[(sensitivity.metric == METRICS[1]) & (sensitivity.initial_seconds_excluded == 1)]
    mdf_1s_median = float(mdf_1s.median_slope_hz_per_s.median())
    mnf_1s_median = float(mnf_1s.median_slope_hz_per_s.median())
    mdf_1s_negative = float(mdf_1s.negative_slope_fraction.mean())
    mnf_1s_negative = float(mnf_1s.negative_slope_fraction.mean())
    rms_median = float(windows.rms.median())
    rms_range = (float(windows.rms.min()), float(windows.rms.max()))
    duration_summary = qc["duration_s_from_timestamps"].describe()
    content = f"""Exploratory analysis of corrected raw EMG windows
Date: 2026-10-17

Scope and safeguards
--------------------
This report uses data/features_dataset_corrected_unlabeled.csv, which reads the Voltage_mV column and leaves labels blank. No filtering, interpolation, resampling, baseline correction, or fatigue classification was applied. The column name says mV, but calibration/units were not independently verified. MDF/MNF use one demeaned, Hann-windowed FFT periodogram per 250 ms window, with DC omitted and the rest of the spectrum left unfiltered; these values are exploratory. A 250 ms window gives nominal 4 Hz Fourier-bin spacing. The x-axis in the spectral figure is time from the start of each separate recording; it is not cumulative exercise time across repetitions. Group medians and interquartile bands are shown only for bins with at least five participants; thinner coverage remains available in the CSV.

Dataset and acquisition observations
-------------------------------------
- Raw files: {raw_summary['raw_files']}; participants: {raw_summary['participants']}; raw samples: {raw_summary['raw_samples']:,}.
- Processed windows: {len(windows):,}; source recordings represented: {windows.record_id.nunique()}; participant IDs represented: {windows.subject_id.nunique()}.
- Median timestamp increment yields {raw_summary['estimated_fs_min_hz']:.3f} to {raw_summary['estimated_fs_max_hz']:.3f} Hz across files. This is inferred from Relative_Time_s, not verified against acquisition settings.
- Timestamp discontinuities: {raw_summary['timestamp_discontinuities']} in {raw_summary['files_with_gaps']} files. Largest observed interval: {raw_summary['largest_gap_interval_s']*1000:.1f} ms. Windows crossing a discontinuity are excluded.
- Recording duration from first to last timestamp: median {duration_summary['50%']:.2f} s; range {duration_summary['min']:.2f}-{duration_summary['max']:.2f} s. Each file is a separate recording and timestamps reset; repetition/event semantics are not documented.
- All generated time-feature and spectral values are finite. The raw signal numeric range across files is {raw_summary['raw_signal_min']:.3f} to {raw_summary['raw_signal_max']:.3f}; {100*raw_summary['raw_zero_fraction']:.3f}% of raw signal values are exactly zero. No ADC rail limits are documented, so clipping/saturation cannot be diagnosed from this alone.
- Across adjacent raw samples, {100*raw_summary['adjacent_equal_fraction']:.3f}% have exactly equal recorded values; the longest exact repeated-value run is {raw_summary['longest_exact_repeat_samples']} samples ({raw_summary['longest_exact_repeat_samples']/raw_summary['estimated_fs_min_hz']*1000:.1f} ms at the timestamp-derived rate). No fully constant 250-sample window was found. This argues against an obvious sustained flatline, but does not rule out acquisition dropouts or clipping.
- The recorded ADC values span {raw_summary['adc_min']:.0f} to {raw_summary['adc_max']:.0f} counts. A linear fit between the two stored columns gives slope {raw_summary['adc_voltage_linear_slope']:.6f} recorded units/count, intercept {raw_summary['adc_voltage_linear_intercept']:.4f}, R² {raw_summary['adc_voltage_linear_r2']:.8f}; absolute residual q99/max are {raw_summary['adc_voltage_residual_abs_q99']:.4f}/{raw_summary['adc_voltage_residual_abs_max']:.4f}. This describes the CSV columns only and does not establish physical calibration or ADC rails.
- Abrupt adjacent-sample changes occur: the largest absolute step is {raw_summary['largest_step_recorded_units']:.2f} recorded units. The 10 largest per-record steps and times are listed in eda_largest_adjacent_steps.csv for review; they are candidates, not automatically artifacts.
- The largest absolute sample is {extreme['value']:.2f} recorded units ({extreme['adc_value']:.0f} ADC counts) at {extreme['time_s']:.4f} s in {extreme['subject_id']}, {extreme['record_id']}. The neighboring samples form a brief high-amplitude excursion; this is a candidate transient/artifact for review, not a confirmed clipped sample. See eda_largest_excursion_excerpt.png.

Amplitude and spectral observations
-----------------------------------
- Across windows, RMS median is {rms_median:.3f} recorded units (range {rms_range[0]:.3f}-{rms_range[1]:.3f}); see the participant/side figure and CSV for distributions. Cross-participant differences may reflect calibration, placement, force, or other factors; units and protocol do not allow those causes to be separated here.
- Full-record median MDF slope is {median_mdf_slope:.4f} Hz/s ({100*mdf_negative_fraction:.1f}% of slopes negative). Omitting only the first second changes the side-median MDF slope to {mdf_1s_median:.4f} Hz/s and the average side-level negative-slope fraction to {100*mdf_1s_negative:.1f}%; MNF changes to {mnf_1s_median:.4f} Hz/s and {100*mnf_1s_negative:.1f}%. This sensitivity indicates that early spectral excursions strongly affect the apparent direction. These are descriptive within-file fits, not fatigue estimates. Details for 0, 1, 2, and 5 s omissions are in eda_spectral_slope_sensitivity.csv.
- MDF/MNF vary within and between files. The binned figure gives participant-level median trajectories and interquartile ranges by side. It summarizes within-record elapsed time only; it does not join repetitions into a single exercise progression. The early and late spectral excursions are not interpreted as fatigue because their signal/event context is unknown.
- In the first 1 s bin, participant-level median MDF is {initial_spectral['right'][0]:.0f} Hz (right) and {initial_spectral['left'][0]:.0f} Hz (left); in the next bin it is {initial_spectral['right'][1]:.0f} and {initial_spectral['left'][1]:.0f} Hz. This large initial shift across all 15 participants makes the first window a questionable baseline; it may reflect recording startup or other signal/context effects, which the files cannot distinguish.
- No labeled rest/reference segment or contraction/event markers are present in the files. A defensible initial reference and a temporal fatigue progression therefore remain unconfirmed.

Interpretation and next decision
--------------------------------
These plots support checking the data structure and describing recorded signals, but do not justify thresholds for four fatigue phases. Before labels or models, confirm electrode/muscle identity, voltage calibration, acquisition filtering and protocol, then determine whether repetition files contain a continuous contraction and where a baseline/reference lies. If this cannot be recovered, the project should state that the labels are exploratory operational categories and avoid claims of physiological fatigue stages.

Generated figures
-----------------
- eda_raw_trace_example.png: first two seconds of SUBJ01, repetition R1, by side; illustrative trace, not identified rest.
- eda_largest_excursion_excerpt.png: 100 ms around the largest absolute recorded sample, shown for quality review.
- eda_rms_by_participant_side.png: window RMS distributions by pseudonymous participant and side.
- eda_spectral_trajectories.png: unfiltered MDF/MNF over time within separate recordings, by side.
- eda_largest_adjacent_steps.csv: largest within-record adjacent-sample steps, retained as review candidates.
"""
    (RESULTS / "exploratory_analysis.txt").write_text(content, encoding="utf-8")
    print(f"Analysis saved to {RESULTS}")
    print(f"Files={raw_summary['raw_files']}; windows={len(windows)}; participants={windows.subject_id.nunique()}")
    print(f"Timestamp gaps={raw_summary['timestamp_discontinuities']} in {raw_summary['files_with_gaps']} files; fs={raw_summary['estimated_fs_min_hz']:.3f}-{raw_summary['estimated_fs_max_hz']:.3f} Hz")
    print(f"MDF median per-record slope={median_mdf_slope:.4f} Hz/s; negative slopes={100*mdf_negative_fraction:.1f}%")
    print(f"MDF median slope after omitting first second={mdf_1s_median:.4f} Hz/s; MNF={mnf_1s_median:.4f} Hz/s")


if __name__ == "__main__":
    main()
