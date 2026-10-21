"""Detect candidate activation peaks and summarize their progression.

This is an exploratory aid, not a validated fatigue detector. An Otsu split on
the smoothed RMS envelope proposes a per-file activity cutoff; prominent peaks
above it, with a minimum separation, are provisionally treated as repetitions.
Counts and phase assignments must be checked against video/manual annotations.
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


R_ORDER = ["R0", "R0_1", "R1", "R2", "R3", "R4", "R5"]
STAGE_NAMES = ["early", "middle", "late"]
FEATURE_COLUMNS = ["rms", "mav", "variance", "waveform_length", "mdf_hz_filtered", "mnf_hz_filtered"]
SIDE_COLORS = {"right": "#31688e", "left": "#d17a22"}


def otsu_cutoff(values: np.ndarray) -> tuple[float, float, float]:
    """Return RMS onset cutoff, lower-state median, and Otsu separability."""
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if not len(clean):
        return float("nan"), float("nan"), 0.0
    transformed = np.log1p(np.maximum(clean, 0.0))
    total_variance = float(np.var(transformed))
    baseline = float(np.quantile(clean, 0.2))
    if len(clean) < 10 or total_variance <= np.finfo(float).eps:
        return float(np.max(clean)), baseline, 0.0

    bins = min(256, max(16, int(np.sqrt(len(transformed)))) )
    hist, edges = np.histogram(transformed, bins=bins)
    probability = hist.astype(float) / hist.sum()
    centers = (edges[:-1] + edges[1:]) / 2
    omega = np.cumsum(probability)
    mean_cumulative = np.cumsum(probability * centers)
    mean_total = mean_cumulative[-1]
    denominator = omega * (1 - omega)
    between = np.zeros_like(denominator)
    valid = denominator > 0
    between[valid] = (mean_total * omega[valid] - mean_cumulative[valid]) ** 2 / denominator[valid]
    # Do not choose a split that leaves fewer than about 5% of windows on one side.
    valid &= (omega >= 0.05) & (omega <= 0.95)
    if not valid.any():
        return float(np.quantile(clean, 0.75)), baseline, 0.0
    index = int(np.argmax(np.where(valid, between, -np.inf)))
    cutoff = float(np.expm1(centers[index]))
    low = clean[clean <= cutoff]
    if len(low):
        baseline = float(np.median(low))
    separability = float(np.clip(between[index] / total_variance, 0.0, 1.0))
    return cutoff, baseline, separability


def assign_stages(count: int) -> list[str]:
    if count < 3:
        return ["insufficient_bouts"] * count
    labels = []
    for i in range(count):
        stage_index = min(int(((i + 0.5) / count) * 3), 2)
        labels.append(STAGE_NAMES[stage_index])
    return labels


def draw_record_plot(
    root: Path,
    record: pd.DataFrame,
    bouts: pd.DataFrame,
    onset: float,
    output_path: Path,
    signal_column: str,
) -> None:
    info = record.iloc[0]
    raw_path = root / str(info["source_file"])
    raw = pd.read_csv(raw_path, usecols=["Relative_Time_s", signal_column])
    raw_time = pd.to_numeric(raw["Relative_Time_s"], errors="raise").to_numpy(dtype=float)
    raw_signal = pd.to_numeric(raw[signal_column], errors="raise").to_numpy(dtype=float)
    raw_time = raw_time - raw_time[0]
    raw_signal = raw_signal.copy()
    dt = float(np.median(np.diff(raw_time))) if len(raw_time) > 1 else np.nan
    if np.isfinite(dt):
        breaks = np.flatnonzero(np.diff(raw_time) > dt * 1.01) + 1
        raw_time[breaks] = np.nan
        raw_signal[breaks] = np.nan

    t = ((record["start_time_s"].to_numpy(float) + record["end_time_s"].to_numpy(float)) / 2)
    t = t - float(record["start_time_s"].iloc[0])
    smooth = record["rms_smooth"].to_numpy(float)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(raw_time, raw_signal, color="#496a81", linewidth=0.45, rasterized=True)
    axes[0].set_ylabel(f"{signal_column}\n(units unverified)")
    axes[0].set_title(
        f"{info['subject_id']} | {info['repetition_id']} | {info['side']} | "
        "candidate repetition peaks"
    )
    axes[0].grid(alpha=0.18)

    axes[1].plot(t, smooth, color="#a94a32", linewidth=1.0, label="Smoothed RMS envelope")
    axes[1].axhline(onset, color="#b21f35", linestyle="--", linewidth=1, label="Otsu activity cutoff")
    for _, bout in bouts.iterrows():
        axes[1].axvspan(bout["cycle_start_s"], bout["cycle_end_s"], color="#62a87c", alpha=0.10)
        axes[2].axvspan(bout["cycle_start_s"], bout["cycle_end_s"], color="#62a87c", alpha=0.08)
        axes[1].axvline(bout["peak_time_s"], color="#315d3d", linewidth=0.7, alpha=0.7)
        axes[2].axvline(bout["peak_time_s"], color="#315d3d", linewidth=0.7, alpha=0.7)
        axes[1].text(bout["peak_time_s"], bout["peak_smoothed_rms"], str(int(bout["candidate_rep"])),
                     ha="center", va="bottom", fontsize=7, color="#315d3d")
    axes[1].set_ylabel("RMS (recorded units)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    axes[1].grid(alpha=0.18)

    axes[2].plot(t, record["mdf_hz_filtered"], color="#31688e", linewidth=0.8, label="MDF")
    axes[2].plot(t, record["mnf_hz_filtered"], color="#d08c28", linewidth=0.8, label="MNF")
    axes[2].set_ylabel("Frequency (Hz)\n(unfiltered, provisional)")
    axes[2].set_xlabel("Time within this file (s)")
    axes[2].legend(frameon=False, ncol=2, loc="upper right")
    axes[2].grid(alpha=0.18)
    fig.suptitle("Automatic segmentation is a candidate; inspect against video/count if available", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=125)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "fatigue_candidates")
    parser.add_argument("--min-peak-distance-seconds", type=float, default=1.2,
                        help="Minimum separation between envelope peaks; default 2 s, provisional from the prescribed cadence")
    parser.add_argument("--min-peak-prominence-fraction", type=float, default=0.15,
                        help="Prominence as a fraction of the record's 90th-20th percentile RMS range")
    parser.add_argument("--signal-column", choices=["Voltage_mV", "ADC_Value"], default="Voltage_mV")
    args = parser.parse_args()
    if args.min_peak_distance_seconds <= 0 or not 0 <= args.min_peak_prominence_fraction <= 1:
        raise ValueError("Peak distance must be positive and prominence fraction must be between zero and one")

    frame = pd.read_csv(args.dataset)
    required = {
        "subject_id", "record_id", "source_file", "repetition_id", "side",
        "segment_id", "start_time_s", "end_time_s", *FEATURE_COLUMNS,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature dataset is missing columns: {sorted(missing)}")

    all_bouts: list[dict[str, object]] = []
    record_summaries: list[dict[str, object]] = []
    gallery: list[tuple[str, str]] = []
    records = frame.groupby(["subject_id", "record_id"], sort=True, observed=True)
    for (subject, record_id), original in records:
        record = original.sort_values("start_time_s").reset_index(drop=True).copy()
        pseudonymous_record_id = f"{subject}_{record['repetition_id'].iloc[0]}_{record['side'].iloc[0]}"
        step = float(np.median(np.diff(record["start_time_s"].to_numpy(float)))) if len(record) > 1 else 0.125
        smooth_window = max(3, int(round(0.625 / max(step, 1e-6))) | 1)
        record["rms_smooth"] = record.groupby("segment_id", sort=False)["rms"].transform(
            lambda values: values.rolling(smooth_window, center=True, min_periods=1).median()
        ).to_numpy(float)
        onset, baseline, separability = otsu_cutoff(record["rms_smooth"].to_numpy(float))
        smooth = record["rms_smooth"].to_numpy(float)
        prominence = max(
            0.0,
            float(np.quantile(smooth, 0.9) - np.quantile(smooth, 0.2)) * args.min_peak_prominence_fraction,
        )
        min_distance_windows = max(1, int(np.ceil(args.min_peak_distance_seconds / max(step, 1e-6))))
        midpoints = (record["start_time_s"].to_numpy(float) + record["end_time_s"].to_numpy(float)) / 2
        file_start = float(record["start_time_s"].iloc[0])
        peak_items: list[dict[str, float | int]] = []
        # Detect peaks and define cycle bounds independently within each
        # contiguous timestamp segment; never bridge a recorded data gap.
        for segment_id, segment in record.groupby("segment_id", sort=False):
            positions = segment.index.to_numpy()
            local_peaks, peak_info = find_peaks(
                segment["rms_smooth"].to_numpy(float),
                height=onset,
                distance=min_distance_windows,
                prominence=prominence,
            )
            local_abs_peaks = positions[local_peaks]
            local_times = midpoints[local_abs_peaks]
            segment_start = float(segment["start_time_s"].iloc[0])
            segment_end = float(segment["end_time_s"].iloc[-1])
            for local_index, (peak_idx, peak_time) in enumerate(zip(local_abs_peaks, local_times)):
                previous_interval = local_times[local_index] - local_times[local_index - 1] if local_index > 0 else None
                next_interval = local_times[local_index + 1] - local_times[local_index] if local_index + 1 < len(local_times) else None
                left = (local_times[local_index - 1] + peak_time) / 2 if previous_interval is not None else peak_time - (next_interval or args.min_peak_distance_seconds) / 2
                right = (peak_time + local_times[local_index + 1]) / 2 if next_interval is not None else peak_time + (previous_interval or args.min_peak_distance_seconds) / 2
                peak_items.append({
                    "peak_idx": int(peak_idx),
                    "peak_time": float(peak_time),
                    "cycle_start": float(max(segment_start, left)),
                    "cycle_end": float(min(segment_end, right)),
                    "peak_prominence": float(peak_info["prominences"][local_index]),
                    "segment_id": int(segment_id),
                })
        peak_items.sort(key=lambda item: item["peak_idx"])
        stages = assign_stages(len(peak_items))
        record_bouts: list[dict[str, object]] = []
        for bout_index, (item, stage) in enumerate(zip(peak_items, stages), start=1):
            peak_idx = int(item["peak_idx"])
            peak_time = float(item["peak_time"])
            cycle_start = float(item["cycle_start"])
            cycle_end = float(item["cycle_end"])
            selected = (
                (midpoints >= cycle_start) & (midpoints < cycle_end)
                & (record["segment_id"].to_numpy() == item["segment_id"])
            )
            cycle_window_count = int(selected.sum())
            active_selected = selected & (record["rms_smooth"].to_numpy(float) >= onset)
            chunk = record.loc[active_selected]
            if chunk.empty:
                chunk = record.loc[selected]
            if chunk.empty:
                chunk = record.iloc[[int(peak_idx)]]
            start_s = float(cycle_start - file_start)
            end_s = float(cycle_end - file_start)
            row: dict[str, object] = {
                "subject_id": subject,
                "record_id": pseudonymous_record_id,
                "repetition_code": record["repetition_id"].iloc[0],
                "side": record["side"].iloc[0],
                "candidate_rep": bout_index,
                "stage_by_candidate_rep_order": stage,
                "segment_id": item["segment_id"],
                "cycle_start_s": start_s,
                "cycle_end_s": end_s,
                "estimated_cycle_duration_s": end_s - start_s,
                "peak_time_s": float(peak_time - file_start),
                "mean_rms": float(chunk["rms"].mean()),
                "peak_smoothed_rms": float(smooth[int(peak_idx)]),
                "peak_prominence_rms": item["peak_prominence"],
                "mean_mav": float(chunk["mav"].mean()),
                "mean_variance": float(chunk["variance"].mean()),
                "mean_waveform_length": float(chunk["waveform_length"].mean()),
                "mean_mdf_hz_unfiltered": float(chunk["mdf_hz_filtered"].mean()),
                "mean_mnf_hz_unfiltered": float(chunk["mnf_hz_filtered"].mean()),
                "window_count": len(chunk),
                "cycle_window_count": cycle_window_count,
                "features_use_active_windows_only": True,
                "threshold_onset_rms": onset,
                "baseline_rms_estimate": baseline,
                "otsu_separability": separability,
                "manual_validation_required": True,
            }
            record_bouts.append(row)
            all_bouts.append(row)

        # Per-record feature means for each of the three ordinal bout groups.
        bout_frame = pd.DataFrame(record_bouts)
        phase_values: dict[str, float] = {}
        if len(bout_frame):
            for stage in STAGE_NAMES:
                subset = bout_frame[bout_frame["stage_by_candidate_rep_order"] == stage]
                for col in ["mean_rms", "mean_mdf_hz_unfiltered", "mean_mnf_hz_unfiltered"]:
                    phase_values[f"{stage}_{col}"] = float(subset[col].mean()) if len(subset) else float("nan")
        n_bouts = len(record_bouts)
        mdf_early, mdf_late = phase_values.get("early_mean_mdf_hz_unfiltered", np.nan), phase_values.get("late_mean_mdf_hz_unfiltered", np.nan)
        mnf_early, mnf_late = phase_values.get("early_mean_mnf_hz_unfiltered", np.nan), phase_values.get("late_mean_mnf_hz_unfiltered", np.nan)
        rms_early, rms_late = phase_values.get("early_mean_rms", np.nan), phase_values.get("late_mean_rms", np.nan)
        has_all_stages = n_bouts >= 3 and np.all(np.isfinite([mdf_early, mdf_late, mnf_early, mnf_late, rms_early, rms_late]))
        record_summaries.append({
            "subject_id": subject,
            "record_id": pseudonymous_record_id,
            "repetition_code": record["repetition_id"].iloc[0],
            "side": record["side"].iloc[0],
            "window_count": len(record),
            "estimated_step_s": step,
            "otsu_onset_threshold_rms": onset,
            "estimated_low_state_median_rms": baseline,
            "otsu_separability": separability,
            "candidate_rep_count": n_bouts,
            "stage_comparison_available": n_bouts >= 3,
            "candidate_MDF_and_MNF_decline_early_to_late": bool(has_all_stages and mdf_late < mdf_early and mnf_late < mnf_early),
            "RMS_increase_early_to_late": bool(has_all_stages and rms_late > rms_early),
            "median_estimated_cycle_duration_s": float(np.median([b["estimated_cycle_duration_s"] for b in record_bouts])) if n_bouts else np.nan,
            "median_inter_peak_interval_s": float(np.median([
                float(current["peak_time"] - previous["peak_time"])
                for previous, current in zip(peak_items[:-1], peak_items[1:])
                if previous["segment_id"] == current["segment_id"]
            ])) if any(
                previous["segment_id"] == current["segment_id"]
                for previous, current in zip(peak_items[:-1], peak_items[1:])
            ) else np.nan,
            "manual_validation_required": True,
            **phase_values,
        })

        rep = str(record["repetition_id"].iloc[0])
        side = str(record["side"].iloc[0])
        image_path = Path(str(subject)) / f"{rep}_{side}_candidate_reps.png"
        draw_record_plot(
            ROOT, record, bout_frame, onset, args.output_dir / "plots" / image_path, args.signal_column
        )
        gallery.append((str(subject), (Path("plots") / image_path).as_posix()))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    bouts_df = pd.DataFrame(all_bouts)
    summaries_df = pd.DataFrame(record_summaries)
    bouts_df.to_csv(args.output_dir / "candidate_repetitions.csv", index=False)
    summaries_df.to_csv(args.output_dir / "recording_segmentation_summary.csv", index=False)

    # Give each participant equal weight in the stage summary, rather than
    # pooling all overlapping windows or bouts as independent observations.
    usable = summaries_df[summaries_df["stage_comparison_available"]]
    stage_records = []
    for _, record in usable.iterrows():
        for stage in STAGE_NAMES:
            stage_records.append({
                "subject_id": record["subject_id"],
                "side": record["side"],
                "stage": stage,
                "mean_rms": record.get(f"{stage}_mean_rms", np.nan),
                "mean_mdf_hz_unfiltered": record.get(f"{stage}_mean_mdf_hz_unfiltered", np.nan),
                "mean_mnf_hz_unfiltered": record.get(f"{stage}_mean_mnf_hz_unfiltered", np.nan),
            })
    stage_frame = pd.DataFrame(stage_records)
    if len(stage_frame):
        # First average any multiple files per person/side/stage, then across participants.
        participant_stage = stage_frame.groupby(["subject_id", "side", "stage"], observed=True, as_index=False).mean(numeric_only=True)
        aggregate = participant_stage.groupby(["side", "stage"], observed=True).agg(
            participant_count=("subject_id", "nunique"),
            mean_rms=("mean_rms", "mean"), sd_rms=("mean_rms", "std"),
            mean_mdf_hz_unfiltered=("mean_mdf_hz_unfiltered", "mean"),
            sd_mdf_hz_unfiltered=("mean_mdf_hz_unfiltered", "std"),
            mean_mnf_hz_unfiltered=("mean_mnf_hz_unfiltered", "mean"),
            sd_mnf_hz_unfiltered=("mean_mnf_hz_unfiltered", "std"),
        ).reset_index()
        aggregate.to_csv(args.output_dir / "exploratory_stage_summary.csv", index=False)
        fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
        trend_columns = [
            ("mean_rms", "sd_rms", "Mean active-window RMS (recorded units)"),
            ("mean_mdf_hz_unfiltered", "sd_mdf_hz_unfiltered", "Mean MDF (Hz, unfiltered/provisional)"),
            ("mean_mnf_hz_unfiltered", "sd_mnf_hz_unfiltered", "Mean MNF (Hz, unfiltered/provisional)"),
        ]
        stage_x = {"early": 0, "middle": 1, "late": 2}
        for ax, (mean_col, sd_col, ylabel) in zip(axes, trend_columns):
            for side, color in SIDE_COLORS.items():
                series = aggregate[aggregate["side"] == side].set_index("stage").reindex(STAGE_NAMES)
                if series[mean_col].notna().any():
                    y = series[mean_col].to_numpy(float)
                    sd = series[sd_col].fillna(0).to_numpy(float)
                    x = np.array([stage_x[name] for name in STAGE_NAMES])
                    ax.plot(x, y, marker="o", color=color, linewidth=1.8, label=side)
                    lower = np.maximum(y - sd, 0) if mean_col == "mean_rms" else y - sd
                    ax.fill_between(x, lower, y + sd, color=color, alpha=0.15, linewidth=0)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.legend(frameon=False)
        axes[-1].set_xticks([0, 1, 2], STAGE_NAMES)
        axes[-1].set_xlabel("Stage by order of detected envelope peaks")
        fig.suptitle("Exploratory trend across candidate repetition stages\nPeaks and fatigue interpretation are not yet independently validated", fontsize=11)
        fig.savefig(args.output_dir / "candidate_stage_trends.png", dpi=150)
        plt.close(fig)
        change_rows = []
        for side, side_group in participant_stage.groupby("side", observed=True):
            for feature in ["mean_rms", "mean_mdf_hz_unfiltered", "mean_mnf_hz_unfiltered"]:
                pivot = side_group.pivot(index="subject_id", columns="stage", values=feature)
                if not {"early", "late"}.issubset(pivot.columns):
                    continue
                paired = pivot[["early", "late"]].dropna()
                delta = paired["late"] - paired["early"]
                percent = delta / paired["early"].replace(0, np.nan) * 100
                change_rows.append({
                    "side": side,
                    "feature": feature,
                    "participant_count": len(paired),
                    "mean_early": paired["early"].mean(),
                    "mean_late": paired["late"].mean(),
                    "mean_paired_change_late_minus_early": delta.mean(),
                    "mean_paired_percent_change": percent.mean(),
                    "sd_paired_change": delta.std(),
                })
        pd.DataFrame(change_rows).to_csv(args.output_dir / "early_vs_late_change_by_side.csv", index=False)
    else:
        pd.DataFrame(columns=["side", "stage", "participant_count", "mean_rms", "mean_mdf_hz_unfiltered", "mean_mnf_hz_unfiltered"]).to_csv(
            args.output_dir / "exploratory_stage_summary.csv", index=False
        )
        pd.DataFrame().to_csv(args.output_dir / "early_vs_late_change_by_side.csv", index=False)

    by_subject: dict[str, list[str]] = {}
    for subject, image in gallery:
        by_subject.setdefault(subject, []).append(image)
    sections = []
    for subject in sorted(by_subject):
        links = []
        for image in sorted(by_subject[subject]):
            label = html.escape(Path(image).stem.replace("_candidate_bouts", ""))
            safe_path = html.escape(image, quote=True)
            links.append(f'<a href="{safe_path}"><img loading="lazy" src="{safe_path}" alt="{label}" style="width:min(100%,1000px);height:auto"></a>')
        sections.append(f"<section><h2>{html.escape(subject)}</h2>{''.join(links)}</section>")
    document = """<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Picos candidatos de repetición</title><style>body{font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}
section{border-top:1px solid #bbb;margin-top:2rem;padding-top:1rem}img{border:1px solid #ddd;margin:1rem 0}
.note{background:#fff4d6;padding:1rem}</style></head><body><h1>Picos candidatos de repetición</h1>
<p class="note">El umbral Otsu y los picos son automáticos y requieren validación manual. Un pico no se confirma como repetición ni como extensión de rodilla.
Las fases temprana/media/tardía se asignan por el orden de picos detectados dentro de cada archivo. MDF/MNF son espectros sin filtrar y no son etiquetas independientes de fatiga.</p>
""" + "\n".join(sections) + "\n</body></html>"
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")

    report = [
        "EXPLORATORY ACTIVATION-EPISODE AND FATIGUE-INDICATOR SUMMARY",
        "Candidate peaks are detected from smoothed RMS above a per-record Otsu activity cutoff.",
        f"Minimum peak separation: {args.min_peak_distance_seconds:g} s; prominence fraction: {args.min_peak_prominence_fraction:g}.",
        "Cycle intervals are midpoint-to-midpoint between peaks; peaks are provisionally treated as repetitions.",
        "Counts and cycle intervals require manual/video validation.",
        "MDF/MNF are unfiltered provisional features. This is not validated physiological fatigue classification.",
        f"Records processed: {len(summaries_df)}",
        f"Records with >=3 candidate peaks: {int(summaries_df['stage_comparison_available'].sum())}",
        f"Total candidate peaks/repetitions: {len(bouts_df)}",
        f"Median candidate peaks per record: {summaries_df['candidate_rep_count'].median():.1f}",
        f"Median within-record inter-peak interval: {summaries_df['median_inter_peak_interval_s'].median():.2f} s",
        f"Mean Otsu separability: {summaries_df['otsu_separability'].mean():.3f}",
        f"Records with lower late MDF and MNF than early: {int(summaries_df['candidate_MDF_and_MNF_decline_early_to_late'].sum())}",
        f"Records with higher late RMS than early: {int(summaries_df['RMS_increase_early_to_late'].sum())}",
        "",
        "Use recording_segmentation_summary.csv to review event counts and threshold diagnostics.",
        "Use candidate_repetitions.csv for per-cycle features and within-file early/middle/late assignments.",
        "Use exploratory_stage_summary.csv for exploratory group summaries; all records currently have at least three candidate peaks.",
        "Use early_vs_late_change_by_side.csv for participant-paired descriptive early-to-late changes.",
        "See candidate_stage_trends.png for the exploratory stage trend plot.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("Candidate envelope-peak analysis complete (not validated fatigue detection).")
    print(f"Records: {len(summaries_df)}; candidate peaks: {len(bouts_df)}; records with >=3 peaks: {int(summaries_df['stage_comparison_available'].sum())}")
    print(f"Review plots: {args.output_dir / 'index.html'}")
    print(f"Candidate repetitions: {args.output_dir / 'candidate_repetitions.csv'}")
    print(f"Per-record summary: {args.output_dir / 'recording_segmentation_summary.csv'}")
    print(f"Stage summary: {args.output_dir / 'exploratory_stage_summary.csv'}")


if __name__ == "__main__":
    main()
