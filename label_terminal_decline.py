"""Create provisional labels for sustained terminal RMS declines.

These are operational signal-pattern labels, not independently verified
physiological fatigue labels. The output keeps inactive and uncertain samples
separate from active samples before the candidate decline.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

from detect_fatigue_candidates import otsu_cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--candidate-peaks", type=Path, default=ROOT / "results" / "fatigue_candidates" / "candidate_repetitions.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "terminal_decline_candidates")
    parser.add_argument("--minimum-duration", type=float, default=1.0,
                        help="Minimum time from terminal candidate peak to threshold crossing, in seconds")
    parser.add_argument("--maximum-end-peak-ratio", type=float, default=0.75,
                        help="Maximum final-active RMS / peak RMS ratio for a candidate decline")
    args = parser.parse_args()

    features = pd.read_csv(args.dataset)
    peaks = pd.read_csv(args.candidate_peaks)
    rows: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []

    for (subject_id, record_id), source in features.groupby(["subject_id", "record_id"], sort=True):
        record = source.sort_values("start_time_s").reset_index(drop=True).copy()
        times = (record["start_time_s"].to_numpy(float) + record["end_time_s"].to_numpy(float)) / 2
        steps = np.diff(record["start_time_s"].to_numpy(float))
        step = float(np.median(steps)) if len(steps) else 0.125
        smooth_window = max(3, int(round(0.625 / max(step, 1e-6))) | 1)
        smooth = record.groupby("segment_id", sort=False)["rms"].transform(
            lambda values: values.rolling(smooth_window, center=True, min_periods=1).median()
        ).to_numpy(float)
        cutoff, baseline, separability = otsu_cutoff(smooth)
        key = f"{subject_id}_{record['repetition_id'].iloc[0]}_{record['side'].iloc[0]}"
        record_peaks = peaks[peaks["record_id"].astype(str).eq(key)].sort_values("peak_time_s")
        record["rms_smooth"] = smooth
        record["activity_cutoff"] = cutoff
        record["terminal_decline_status"] = "inactive_or_uncertain"
        record.loc[smooth >= cutoff, "terminal_decline_status"] = "active_before_candidate"

        event: dict[str, object] = {}
        if len(record_peaks):
            last_peak = record_peaks.iloc[-1]
            absolute_peak_time = float(record["start_time_s"].iloc[0]) + float(last_peak["peak_time_s"])
            peak_idx = int(np.argmin(np.abs(times - absolute_peak_time)))
            segment_id = int(record["segment_id"].iloc[peak_idx])
            same_segment = np.flatnonzero(
                (np.arange(len(record)) >= peak_idx) & (record["segment_id"].to_numpy() == segment_id)
            )
            below = same_segment[smooth[same_segment] < cutoff]
            end_idx = int(below[0]) if len(below) else int(same_segment[-1])
            active_tail = np.arange(peak_idx, end_idx + 1)
            active_tail = active_tail[smooth[active_tail] >= cutoff]
            record.loc[active_tail, "terminal_decline_status"] = "uncertain_terminal_tail"
            duration = float(times[active_tail[-1]] - times[peak_idx]) if len(active_tail) else 0.0
            peak_value = float(smooth[peak_idx])
            end_ratio = float(smooth[active_tail[-1]] / peak_value) if len(active_tail) and peak_value else float("nan")
            if len(active_tail) >= 3:
                fit = linregress(times[active_tail], smooth[active_tail])
                slope = float(fit.slope)
                r_squared = float(fit.rvalue**2)
            else:
                slope = r_squared = float("nan")
            qualifies = bool(
                duration >= args.minimum_duration
                and np.isfinite(slope) and slope < 0
                and np.isfinite(end_ratio) and end_ratio <= args.maximum_end_peak_ratio
            )
            if qualifies:
                record.loc[active_tail, "terminal_decline_status"] = "candidate_terminal_decline"
            event = {
                "last_candidate_peak_time_s": float(last_peak["peak_time_s"]),
                "terminal_peak_smoothed_rms": peak_value,
                "terminal_active_duration_s": duration,
                "terminal_end_to_peak_rms_ratio": end_ratio,
                "terminal_decline_slope_rms_per_s": slope,
                "terminal_decline_r_squared": r_squared,
                "candidate_decline": qualifies,
            }

        label_cols = ["subject_id", "repetition_id", "side", "start_time_s", "end_time_s",
                      "rms", "rms_smooth", "activity_cutoff", "terminal_decline_status"]
        rows.append(record[label_cols])
        summaries.append({
            "subject_id": subject_id,
            "record_id": key,
            "side": record["side"].iloc[0],
            "repetition_code": record["repetition_id"].iloc[0],
            "window_count": len(record),
            "activity_cutoff_rms": cutoff,
            "baseline_rms_estimate": baseline,
            "otsu_separability": separability,
            "minimum_duration_rule_s": args.minimum_duration,
            "maximum_end_peak_ratio_rule": args.maximum_end_peak_ratio,
            **event,
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    labeled = pd.concat(rows, ignore_index=True)
    summary = pd.DataFrame(summaries)
    labeled.to_csv(args.output_dir / "terminal_decline_window_labels.csv", index=False)
    summary.to_csv(args.output_dir / "terminal_decline_record_summary.csv", index=False)

    counts = summary["candidate_decline"].fillna(False).astype(bool).value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].hist(summary["terminal_active_duration_s"].dropna(), bins=18, color="#52796f", edgecolor="white")
    axes[0].axvline(args.minimum_duration, color="#b21f35", linestyle="--", label="Provisional duration rule")
    axes[0].set(xlabel="Active time after last candidate peak (s)", ylabel="Recordings", title="Terminal decline duration")
    axes[0].legend(frameon=False)
    axes[1].scatter(summary["terminal_active_duration_s"], summary["terminal_end_to_peak_rms_ratio"],
                    c=summary["candidate_decline"].fillna(False).map({True: "#b21f35", False: "#31688e"}), alpha=0.8)
    axes[1].axvline(args.minimum_duration, color="#555555", linestyle="--", linewidth=1)
    axes[1].axhline(args.maximum_end_peak_ratio, color="#555555", linestyle="--", linewidth=1)
    axes[1].set(xlabel="Active terminal duration (s)", ylabel="Final active RMS / peak RMS",
                title="Candidate rule (red = passes duration, decline and ratio)")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(args.output_dir / "terminal_decline_rule_summary.png", dpi=160)
    plt.close(fig)

    n = len(summary)
    n_candidate = int(summary["candidate_decline"].fillna(False).sum())
    report = [
        "PROVISIONAL TERMINAL-DECLINE LABEL AUDIT",
        "These are candidate signal-pattern labels, not verified physiological fatigue labels.",
        f"Records evaluated: {n}",
        f"Records meeting the provisional rule: {n_candidate}/{n} ({n_candidate / n:.1%})",
        f"Rule: terminal active decline after the last detected envelope peak; duration >= {args.minimum_duration:g} s; linear RMS slope < 0; final active RMS / peak RMS <= {args.maximum_end_peak_ratio:g}.",
        "Only active windows in a qualifying terminal tail receive candidate_terminal_decline.",
        "Earlier active windows are active_before_candidate; tails that do not pass the rule are uncertain_terminal_tail; inactive windows remain inactive_or_uncertain.",
        "A terminal decline can also reflect voluntary relaxation, stopping, electrode movement, or acquisition effects.",
        "The cutoff and peak are data-derived, and these labels must not be used as independent fatigue ground truth.",
        "No source data were modified. Summary/labels use pseudonymous subject identifiers.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("Provisional terminal-decline labels created; these are not validated fatigue labels.")
    print(f"Records meeting the rule: {n_candidate}/{n} ({n_candidate / n:.1%})")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
