"""Create separate exploratory fatigue-proxy labels without changing source data.

Strategies: active windows in the last third of recording time; active windows
inside candidate repetition cycles whose peak order is in the last third; and
the existing sustained terminal amplitude decline rule. None is independent
physiological ground truth.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--activity-labels", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_window_labels.csv")
    parser.add_argument("--repetitions", type=Path, default=root / "results" / "fatigue_candidates" / "candidate_repetitions.csv")
    parser.add_argument("--amplitude-labels", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_window_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "proposed_fatigue_labels")
    args = parser.parse_args()

    features = pd.read_csv(args.dataset)
    activity = pd.read_csv(args.activity_labels)
    reps = pd.read_csv(args.repetitions)
    amplitude = pd.read_csv(args.amplitude_labels)
    key_cols = ["subject_id", "repetition_id", "side"]
    for frame in (features, activity, amplitude):
        frame["record_key"] = frame[key_cols].astype(str).agg("_".join, axis=1)
        frame["time_key"] = frame["start_time_s"].round(6)
    active_states = {"active_before_candidate", "uncertain_terminal_tail", "candidate_terminal_decline"}
    activity_map = activity.set_index(["record_key", "time_key"])["terminal_decline_status"]
    rows: list[pd.DataFrame] = []

    for record_key, source in features.groupby("record_key", sort=True):
        record = source.sort_values("start_time_s").copy()
        status = pd.Series(
            [activity_map.get((record_key, t), "inactive_or_uncertain") for t in record["time_key"]],
            index=record.index,
        )
        active = status.isin(active_states).to_numpy()
        mid = (record["start_time_s"].to_numpy(float) + record["end_time_s"].to_numpy(float)) / 2
        start, end = float(record["start_time_s"].min()), float(record["end_time_s"].max())
        fraction = (mid - start) / max(end - start, np.finfo(float).eps)

        # Time thirds: active pre-final-third windows are comparison class;
        # active windows in the final third are the proposed fatigue proxy.
        time_target = np.full(len(record), -1, dtype=int)
        time_target[active & (fraction < 2 / 3)] = 0
        time_target[active & (fraction >= 2 / 3)] = 1
        rows.append(make_rows(record, record_key, "last_third_time", time_target, fraction))

        # Repetition thirds: use the existing automatic peak/cycle file. The
        # final stage assignment is inherited from assign_stages() in the
        # segmentation script; it remains a candidate repetition, not verified.
        rep_source = reps[reps["record_id"].astype(str).eq(record_key)].sort_values("candidate_rep")
        rep_target = np.full(len(record), -1, dtype=int)
        if len(rep_source) >= 3:
            centers = mid
            segments = record["segment_id"].to_numpy()
            for rep_row in rep_source.itertuples(index=False):
                chosen = (
                    (centers >= float(rep_row.cycle_start_s))
                    & (centers < float(rep_row.cycle_end_s))
                    & (segments == int(rep_row.segment_id))
                    & active
                )
                rep_target[chosen] = 1 if str(rep_row.stage_by_candidate_rep_order) == "late" else 0
        rows.append(make_rows(record, record_key, "last_third_repetitions", rep_target, fraction))

        # Amplitude decline: preserve the current rule exactly, including its
        # unknown/uncertain windows excluded from training.
        amp_status = amplitude[
            (amplitude["record_key"] == record_key)
        ].set_index("time_key")["terminal_decline_status"]
        aligned = np.array([amp_status.get(t, "inactive_or_uncertain") for t in record["time_key"]])
        amp_target = np.full(len(record), -1, dtype=int)
        if np.any(aligned == "candidate_terminal_decline"):
            amp_target[aligned == "active_before_candidate"] = 0
            amp_target[aligned == "candidate_terminal_decline"] = 1
        rows.append(make_rows(record, record_key, "terminal_amplitude_decline", amp_target, fraction))

    labels = pd.concat(rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_dir / "window_labels_all_strategies.csv", index=False)
    summary = labels[labels["target"] >= 0].groupby(["strategy", "record_key"], as_index=False).agg(
        windows=("target", "size"), positive_windows=("target", "sum"),
    )
    summary["positive_fraction"] = summary["positive_windows"] / summary["windows"]
    summary.to_csv(args.output_dir / "record_label_summary.csv", index=False)
    counts = labels[labels["target"] >= 0].groupby("strategy").agg(
        recordings=("record_key", "nunique"), participants=("subject_id", "nunique"),
        labeled_windows=("target", "size"), positive_windows=("target", "sum"),
    )
    report = [
        "EXPLORATORY FATIGUE-PROXY LABEL STRATEGIES",
        "All strategies are operational labels, not independent physiological fatigue truth.",
        "last_third_time: active windows in the final third of the recording are 1; active earlier windows are 0.",
        "last_third_repetitions: active windows in automatically segmented cycles assigned to the late candidate-repetition stage are 1; early/middle candidate cycles are 0. Windows outside candidate cycles are excluded.",
        "terminal_amplitude_decline: existing label rule copied as-is: active terminal-tail decline following the last detected peak, >=1 s, negative smoothed-RMS slope, final/peak ratio <=0.75, and only in qualifying recordings.",
        "Rows with target -1 are excluded/uncertain and are not negatives. Candidate repetition detection is not manually validated.",
        "The raw data and existing label/model outputs were not modified.",
        "Counts by strategy:",
        counts.to_string(),
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print(counts.to_string())
    print(f"Labels: {args.output_dir / 'window_labels_all_strategies.csv'}")


def make_rows(record: pd.DataFrame, record_key: str, strategy: str, target: np.ndarray, time_fraction: np.ndarray) -> pd.DataFrame:
    out = record[["subject_id", "repetition_id", "side", "start_time_s", "end_time_s", "segment_id"]].copy()
    out["record_key"] = record_key
    out["strategy"] = strategy
    out["target"] = target
    out["recording_time_fraction"] = time_fraction
    return out


if __name__ == "__main__":
    main()
