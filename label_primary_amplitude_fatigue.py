"""Create fatigue labels where only a sustained terminal RMS decline is positive.

Recording-time thirds and automatically detected repetition thirds are retained
as descriptive approximations, never as fatigue targets or training labels.
The rule is inherited from label_terminal_decline.py and remains an operational
proxy, not independently verified physiological fatigue.
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
    parser.add_argument("--amplitude-labels", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_window_labels.csv")
    parser.add_argument("--record-summary", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_record_summary.csv")
    parser.add_argument("--repetitions", type=Path, default=root / "results" / "fatigue_candidates" / "candidate_repetitions.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "primary_amplitude_fatigue_labels")
    args = parser.parse_args()

    features = pd.read_csv(args.dataset)
    amp = pd.read_csv(args.amplitude_labels)
    rec_summary = pd.read_csv(args.record_summary)
    reps = pd.read_csv(args.repetitions)
    key_cols = ["subject_id", "repetition_id", "side"]
    features["record_key"] = features[key_cols].astype(str).agg("_".join, axis=1)
    amp["record_key"] = amp[["subject_id", "repetition_id", "side"]].astype(str).agg("_".join, axis=1)
    amp["time_key"] = amp["start_time_s"].round(6)
    features["time_key"] = features["start_time_s"].round(6)
    qualifying = set(rec_summary.loc[rec_summary["candidate_decline"].astype(bool), "record_id"].astype(str))
    rep_map: dict[str, list[dict[str, object]]] = {}
    for key, group in reps.groupby("record_id", sort=False):
        rep_map[str(key)] = group.to_dict("records")

    rows: list[pd.DataFrame] = []
    for key, rec in features.groupby("record_key", sort=True):
        if key not in qualifying:
            continue
        rec = rec.sort_values("start_time_s").copy()
        status = amp.loc[amp["record_key"].eq(key)].set_index("time_key")["terminal_decline_status"]
        states = rec["time_key"].map(status).fillna("inactive_or_uncertain").to_numpy()
        # Preserve the original rule exactly: only active pre-decline and
        # candidate decline windows enter; all other windows remain excluded.
        y = np.full(len(rec), -1, dtype=int)
        y[states == "active_before_candidate"] = 0
        y[states == "candidate_terminal_decline"] = 1
        if not np.any(y == 1) or not np.any(y == 0):
            continue

        mid = (rec["start_time_s"].to_numpy(float) + rec["end_time_s"].to_numpy(float)) / 2
        lo, hi = float(rec["start_time_s"].min()), float(rec["end_time_s"].max())
        frac = (mid - lo) / max(hi - lo, np.finfo(float).eps)
        thirds = np.where(frac < 1 / 3, "early", np.where(frac < 2 / 3, "middle", "late"))
        rep_stage = np.full(len(rec), "unassigned", dtype=object)
        centers = mid
        segments = rec["segment_id"].to_numpy(int)
        for rep in rep_map.get(key, []):
            mask = (centers >= float(rep["cycle_start_s"])) & (centers < float(rep["cycle_end_s"]))
            mask &= segments == int(rep["segment_id"])
            rep_stage[mask] = str(rep.get("stage_by_candidate_rep_order", "unassigned"))

        out = rec[["subject_id", "repetition_id", "side", "start_time_s", "end_time_s", "segment_id"]].copy()
        out["record_key"] = key
        out["fatigue_target"] = y
        out["fatigue_status"] = np.where(y == 1, "amplitude_decline_fatigue_proxy", np.where(y == 0, "active_pre_decline", "excluded_uncertain_or_inactive"))
        out["recording_third_approximation"] = thirds
        out["candidate_repetition_third_approximation"] = rep_stage
        out["recording_time_fraction"] = frac
        rows.append(out)

    if not rows:
        raise RuntimeError("No qualifying recordings with both active-pre-decline and decline windows were found.")
    labels = pd.concat(rows, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(args.output_dir / "primary_amplitude_fatigue_labels.csv", index=False)
    used = labels[labels["fatigue_target"] >= 0]
    counts = used.groupby("fatigue_target").agg(windows=("fatigue_target", "size"), participants=("subject_id", "nunique"), recordings=("record_key", "nunique"))
    cross = used.groupby(["fatigue_target", "recording_third_approximation"], observed=True).size().rename("windows").reset_index()
    rep_cross = used.groupby(["fatigue_target", "candidate_repetition_third_approximation"], observed=True).size().rename("windows").reset_index()
    counts.to_csv(args.output_dir / "class_counts.csv")
    cross.to_csv(args.output_dir / "fatigue_by_time_third.csv", index=False)
    rep_cross.to_csv(args.output_dir / "fatigue_by_candidate_repetition_third.csv", index=False)
    readme = [
        "PRIMARY OPERATIONAL LABEL: SUSTAINED TERMINAL AMPLITUDE DECLINE",
        "Only the inherited terminal-amplitude-decline rule defines fatigue_target=1. Active pre-decline windows in the same qualifying recording define target=0. Uncertain/inactive windows and recordings without a qualifying decline are excluded, not labeled non-fatigue.",
        "The inherited candidate rule requires an active terminal tail of at least 1 second, negative smoothed-RMS slope, and final/peak RMS ratio <=0.75. These are provisional operational criteria, not physiological ground truth.",
        "Last thirds of recording time and candidate repetitions are descriptive approximation columns only; they do not determine fatigue_target and are not model targets.",
        "Amplitude-defined target shares signal amplitude with amplitude-derived model features. Spectral-only results are therefore the primary less-circular prediction analysis; amplitude/combined feature sets are exploratory and potentially circular.",
        "Original data and previous labels/results are not modified.",
        "Class counts (target 0=active before decline; target 1=terminal decline):",
        counts.to_string(),
    ]
    (args.output_dir / "README.txt").write_text("\n".join(readme), encoding="utf-8")
    print(counts.to_string())
    print(f"Saved labels and descriptive thirds to {args.output_dir}")


if __name__ == "__main__":
    main()
