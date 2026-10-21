"""Repetition-level spectral/amplitude progression and an early-vs-late repetition model.

PRIMARY result: per-recording slope of mean MDF/MNF and peak RMS across the
automatically detected candidate repetitions (classic fatigue pattern: MDF/MNF
decrease while amplitude increases), summarized per participant.

SECONDARY result: leave-one-participant-out classifier of first-third vs
last-third candidate repetitions, using features expressed relative to the
same recording's first repetitions (label-free, offline normalization).

IMPORTANT: repetition boundaries come from unvalidated automatic peak
detection (manual_validation_required=True), MDF/MNF are band-pass/notch filtered, and the
target is repetition ORDER (time-based). These outputs show whether the
signal progresses consistently within a set; they are not validated
physiological fatigue labels.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

COLUMNS = {
    "mdf": "mean_mdf_hz_unfiltered",
    "mnf": "mean_mnf_hz_unfiltered",
    "peak_rms": "peak_smoothed_rms",
    "mean_rms": "mean_rms",
}
BASELINE_REPS = 3          # first repetitions used as the within-recording baseline
MIN_REPS_SLOPE = 4         # minimum repetitions to fit a slope
MIN_REPS_MODEL = 6         # minimum repetitions so first/last thirds are distinct
MAX_REPS_PLOT = 12
FEATURE_SETS = {
    "spectral_relative": ["mdf_rel", "mnf_rel"],
    "spectral_amplitude_relative": ["mdf_rel", "mnf_rel", "peak_rms_rel", "mean_rms_rel"],
}


def load(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"subject_id", "record_id", "candidate_rep", *COLUMNS.values()}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Candidate-repetition table is missing columns: {sorted(missing)}")
    frame = frame.rename(columns={v: k for k, v in COLUMNS.items()})
    return frame.sort_values(["record_id", "candidate_rep"]).reset_index(drop=True)


def slope_tables(reps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for record_id, group in reps.groupby("record_id", sort=True):
        if len(group) < MIN_REPS_SLOPE:
            continue
        x = np.arange(1, len(group) + 1, dtype=float)
        row = {"record_id": record_id, "subject_id": group["subject_id"].iloc[0], "repetitions": len(group)}
        for name in COLUMNS:
            row[f"{name}_slope_per_rep"] = float(linregress(x, group[name].to_numpy(float)).slope)
        rows.append(row)
    records = pd.DataFrame(rows)
    summary = []
    for subject, group in records.groupby("subject_id", sort=True):
        row = {"subject_id": subject, "records": len(group)}
        for name in COLUMNS:
            col = f"{name}_slope_per_rep"
            row[f"median_{col}"] = float(group[col].median())
            row[f"share_records_negative_{name}"] = float((group[col] < 0).mean())
        summary.append(row)
    return records, pd.DataFrame(summary)


def add_relative_features(reps: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in reps.groupby("record_id", sort=True):
        if len(group) < MIN_REPS_MODEL:
            continue
        group = group.copy()
        baseline = group.iloc[:BASELINE_REPS][list(COLUMNS)].median()
        for name in COLUMNS:
            group[f"{name}_rel"] = group[name] - baseline[name]
        third = len(group) // 3
        group["target"] = -1
        group.iloc[:third, group.columns.get_loc("target")] = 0
        group.iloc[-third:, group.columns.get_loc("target")] = 1
        parts.append(group)
    if not parts:
        raise RuntimeError("No recordings with enough candidate repetitions")
    data = pd.concat(parts, ignore_index=True)
    return data[data["target"] >= 0].reset_index(drop=True)


def run_models(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=5, random_state=42, n_jobs=-1,
        ),
    }
    y = data["target"].to_numpy(int)
    groups = data["subject_id"].astype(str).to_numpy()
    summary_rows, subject_rows, prediction_frames = [], [], []
    for set_name, columns in FEATURE_SETS.items():
        X = data[columns].to_numpy(float)
        for model_name, model in models.items():
            prob = np.full(len(data), np.nan)
            for train, test in LeaveOneGroupOut().split(X, y, groups):
                fitted = clone(model).fit(X[train], y[train])
                prob[test] = fitted.predict_proba(X[test])[:, 1]
            per_subject = []
            for subject in np.unique(groups):
                mask = groups == subject
                if len(np.unique(y[mask])) < 2:
                    continue
                auc = roc_auc_score(y[mask], prob[mask])
                bal = balanced_accuracy_score(y[mask], prob[mask] >= 0.5)
                per_subject.append(auc)
                subject_rows.append({
                    "feature_set": set_name, "model": model_name, "subject_id": subject,
                    "repetitions": int(mask.sum()), "roc_auc": auc, "balanced_accuracy": bal,
                })
            sub = pd.DataFrame([r for r in subject_rows if r["feature_set"] == set_name and r["model"] == model_name])
            summary_rows.append({
                "feature_set": set_name, "model": model_name,
                "participants": int(sub["subject_id"].nunique()),
                "recordings": int(data["record_id"].nunique()), "repetitions": len(data),
                "pooled_roc_auc": float(roc_auc_score(y, prob)),
                "subject_mean_roc_auc": float(sub["roc_auc"].mean()),
                "subject_sd_roc_auc": float(sub["roc_auc"].std(ddof=1)),
                "subject_min_roc_auc": float(sub["roc_auc"].min()),
                "subject_mean_balanced_accuracy": float(sub["balanced_accuracy"].mean()),
            })
            out = data[["subject_id", "record_id", "candidate_rep", "target"]].copy()
            out["feature_set"], out["model"], out["probability_late"] = set_name, model_name, prob
            prediction_frames.append(out)
    return pd.DataFrame(summary_rows), pd.DataFrame(subject_rows), pd.concat(prediction_frames, ignore_index=True)


def progression_figure(reps: pd.DataFrame, path: Path) -> None:
    prepared = []
    for _, group in reps.groupby("record_id", sort=True):
        if len(group) < MIN_REPS_SLOPE:
            continue
        group = group.copy()
        baseline = group.iloc[:BASELINE_REPS][["mdf", "peak_rms"]].median()
        group["mdf_rel"] = group["mdf"] - baseline["mdf"]
        group["peak_rms_rel"] = group["peak_rms"] - baseline["peak_rms"]
        group["rep"] = np.arange(1, len(group) + 1)
        prepared.append(group[group["rep"] <= MAX_REPS_PLOT])
    frame = pd.concat(prepared, ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, column, label in [
        (axes[0], "mdf_rel", "MDF change vs first 3 repetitions (Hz)"),
        (axes[1], "peak_rms_rel", "Peak RMS change vs first 3 repetitions"),
    ]:
        per_subject = frame.groupby(["subject_id", "rep"], as_index=False)[column].mean()
        for _, sub in per_subject.groupby("subject_id"):
            ax.plot(sub["rep"], sub[column], color="#7a8b99", alpha=0.35, linewidth=0.9)
        wide = per_subject.pivot(index="rep", columns="subject_id", values=column)
        wide = wide[wide.notna().sum(axis=1) >= 5]
        mean, sd = wide.mean(axis=1), wide.std(axis=1, ddof=1)
        ax.plot(mean.index, mean, color="#b21f35", linewidth=2.2, label="Mean across participants")
        ax.fill_between(mean.index, mean - sd, mean + sd, color="#b21f35", alpha=0.15, linewidth=0)
        ax.axhline(0, color="#333333", linewidth=0.8, linestyle=":")
        ax.set(xlabel="Candidate repetition number", ylabel=label)
        ax.grid(alpha=0.2)
    axes[0].legend(frameon=False)
    fig.suptitle(
        "Within-recording progression across candidate repetitions (unvalidated segmentation, band-pass/notch filtered spectrum)",
        fontsize=10,
    )
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, default=root / "results" / "fatigue_candidates" / "candidate_repetitions.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "repetition_progression")
    args = parser.parse_args()

    reps = load(args.candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records, participants = slope_tables(reps)
    records.to_csv(args.output_dir / "record_slopes.csv", index=False)
    participants.to_csv(args.output_dir / "participant_slopes.csv", index=False)
    progression_figure(reps, args.output_dir / "repetition_progression.png")

    data = add_relative_features(reps)
    summary, subject_metrics, predictions = run_models(data)
    summary.to_csv(args.output_dir / "model_metrics.csv", index=False)
    subject_metrics.to_csv(args.output_dir / "model_metrics_by_subject.csv", index=False)
    predictions.to_csv(args.output_dir / "oof_predictions.csv", index=False)

    neg_mdf = int((records["mdf_slope_per_rep"] < 0).sum())
    neg_mnf = int((records["mnf_slope_per_rep"] < 0).sum())
    pos_rms = int((records["peak_rms_slope_per_rep"] > 0).sum())
    subj_neg = int((participants["median_mdf_slope_per_rep"] < 0).sum())
    report = [
        "REPETITION-LEVEL PROGRESSION AND EARLY-VS-LATE MODEL",
        "Repetitions are automatic candidates (not manually validated); MDF/MNF are band-pass/notch filtered. Not validated fatigue.",
        f"Recordings with >= {MIN_REPS_SLOPE} repetitions: {len(records)}; participants: {records['subject_id'].nunique()}",
        f"MDF slope < 0 in {neg_mdf}/{len(records)} recordings; MNF slope < 0 in {neg_mnf}/{len(records)}; peak-RMS slope > 0 in {pos_rms}/{len(records)}.",
        f"Participants whose median MDF slope is negative: {subj_neg}/{len(participants)}",
        f"Median MDF slope across recordings: {records['mdf_slope_per_rep'].median():+.2f} Hz per repetition.",
        "",
        f"Model: first-third (0) vs last-third (1) repetitions; middle third excluded; recordings with >= {MIN_REPS_MODEL} repetitions.",
        f"Features are relative to the median of each recording's first {BASELINE_REPS} repetitions (offline, label-free).",
        "Validation: leave-one-participant-out; metrics are per held-out participant (mean, SD, minimum).",
        "The target is repetition order, i.e. time-based; it shows consistent within-set progression, not independent fatigue ground truth.",
        summary.round(3).to_string(index=False),
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report[3:6]))
    print(summary.round(3).to_string(index=False))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
