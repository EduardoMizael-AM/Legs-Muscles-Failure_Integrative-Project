"""Compare time-thirds and envelope-peak stage definitions.

All reported model metrics measure recovery of constructed proxy labels, not
accuracy against independent physiological fatigue ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
STAGES = ["early", "middle", "late"]
FEATURES = ["rms", "mav", "variance", "waveform_length"]
TARGETS = {
    "time_thirds": "stage_by_time_thirds",
    "peak_thirds": "stage_by_candidate_rep_order",
}


def attach_labels(features: pd.DataFrame, cycles: pd.DataFrame) -> pd.DataFrame:
    required_features = {
        "subject_id", "record_id", "repetition_id", "side", "start_time_s", "end_time_s", *FEATURES,
    }
    missing = required_features.difference(features.columns)
    if missing:
        raise ValueError(f"Feature table missing columns: {sorted(missing)}")
    required_cycles = {
        "subject_id", "repetition_code", "side", "candidate_rep",
        "stage_by_candidate_rep_order", "cycle_start_s", "cycle_end_s",
    }
    missing = required_cycles.difference(cycles.columns)
    if missing:
        raise ValueError(f"Candidate-cycle table missing columns: {sorted(missing)}")

    frame = features.copy()
    frame = frame.sort_values(["subject_id", "record_id", "start_time_s"]).reset_index(drop=True)
    rank = frame.groupby(["subject_id", "record_id"], sort=False).cumcount()
    count = frame.groupby(["subject_id", "record_id"], sort=False)["record_id"].transform("size")
    position = (rank + 0.5) / count
    frame[TARGETS["time_thirds"]] = np.select(
        [position < 1 / 3, position < 2 / 3], ["early", "middle"], default="late"
    )
    frame[TARGETS["peak_thirds"]] = pd.NA
    frame["elapsed_mid_s"] = (
        (frame["start_time_s"] + frame["end_time_s"]) / 2
        - frame.groupby(["subject_id", "record_id"], sort=False)["start_time_s"].transform("min")
    )

    cycle_groups = {
        key: group.sort_values("cycle_start_s")
        for key, group in cycles.groupby(["subject_id", "repetition_code", "side"], sort=False)
    }
    for (subject, rep, side, record_id), rows in frame.groupby(
        ["subject_id", "repetition_id", "side", "record_id"], sort=False
    ):
        cycle_set = cycle_groups.get((subject, rep, side))
        if cycle_set is None:
            continue
        row_indices = rows.index.to_numpy()
        elapsed = rows["elapsed_mid_s"].to_numpy(float)
        labels = np.full(len(rows), None, dtype=object)
        for _, cycle in cycle_set.iterrows():
            within = (elapsed >= float(cycle["cycle_start_s"])) & (elapsed < float(cycle["cycle_end_s"]))
            labels[within] = str(cycle["stage_by_candidate_rep_order"])
        frame.loc[row_indices, TARGETS["peak_thirds"]] = labels

    frame[TARGETS["peak_thirds"]] = frame[TARGETS["peak_thirds"]].astype("string")
    return frame


def stage_feature_summary(data: pd.DataFrame, target_name: str, target_column: str) -> pd.DataFrame:
    selected = data[data[target_column].isin(STAGES)].copy()
    # First average windows within each recording/phase; then average records
    # within participant, so longer files do not dominate the group means.
    per_record = selected.groupby(
        ["subject_id", "record_id", "side", target_column], observed=True, as_index=False
    )[FEATURES].mean()
    per_subject = per_record.groupby(
        ["subject_id", "side", target_column], observed=True, as_index=False
    )[FEATURES].mean()
    summary = per_subject.groupby(["side", target_column], observed=True).agg(
        participant_count=("subject_id", "nunique"),
        **{f"mean_{feature}": (feature, "mean") for feature in FEATURES},
        **{f"sd_{feature}": (feature, "std") for feature in FEATURES},
    ).reset_index().rename(columns={target_column: "stage"})
    summary["label_method"] = target_name
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument(
        "--candidate-cycles", type=Path,
        default=ROOT / "results" / "fatigue_candidates" / "candidate_repetitions.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "method_comparison")
    args = parser.parse_args()

    features = pd.read_csv(args.dataset)
    cycles = pd.read_csv(args.candidate_cycles)
    data = attach_labels(features, cycles)
    matched = data[data[TARGETS["peak_thirds"]].notna()].copy()
    if matched.empty:
        raise ValueError("No feature windows could be matched to candidate cycles")

    agreement = float(np.mean(matched[TARGETS["time_thirds"]] == matched[TARGETS["peak_thirds"]]))
    kappa = float(cohen_kappa_score(
        matched[TARGETS["time_thirds"]], matched[TARGETS["peak_thirds"]], labels=STAGES
    ))
    confusion = confusion_matrix(
        matched[TARGETS["time_thirds"]], matched[TARGETS["peak_thirds"]], labels=STAGES
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched[["subject_id", "repetition_id", "side", "start_time_s", "end_time_s",
             TARGETS["time_thirds"], TARGETS["peak_thirds"]]].to_csv(
        args.output_dir / "matched_window_stage_labels.csv", index=False
    )

    summaries = pd.concat([
        stage_feature_summary(matched, label_name, target_column)
        for label_name, target_column in TARGETS.items()
    ], ignore_index=True)
    summaries.to_csv(args.output_dir / "stage_feature_summary_by_method.csv", index=False)

    X = matched[FEATURES].to_numpy(float)
    groups = matched["subject_id"].astype(str).to_numpy()
    cv = LeaveOneGroupOut()
    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }
    metrics = []
    prediction_outputs = []
    for label_name, target_column in TARGETS.items():
        y = matched[target_column].astype(str).to_numpy()
        for model_name, model in models.items():
            predicted = cross_val_predict(model, X, y, groups=groups, cv=cv, method="predict", n_jobs=1)
            metrics.append({
                "label_method": label_name,
                "model": model_name,
                "window_count": len(y),
                "participant_count": matched["subject_id"].nunique(),
                "balanced_accuracy_vs_proxy": balanced_accuracy_score(y, predicted),
                "macro_f1_vs_proxy": f1_score(y, predicted, labels=STAGES, average="macro", zero_division=0),
                "macro_precision_vs_proxy": precision_score(y, predicted, labels=STAGES, average="macro", zero_division=0),
                "macro_recall_vs_proxy": recall_score(y, predicted, labels=STAGES, average="macro", zero_division=0),
            })
            predictions = matched[["subject_id", "repetition_id", "side", "start_time_s", "end_time_s"]].copy()
            predictions["label_method"] = label_name
            predictions["model"] = model_name
            predictions["proxy_true_stage"] = y
            predictions["proxy_predicted_stage"] = predicted
            prediction_outputs.append(predictions)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(args.output_dir / "proxy_model_metrics.csv", index=False)
    pd.concat(prediction_outputs, ignore_index=True).to_csv(
        args.output_dir / "proxy_model_oof_predictions.csv", index=False
    )

    report = [
        "COMPARISON OF TWO OPERATIONAL STAGE PROXIES",
        "This is not a measurement of accuracy against physiological fatigue ground truth.",
        f"All corrected windows: {len(data)}",
        f"Windows inside a detected candidate cycle (comparison coverage): {len(matched)} ({len(matched) / len(data):.1%})",
        f"Participants: {matched['subject_id'].nunique()}",
        f"Window-level agreement between time-thirds and candidate-peak thirds: {agreement:.3f}",
        f"Cohen's kappa between the two constructed labelings: {kappa:.3f}",
        "Confusion matrix (rows=time thirds, columns=peak-order thirds; early, middle, late):",
        str(confusion),
        "",
        "LOSO model metrics below assess reconstruction of each proxy label only.",
        "They must not be called fatigue accuracy, sensitivity, or precision against a true fatigue outcome.",
        metrics_frame.to_string(index=False),
        "",
        "The same four time-domain features are used for both targets; RMS also contributes to candidate-peak detection.",
        "Thus peak-label model scores are internal consistency checks and may be partly circular.",
        "MDF/MNF remain excluded from model predictors and are only summarized as exploratory outcomes.",
    ]
    (args.output_dir / "method_comparison_report.txt").write_text("\n".join(report), encoding="utf-8")
    print("Proxy-method comparison complete; no independent fatigue accuracy is available.")
    print(f"Matched windows: {len(matched)}/{len(data)} ({len(matched) / len(data):.1%})")
    print(f"Agreement: {agreement:.3f}; Cohen's kappa: {kappa:.3f}")
    print(metrics_frame.to_string(index=False))
    print(f"Report: {args.output_dir / 'method_comparison_report.txt'}")


if __name__ == "__main__":
    main()
