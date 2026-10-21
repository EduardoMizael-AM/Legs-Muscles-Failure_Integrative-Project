"""Train LOSO models to identify the provisional terminal-decline label.

The target is the operational label from label_terminal_decline.py, not
independent physiological fatigue ground truth. Only recordings with a
qualifying terminal-decline segment are included; other recordings are not
silently relabeled as fatigue-negative.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "amplitude_derived": ["mav", "variance", "waveform_length"],
    "unfiltered_spectral": ["mdf_hz_filtered", "mnf_hz_filtered"],
    "combined": ["mav", "variance", "waveform_length", "mdf_hz_filtered", "mnf_hz_filtered"],
}
INNER_RECALL_TARGETS = [0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01]


def choose_thresholds(y: np.ndarray, probabilities: np.ndarray, recall_targets: list[float]) -> dict[float, float]:
    """Choose precision-maximizing thresholds from inner grouped-CV PR data."""
    precision, recall, thresholds = precision_recall_curve(y, probabilities)
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(precision), where=(precision + recall) > 0)
    chosen: dict[float, float] = {}
    for target in recall_targets:
        feasible = np.flatnonzero(recall >= target)
        if len(feasible):
            # Prefer precision, then F1, then a higher threshold.
            best = max(feasible, key=lambda i: (precision[i], f1[i], thresholds[i]))
        else:
            best = max(range(len(thresholds)), key=lambda i: (f1[i], precision[i], thresholds[i]))
        chosen[target] = float(thresholds[best])
    return chosen


def longest_contiguous_run(mask: np.ndarray, times: np.ndarray, segments: np.ndarray) -> int:
    """Longest true run, resetting across timestamp gaps or segment changes."""
    order = np.argsort(times)
    times = times[order]
    segments = segments[order]
    mask = mask[order]
    deltas = np.diff(times)
    expected_step = float(np.median(deltas[(deltas > 0) & (deltas <= 0.25)])) if np.any((deltas > 0) & (deltas <= 0.25)) else 0.125
    run = maximum = 0
    for i, selected in enumerate(mask):
        contiguous = (
            i > 0
            and segments[i] == segments[i - 1]
            and 0 < times[i] - times[i - 1] <= expected_step * 1.5
        )
        if selected:
            run = run + 1 if contiguous and mask[i - 1] else 1
            maximum = max(maximum, run)
        else:
            run = 0
    return maximum


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--labels", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_window_labels.csv")
    parser.add_argument("--record-summary", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_record_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "terminal_decline_model")
    args = parser.parse_args()

    features = pd.read_csv(args.dataset)
    labels = pd.read_csv(args.labels)
    record_summary = pd.read_csv(args.record_summary)
    qualifying = set(record_summary.loc[record_summary["candidate_decline"].fillna(False), "record_id"].astype(str))
    if not qualifying:
        raise ValueError("No recordings meet the candidate terminal-decline rule; regenerate labels first")

    # Construct pseudonymous recording keys; exclude raw source paths and filenames.
    for frame in (features, labels):
        frame["record_key"] = (
            frame["subject_id"].astype(str) + "_" + frame["repetition_id"].astype(str) + "_" + frame["side"].astype(str)
        )
        frame["time_key"] = frame["start_time_s"].round(6)
    selected_features = features[features["record_key"].isin(qualifying)].copy()
    selected_labels = labels[labels["record_key"].isin(qualifying)].copy()
    merged = selected_features.merge(
        selected_labels[["record_key", "time_key", "terminal_decline_status"]],
        on=["record_key", "time_key"], how="inner", validate="one_to_one",
    )
    merged = merged[merged["terminal_decline_status"].isin(
        ["active_before_candidate", "candidate_terminal_decline"]
    )].copy()
    merged["target_candidate_fatigue"] = (merged["terminal_decline_status"] == "candidate_terminal_decline").astype(int)
    if merged["subject_id"].nunique() < 3 or merged["target_candidate_fatigue"].nunique() != 2:
        raise ValueError("Need both classes and at least three participants for subject-wise validation")

    groups = merged["subject_id"].astype(str).to_numpy()
    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }
    metrics: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    subject_metrics: list[dict[str, object]] = []
    event_metrics: list[dict[str, object]] = []

    for feature_set_name, columns in FEATURE_SETS.items():
        clean = merged.dropna(subset=columns).copy()
        X = clean[columns].to_numpy(float)
        y = clean["target_candidate_fatigue"].to_numpy(int)
        groups_clean = clean["subject_id"].astype(str).to_numpy()
        if len(np.unique(groups_clean)) < 3 or len(np.unique(y)) != 2:
            continue
        for model_name, model in models.items():
            # Nested group validation: the probability threshold is selected
            # using only other participants inside each outer training fold.
            prediction_parts = {target: [] for target in INNER_RECALL_TARGETS}
            probability_parts = []
            threshold_parts = {target: [] for target in INNER_RECALL_TARGETS}
            logo = LeaveOneGroupOut()
            for fold_index, (train_idx, test_idx) in enumerate(logo.split(X, y, groups_clean), start=1):
                X_train, y_train = X[train_idx], y[train_idx]
                train_groups = groups_clean[train_idx]
                inner_splits = min(3, len(np.unique(train_groups)))
                inner_cv = GroupKFold(n_splits=inner_splits)
                inner_probabilities = cross_val_predict(
                    clone(model), X_train, y_train, groups=train_groups,
                    cv=inner_cv, method="predict_proba", n_jobs=1,
                )[:, 1]
                thresholds = choose_thresholds(y_train, inner_probabilities, INNER_RECALL_TARGETS)
                fitted = clone(model).fit(X_train, y_train)
                test_probability = fitted.predict_proba(X[test_idx])[:, 1]
                probability_parts.append(test_probability)
                for target, threshold in thresholds.items():
                    prediction_parts[target].append(test_probability >= threshold)
                    threshold_parts[target].append(np.full(len(test_idx), threshold, dtype=float))
                print(f"{feature_set_name} / {model_name}: outer participant {fold_index}/{len(np.unique(groups_clean))}")
            probability = np.concatenate(probability_parts)
            prediction_default = (probability >= 0.50).astype(int)
            keys = clean["record_key"].astype(str).to_numpy()
            for recall_target in INNER_RECALL_TARGETS:
                prediction = np.concatenate(prediction_parts[recall_target]).astype(int)
                selected_threshold = np.concatenate(threshold_parts[recall_target])
                metrics.append({
                "feature_set": feature_set_name,
                "model": model_name,
                "minimum_recall_constraint_inner": recall_target,
                "participants": clean["subject_id"].nunique(),
                "qualifying_recordings": clean["record_key"].nunique(),
                "windows": len(clean),
                "positive_candidate_windows": int(y.sum()),
                "mean_training_selected_threshold": float(np.mean(selected_threshold)),
                "median_training_selected_threshold": float(np.median(selected_threshold)),
                "balanced_accuracy_proxy": balanced_accuracy_score(y, prediction),
                "f1_candidate_fatigue_proxy": f1_score(y, prediction, zero_division=0),
                "precision_candidate_fatigue_proxy": precision_score(y, prediction, zero_division=0),
                "recall_candidate_fatigue_proxy": recall_score(y, prediction, zero_division=0),
                "average_precision_proxy": average_precision_score(y, probability),
                "roc_auc_proxy": roc_auc_score(y, probability),
                "balanced_accuracy_at_fixed_0_5": balanced_accuracy_score(y, prediction_default),
                "f1_at_fixed_0_5": f1_score(y, prediction_default, zero_division=0),
                "precision_at_fixed_0_5": precision_score(y, prediction_default, zero_division=0),
                "recall_at_fixed_0_5": recall_score(y, prediction_default, zero_division=0),
                })
                result = clean[["subject_id", "repetition_id", "side", "start_time_s", "end_time_s", "record_key"]].copy()
                result["feature_set"] = feature_set_name
                result["model"] = model_name
                result["minimum_recall_constraint_inner"] = recall_target
                result["proxy_true"] = y
                result["proxy_predicted"] = prediction
                result["proxy_predicted_at_fixed_0_5"] = prediction_default
                result["proxy_probability"] = probability
                result["threshold_selected_on_training_participants"] = selected_threshold
                prediction_frames.append(result)
                for subject in sorted(clean["subject_id"].unique()):
                    mask = groups_clean == subject
                    subject_metrics.append({
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "minimum_recall_constraint_inner": recall_target,
                    "subject_id": subject,
                    "windows": int(mask.sum()),
                    "positive_windows": int(y[mask].sum()),
                    "balanced_accuracy_proxy": balanced_accuracy_score(y[mask], prediction[mask]),
                    "f1_candidate_fatigue_proxy": f1_score(y[mask], prediction[mask], zero_division=0),
                    "precision_candidate_fatigue_proxy": precision_score(y[mask], prediction[mask], zero_division=0),
                    "recall_candidate_fatigue_proxy": recall_score(y[mask], prediction[mask], zero_division=0),
                    })
                record_predictions = pd.DataFrame({
                "record_key": keys,
                "subject_id": clean["subject_id"].astype(str).to_numpy(),
                "true": y,
                "predicted": prediction,
                "start_time_s": clean["start_time_s"].to_numpy(float),
                "segment_id": clean["segment_id"].to_numpy(int),
                })
                for record_key, record_rows in record_predictions.groupby("record_key", sort=True):
                    true_tail = record_rows["true"].to_numpy() == 1
                    pred = record_rows["predicted"].to_numpy() == 1
                    times = record_rows["start_time_s"].to_numpy(float)
                    segments = record_rows["segment_id"].to_numpy(int)
                    candidate_run = longest_contiguous_run(pred & true_tail, times, segments)
                    predecline_run = longest_contiguous_run(pred & ~true_tail, times, segments)
                    for required_run in (1, 4, 8):
                        event_metrics.append({
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "minimum_recall_constraint_inner": recall_target,
                            "record_key": record_key,
                            "subject_id": record_rows["subject_id"].iloc[0],
                            "required_consecutive_windows": required_run,
                            "candidate_event_detected": candidate_run >= required_run,
                            "candidate_tail_window_recall": float(np.mean(pred[true_tail])) if true_tail.any() else np.nan,
                            "predecline_false_alarm": predecline_run >= required_run,
                        })

    if not metrics:
        raise ValueError("No model results produced; check labels, feature values and participant coverage")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames, ignore_index=True).to_csv(
        args.output_dir / "terminal_decline_oof_predictions.csv", index=False
    )
    pd.DataFrame(subject_metrics).to_csv(args.output_dir / "terminal_decline_subject_metrics.csv", index=False)
    event_frame = pd.DataFrame(event_metrics)
    event_frame.to_csv(args.output_dir / "terminal_decline_record_event_metrics.csv", index=False)
    event_summary = event_frame[event_frame["required_consecutive_windows"].eq(1)].groupby(
        ["feature_set", "model", "minimum_recall_constraint_inner"], as_index=False
    ).agg(
        qualifying_records_for_event_metrics=("record_key", "nunique"),
        candidate_event_hit_rate=("candidate_event_detected", "mean"),
        predecline_false_alarm_record_rate=("predecline_false_alarm", "mean"),
        mean_candidate_tail_window_recall=("candidate_tail_window_recall", "mean"),
    )
    metrics_frame = pd.DataFrame(metrics).merge(
        event_summary, on=["feature_set", "model", "minimum_recall_constraint_inner"], how="left"
    )
    metrics_frame.to_csv(args.output_dir / "terminal_decline_model_metrics.csv", index=False)
    persistence_summary = event_frame.groupby(
        ["feature_set", "model", "minimum_recall_constraint_inner", "required_consecutive_windows"], as_index=False
    ).agg(
        qualifying_recordings=("record_key", "nunique"),
        candidate_event_hit_rate=("candidate_event_detected", "mean"),
        predecline_false_alarm_record_rate=("predecline_false_alarm", "mean"),
    )
    persistence_summary.to_csv(args.output_dir / "terminal_decline_persistence_summary.csv", index=False)

    report = [
        "MODEL FOR THE PROVISIONAL TERMINAL-DECLINE LABEL",
        "This is proxy-label prediction, not independently validated physiological fatigue detection.",
        f"Primary fatigue-proxy rule: active terminal decline >= {float(record_summary['minimum_duration_rule_s'].iloc[0]):g} s, negative slope, end/peak RMS <= 0.75.",
        f"Qualifying recordings only: {len(qualifying)}; participants represented: {merged['subject_id'].nunique()}",
        f"Active pre-decline windows: {int((merged['target_candidate_fatigue'] == 0).sum())}",
        f"Candidate terminal-decline windows: {int(merged['target_candidate_fatigue'].sum())}",
        "Uncertain tails, inactive windows, and recordings without a qualifying decline were excluded, not assigned negative labels.",
        "Validation: outer leave-one-participant-out. Probability threshold is selected using inner grouped CV on training participants only, maximizing precision subject to one of several minimum recall constraints (0.50, 0.30, 0.20, 0.10, 0.05, 0.02, 0.01; or F1 if unattainable).",
        "The result table includes each recall-constrained operating point and fixed-threshold 0.50 metrics. Record-event outputs report candidate-event hits and pre-decline false-alarm recordings.",
        "Persistence summaries require 1, 4, or 8 consecutive predicted windows (approximately 0, 0.375, or 0.875 s at the 0.125-s step) before counting an event hit or pre-decline alarm.",
        "Overlapping 250-ms windows with 125-ms step are temporally correlated.",
        "Feature sets: amplitude-derived features omit RMS itself but remain correlated with the RMS-based label rule; spectral MDF/MNF are raw and unfiltered, so their physiological interpretation is unverified.",
        "These outputs do not estimate participant-level fatigue detection or generalization to recordings without a candidate terminal decline.",
        "",
        metrics_frame.to_string(index=False),
        "",
        "EVENT PERSISTENCE SUMMARY",
        persistence_summary.to_string(index=False),
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("Nested LOSO models fitted for the operational terminal-decline proxy; results are not independent fatigue accuracy.")
    print(metrics_frame.to_string(index=False))
    print(f"Report: {args.output_dir / 'README.txt'}")


if __name__ == "__main__":
    main()
