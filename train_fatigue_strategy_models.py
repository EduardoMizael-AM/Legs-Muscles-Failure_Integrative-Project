"""Compare subject-held-out models for the exploratory fatigue proxies."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "amplitude_derived": ["mav", "variance", "waveform_length"],
    "unfiltered_spectral": ["mdf_hz_filtered", "mnf_hz_filtered"],
    "combined": ["mav", "variance", "waveform_length", "mdf_hz_filtered", "mnf_hz_filtered"],
}
STRATEGIES = ["last_third_time", "last_third_repetitions", "terminal_amplitude_decline"]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--labels", type=Path, default=root / "results" / "proposed_fatigue_labels" / "window_labels_all_strategies.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "proposed_fatigue_models")
    args = parser.parse_args()

    features = pd.read_csv(args.dataset)
    labels = pd.read_csv(args.labels)
    features["record_key"] = features[["subject_id", "repetition_id", "side"]].astype(str).agg("_".join, axis=1)
    features["time_key"] = features["start_time_s"].round(6)
    label_cols = ["record_key", "start_time_s", "end_time_s", "strategy", "target"]
    merged = features.merge(labels[label_cols], on=["record_key", "start_time_s", "end_time_s"], how="inner", validate="one_to_many")
    merged = merged[merged["target"] >= 0].copy()

    models = {
        "LDA": make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        strategy_data = merged[merged["strategy"] == strategy]
        for feature_set, columns in FEATURE_SETS.items():
            clean = strategy_data.dropna(subset=columns).copy()
            y = clean["target"].to_numpy(int)
            groups = clean["subject_id"].astype(str).to_numpy()
            if len(np.unique(y)) != 2 or len(np.unique(groups)) < 3:
                print(f"Skipping {strategy}/{feature_set}: insufficient two-class or participant support")
                continue
            X = clean[columns].to_numpy(float)
            for model_name, model in models.items():
                prob = np.full(len(clean), np.nan)
                logo = LeaveOneGroupOut()
                for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
                    fitted = clone(model).fit(X[train_idx], y[train_idx])
                    prob[test_idx] = fitted.predict_proba(X[test_idx])[:, 1]
                    print(f"{strategy} / {feature_set} / {model_name}: held-out participant {fold}/{len(np.unique(groups))}")
                pred = (prob >= 0.5).astype(int)
                metric_rows.append({
                    "strategy": strategy,
                    "feature_set": feature_set,
                    "model": model_name,
                    "threshold": 0.5,
                    "participants": int(clean["subject_id"].nunique()),
                    "recordings": int(clean["record_key"].nunique()),
                    "windows": len(clean),
                    "positive_windows": int(y.sum()),
                    "positive_prevalence": float(y.mean()),
                    "accuracy": accuracy_score(y, pred),
                    "balanced_accuracy": balanced_accuracy_score(y, pred),
                    "precision": precision_score(y, pred, zero_division=0),
                    "recall": recall_score(y, pred, zero_division=0),
                    "f1": f1_score(y, pred, zero_division=0),
                    "average_precision": average_precision_score(y, prob),
                    "roc_auc": roc_auc_score(y, prob),
                })
                output = clean[["subject_id", "repetition_id", "side", "start_time_s", "end_time_s", "record_key"]].copy()
                output["strategy"] = strategy
                output["feature_set"] = feature_set
                output["model"] = model_name
                output["target"] = y
                output["probability"] = prob
                output["predicted_at_0_5"] = pred
                prediction_frames.append(output)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_dir / "model_metrics.csv", index=False)
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    report = [
        "MODELS FOR EXPLORATORY FATIGUE-PROXY LABELS",
        "Validation: leave-one-participant-out. Every reported prediction is from a participant excluded from fitting.",
        "All scores use the fixed probability threshold 0.5; threshold tuning has not been applied.",
        "The three target strategies are not independent physiological fatigue ground truth. Repetition-stage labels depend on automatic, unvalidated peak segmentation.",
        "Feature families: amplitude-derived (MAV, variance, waveform length), unfiltered provisional spectral (MDF/MNF), and combined.",
        "Metrics:",
        metrics.to_string(index=False) if len(metrics) else "No model metrics were produced.",
        "",
        "Detailed per-window out-of-fold probabilities are in oof_predictions.csv.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n", metrics.to_string(index=False))
    print(f"Results: {args.output_dir}")


if __name__ == "__main__":
    main()
