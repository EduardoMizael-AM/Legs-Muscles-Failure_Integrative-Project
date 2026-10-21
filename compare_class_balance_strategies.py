"""Compare class-balancing strategies under grouped validation.

Uses outer leave-one-participant-out evaluation, training-only scaling, an
always-majority baseline, a simple RMS+MDF LDA baseline, and an inner grouped
grid search for Random Forest. Outputs are kept separate from earlier runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


STRATEGIES = ["last_third_time", "last_third_repetitions", "terminal_amplitude_decline"]
FEATURE_SETS = {
    "amplitude_derived": ["mav", "variance", "waveform_length"],
    "unfiltered_spectral": ["mdf_hz_filtered", "mnf_hz_filtered"],
    "combined": ["mav", "variance", "waveform_length", "mdf_hz_filtered", "mnf_hz_filtered"],
}
SIMPLE_BASELINE_FEATURES = ["rms", "mdf_hz_filtered"]
RF_GRID = {
    "max_depth": [4, 8],
    "min_samples_leaf": [20],
}


def metric_values(y: np.ndarray, pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    p, r, f, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=0,
    )
    values = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan"),
        "precision_macro": float(np.mean(p)),
        "recall_macro": float(np.mean(r)),
        "f1_macro": float(np.mean(f)),
        "precision_nonfatigue": float(p[0]),
        "recall_nonfatigue": float(r[0]),
        "f1_nonfatigue": float(f[0]),
        "support_nonfatigue": int(support[0]),
        "precision_fatigue": float(p[1]),
        "recall_fatigue": float(r[1]),
        "f1_fatigue": float(f[1]),
        "support_fatigue": int(support[1]),
    }
    values["average_precision"] = float(average_precision_score(y, probability)) if np.any(y == 1) else float("nan")
    values["roc_auc"] = float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")
    return values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--labels", type=Path, default=root / "results" / "proposed_fatigue_labels" / "window_labels_all_strategies.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "class_balance_comparison")
    args = parser.parse_args()

    features = pd.read_csv(args.dataset)
    labels = pd.read_csv(args.labels)
    features["record_key"] = features[["subject_id", "repetition_id", "side"]].astype(str).agg("_".join, axis=1)
    merged = features.merge(
        labels[["record_key", "start_time_s", "end_time_s", "strategy", "target"]],
        on=["record_key", "start_time_s", "end_time_s"], how="inner", validate="one_to_many",
    )
    merged = merged[merged["target"] >= 0].copy()

    metric_rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    confusion_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for strategy in STRATEGIES:
        strategy_data = merged[merged["strategy"] == strategy]
        model_specs: list[tuple[str, list[str] | None, str]] = [("majority_baseline", None, "dummy")]
        model_specs.append(("simple_LDA_RMS_MDF_empirical_prior", SIMPLE_BASELINE_FEATURES, "lda_empirical"))
        model_specs.append(("simple_LDA_RMS_MDF_equal_prior", SIMPLE_BASELINE_FEATURES, "lda_equal"))
        for feature_set, columns in FEATURE_SETS.items():
            model_specs.append((f"LDA_empirical_prior_{feature_set}", columns, "lda_empirical"))
            model_specs.append((f"LDA_equal_prior_{feature_set}", columns, "lda_equal"))
        for feature_set, columns in FEATURE_SETS.items():
            model_specs.append((f"RandomForest_balanced_{feature_set}", columns, "rf_balanced"))
            model_specs.append((f"RandomForest_unweighted_{feature_set}", columns, "rf_unweighted"))

        for model_name, columns, model_type in model_specs:
            clean = strategy_data.copy() if columns is None else strategy_data.dropna(subset=columns).copy()
            y = clean["target"].to_numpy(int)
            groups = clean["subject_id"].astype(str).to_numpy()
            if len(np.unique(y)) != 2 or len(np.unique(groups)) < 4:
                print(f"Skipping {strategy}/{model_name}: insufficient two-class or participant support")
                continue
            X = None if columns is None else clean[columns].to_numpy(float)
            prob_oof = np.full(len(clean), np.nan)
            pred_oof = np.full(len(clean), -1, dtype=int)
            logo = LeaveOneGroupOut()

            for fold_index, (train_idx, test_idx) in enumerate(logo.split(np.zeros(len(y)), y, groups), start=1):
                y_train, y_test = y[train_idx], y[test_idx]
                if model_type == "dummy":
                    estimator = DummyClassifier(strategy="most_frequent")
                    estimator.fit(np.zeros((len(train_idx), 1)), y_train)
                    fold_prob = estimator.predict_proba(np.zeros((len(test_idx), 1)))[:, 1]
                    selected_params = "majority class from outer training subjects"
                elif model_type in {"lda_empirical", "lda_equal"}:
                    priors = [0.5, 0.5] if model_type == "lda_equal" else None
                    estimator = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(priors=priors))
                    estimator.fit(X[train_idx], y_train)
                    fold_prob = estimator.predict_proba(X[test_idx])[:, 1]
                    prior_label = "equal class priors" if model_type == "lda_equal" else "empirical class priors"
                    selected_params = f"fixed LDA, {prior_label}; scaler fitted on outer training subjects"
                else:
                    train_groups = groups[train_idx]
                    n_inner = min(3, len(np.unique(train_groups)))
                    inner_cv = GroupKFold(n_splits=n_inner)
                    forest = RandomForestClassifier(
                        n_estimators=60,
                        class_weight="balanced" if model_type == "rf_balanced" else None,
                        random_state=42, n_jobs=-1,
                    )
                    search = GridSearchCV(
                        forest, RF_GRID, scoring="f1_macro", cv=inner_cv,
                        refit=True, n_jobs=1, error_score="raise",
                    )
                    search.fit(X[train_idx], y_train, groups=train_groups)
                    estimator = search.best_estimator_
                    fold_prob = estimator.predict_proba(X[test_idx])[:, 1]
                    selected_params = f"class_weight={estimator.class_weight}; {search.best_params_}"
                    tuning_rows.append({
                        "strategy": strategy, "model": model_name,
                        "held_out_subject": str(groups[test_idx][0]),
                        "inner_scoring": "macro_f1", "inner_group_folds": n_inner,
                        "best_params": selected_params,
                        "best_inner_macro_f1": float(search.best_score_),
                    })

                fold_pred = (fold_prob >= 0.5).astype(int)
                prob_oof[test_idx] = fold_prob
                pred_oof[test_idx] = fold_pred
                fold_metric = metric_values(y_test, fold_pred, fold_prob)
                fold_rows.append({
                    "strategy": strategy, "model": model_name,
                    "held_out_subject": str(groups[test_idx][0]),
                    "windows": len(test_idx), "positive_prevalence": float(y_test.mean()),
                    "selected_parameters": selected_params, **fold_metric,
                })
                print(f"{strategy} / {model_name}: participant {fold_index}/{len(np.unique(groups))}")

            pooled = metric_values(y, pred_oof, prob_oof)
            fold_frame = pd.DataFrame([row for row in fold_rows if row["strategy"] == strategy and row["model"] == model_name])
            summary: dict[str, object] = {
                "strategy": strategy, "model": model_name,
                "participants": int(clean["subject_id"].nunique()),
                "recordings": int(clean["record_key"].nunique()), "windows": len(clean),
                "positive_windows": int(y.sum()), "negative_windows": int((y == 0).sum()),
                "positive_prevalence": float(y.mean()), "threshold": 0.5,
            }
            for name, value in pooled.items():
                summary[f"pooled_{name}"] = value
            for column in [c for c in fold_frame.columns if c in metric_values(y[:2], np.array([0, 1]), np.array([0.0, 1.0]))]:
                summary[f"fold_mean_{column}"] = float(fold_frame[column].mean())
                summary[f"fold_sd_{column}"] = float(fold_frame[column].std(ddof=1))
            metric_rows.append(summary)

            cm = confusion_matrix(y, pred_oof, labels=[0, 1])
            for true_class in (0, 1):
                for predicted_class in (0, 1):
                    confusion_rows.append({
                        "strategy": strategy, "model": model_name,
                        "true_class": "no_fatigue_proxy" if true_class == 0 else "fatigue_proxy",
                        "predicted_class": "no_fatigue_proxy" if predicted_class == 0 else "fatigue_proxy",
                        "window_count": int(cm[true_class, predicted_class]),
                    })
            out = clean[["subject_id", "repetition_id", "side", "start_time_s", "end_time_s", "record_key"]].copy()
            out["strategy"] = strategy
            out["model"] = model_name
            out["target"] = y
            out["probability_fatigue"] = prob_oof
            out["predicted_at_0_5"] = pred_oof
            prediction_frames.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(args.output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "fold_metrics_by_subject.csv", index=False)
    pd.DataFrame(confusion_rows).to_csv(args.output_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(args.output_dir / "nested_rf_hyperparameters.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    report = [
        "SUBJECT-WISE VALIDATION OF EXPLORATORY FATIGUE PROXIES",
        "Outer validation: leave-one-participant-out. All windows from a held-out participant are excluded from fitting.",
        "Baselines: fold-specific majority-class DummyClassifier and fixed LDA with RMS + unfiltered MDF, each evaluated with empirical and equal LDA class priors.",
        "LDA standardization is fitted inside each outer training fold. Random Forest hyperparameters are selected by 3-fold GroupKFold inside outer training participants, optimizing macro-F1; no held-out participant is used for tuning.",
        "Random Forest: 60 trees, seed=42; balanced and unweighted variants are compared with the same inner grouped grid (max_depth 4/8, min_samples_leaf=20), scoring macro-F1. All final hard predictions use probability 0.5.",
        "Class distribution is reported per strategy and validation retains natural prevalence. RF weights and LDA priors affect training only; no resampling is applied. The majority baseline predicts the training-fold majority class.",
        "Metrics include pooled OOF values, macro and class-specific scores, mean +/- SD across held-out participants, per-participant results, and pooled confusion matrices.",
        "Features use 250 ms windows with 125 ms step (50% overlap). Overlapping windows are correlated; participant-wise splitting prevents same-participant leakage but does not make windows independent.",
        "Targets remain operational proxies. Time-third labels assume late recording position represents fatigue. Candidate repetition labels use unverified amplitude-peak segmentation. Terminal-decline labels and amplitude-derived features share amplitude information, so this is not independent physiological validation.",
        "See model_metrics.csv for pooled and mean +/- SD metrics; fold_metrics_by_subject.csv for participant results; confusion_matrices.csv for aggregated matrices; nested_rf_hyperparameters.csv for inner tuning; oof_predictions.csv for all held-out predictions.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nMODEL SUMMARY")
    print(pd.DataFrame(metric_rows)[[
        "strategy", "model", "positive_prevalence", "pooled_accuracy",
        "pooled_balanced_accuracy", "pooled_precision_macro", "pooled_recall_macro",
        "pooled_f1_macro", "pooled_precision_fatigue", "pooled_recall_fatigue",
        "pooled_f1_fatigue", "fold_mean_f1_macro", "fold_sd_f1_macro",
    ]].to_string(index=False))
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
