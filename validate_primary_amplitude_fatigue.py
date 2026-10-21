"""Guideline-oriented subject-held-out validation for the amplitude-fatigue proxy.

Time/repetition thirds are descriptive approximations only. The sole target is
the inherited sustained terminal RMS-decline label. Results predict that
operational label and do not independently validate physiological fatigue.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, average_precision_score,
    balanced_accuracy_score, cohen_kappa_score, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_SETS = {
    "spectral_only_primary": ["mdf_hz_filtered", "mnf_hz_filtered"],
    "amplitude_only_exploratory_circular": ["rms", "mav", "variance", "waveform_length"],
    "combined_exploratory_circular": ["rms", "mav", "variance", "waveform_length", "mdf_hz_filtered", "mnf_hz_filtered"],
}
RF_GRID = {"max_depth": [4, 8], "min_samples_leaf": [20]}


def metrics(y: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict[str, float]:
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "cohen_kappa": float(cohen_kappa_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) == 2 else float("nan"),
        "precision_macro": float(np.mean(p)), "recall_macro": float(np.mean(r)), "f1_macro": float(np.mean(f)),
        "precision_active_pre_decline": float(p[0]), "recall_active_pre_decline": float(r[0]), "f1_active_pre_decline": float(f[0]),
        "support_active_pre_decline": int(support[0]),
        "precision_amplitude_decline": float(p[1]), "recall_amplitude_decline": float(r[1]), "f1_amplitude_decline": float(f[1]),
        "support_amplitude_decline": int(support[1]),
        "average_precision": float(average_precision_score(y, score)) if np.any(y == 1) else float("nan"),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan"),
    }
    return result


def write_latex_metrics_table(frame: pd.DataFrame, path: Path) -> None:
    fields = [
        ("Accuracy", "accuracy"), ("Balanced accuracy", "balanced_accuracy"),
        ("Macro precision", "precision_macro"), ("Macro recall", "recall_macro"),
        ("Macro F1", "f1_macro"),
        ("No-decline precision", "precision_active_pre_decline"),
        ("No-decline recall", "recall_active_pre_decline"),
        ("No-decline F1", "f1_active_pre_decline"),
        ("Decline precision", "precision_amplitude_decline"),
        ("Decline recall", "recall_amplitude_decline"),
        ("Decline F1", "f1_amplitude_decline"),
        ("ROC-AUC", "roc_auc"), ("PR-AUC", "average_precision"),
        ("Cohen kappa", "cohen_kappa"),
    ]
    cols = "p{3.0cm}" + "r" * len(fields)
    lines = [
        r"\begin{sidewaystable}[p]", r"\centering", r"\scriptsize",
        r"\caption{Complete participant-fold mean $\pm$ SD metrics for every compared model. Metrics are computed per held-out participant; AUC means omit folds with only one observed class.}",
        r"\label{tab:all_model_metrics}", r"\resizebox{\textheight}{!}{%",
        r"\begin{tabular}{" + cols + "}", r"\toprule",
        "Model & " + " & ".join(label for label, _ in fields) + r" \\",
        r"\midrule",
    ]
    for row in frame.itertuples(index=False):
        record = row._asdict()
        model = str(record["model"]).replace("_", " ")
        cells = []
        for _, metric in fields:
            mean = record.get(f"fold_mean_{metric}", np.nan)
            sd = record.get(f"fold_sd_{metric}", np.nan)
            cells.append(r"--" if pd.isna(mean) or pd.isna(sd) else f"${mean:.3f} \\pm {sd:.3f}$")
        lines.append(model + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}}", r"\end{sidewaystable}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--labels", type=Path, default=root / "results" / "primary_amplitude_fatigue_labels" / "primary_amplitude_fatigue_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "primary_amplitude_fatigue_validation")
    args = parser.parse_args()

    data = pd.read_csv(args.dataset)
    labels = pd.read_csv(args.labels)
    data["record_key"] = data[["subject_id", "repetition_id", "side"]].astype(str).agg("_".join, axis=1)
    label_columns = ["record_key", "start_time_s", "end_time_s", "fatigue_target",
                     "recording_third_approximation", "candidate_repetition_third_approximation"]
    merged = data.merge(labels[label_columns], on=["record_key", "start_time_s", "end_time_s"],
                        how="inner", validate="one_to_one")
    merged = merged[merged["fatigue_target"].isin([0, 1])].copy()

    model_specs: list[tuple[str, list[str] | None, str]] = [("majority_baseline", None, "dummy")]
    # Baseline specified in the project validation checklist; RMS is explicitly
    # identified as sharing the target construction and thus is not independent.
    model_specs.append(("simple_LDA_RMS_MDF_reference_circular", ["rms", "mdf_hz_filtered"], "lda_empirical"))
    model_specs.append(("simple_LDA_RMS_MDF_equal_prior_reference_circular", ["rms", "mdf_hz_filtered"], "lda_equal"))
    for name, columns in FEATURE_SETS.items():
        for prior in ("empirical", "equal"):
            model_specs.append((f"LDA_{name}_{prior}_prior", columns, f"lda_{prior}"))
        for balance in ("balanced", "unweighted"):
            model_specs.append((f"RandomForest_{name}_{balance}", columns, f"rf_{balance}"))

    summaries, folds, predictions, confusions, tuning = [], [], [], [], []
    for model_name, columns, kind in model_specs:
        clean = merged.copy() if columns is None else merged.dropna(subset=columns).copy()
        y = clean["fatigue_target"].to_numpy(int)
        groups = clean["subject_id"].astype(str).to_numpy()
        if len(np.unique(y)) != 2 or len(np.unique(groups)) < 4:
            print(f"Skipping {model_name}: insufficient class/participant support")
            continue
        X = None if columns is None else clean[columns].to_numpy(float)
        pred_oof, prob_oof = np.full(len(y), -1, int), np.full(len(y), np.nan)
        for fold_no, (train, test) in enumerate(LeaveOneGroupOut().split(np.zeros(len(y)), y, groups), 1):
            yt = y[train]
            if kind == "dummy":
                estimator = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(train), 1)), yt)
                prob = estimator.predict_proba(np.zeros((len(test), 1)))[:, 1]
                params = "most frequent class in outer training participants"
            elif kind.startswith("lda"):
                priors = [0.5, 0.5] if kind == "lda_equal" else None
                estimator = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(priors=priors))
                estimator.fit(X[train], yt)
                prob = estimator.predict_proba(X[test])[:, 1]
                params = f"fixed LDA; {'equal' if priors else 'empirical'} priors; train-fold scaling"
            else:
                train_groups = groups[train]
                n_inner = min(3, len(np.unique(train_groups)))
                forest = RandomForestClassifier(n_estimators=100,
                    class_weight="balanced" if kind == "rf_balanced" else None,
                    random_state=42, n_jobs=-1)
                search = GridSearchCV(forest, RF_GRID, scoring="f1_macro",
                    cv=GroupKFold(n_splits=n_inner), refit=True, n_jobs=1,
                    error_score="raise")
                search.fit(X[train], yt, groups=train_groups)
                estimator = search.best_estimator_
                prob = estimator.predict_proba(X[test])[:, 1]
                params = f"class_weight={estimator.class_weight}; {search.best_params_}"
                tuning.append({"model": model_name, "held_out_subject": groups[test][0],
                    "inner_group_folds": n_inner, "inner_metric": "macro_f1",
                    "best_parameters": params, "best_inner_macro_f1": search.best_score_})
            pred = (prob >= 0.5).astype(int)
            pred_oof[test], prob_oof[test] = pred, prob
            folds.append({"model": model_name, "held_out_subject": groups[test][0],
                "windows": len(test), "positive_prevalence": float(y[test].mean()),
                "threshold": 0.5, "parameters": params, **metrics(y[test], pred, prob)})
            print(f"{model_name}: participant fold {fold_no}/{len(np.unique(groups))}")

        pooled = metrics(y, pred_oof, prob_oof)
        fold_frame = pd.DataFrame([x for x in folds if x["model"] == model_name])
        summary = {"model": model_name, "feature_columns": ";".join(columns or []),
            "participants": int(clean.subject_id.nunique()), "recordings": int(clean.record_key.nunique()),
            "windows": len(clean), "positive_windows": int(y.sum()),
            "negative_windows": int((y == 0).sum()), "positive_prevalence": float(y.mean()),
            "decision_threshold": 0.5, **{f"pooled_{k}": v for k, v in pooled.items()}}
        for col in pooled:
            summary[f"fold_mean_{col}"] = float(fold_frame[col].mean())
            summary[f"fold_sd_{col}"] = float(fold_frame[col].std(ddof=1))
        summaries.append(summary)
        cm = confusion_matrix(y, pred_oof, labels=[0, 1])
        for i, true in enumerate(["active_pre_decline", "amplitude_decline"]):
            for j, predicted in enumerate(["active_pre_decline", "amplitude_decline"]):
                confusions.append({"model": model_name, "true_class": true,
                    "predicted_class": predicted, "window_count": int(cm[i, j])})
        out = clean[["subject_id", "repetition_id", "side", "record_key", "start_time_s", "end_time_s",
                     "recording_third_approximation", "candidate_repetition_third_approximation"]].copy()
        out["model"], out["target"], out["probability_amplitude_decline"], out["prediction_at_0_5"] = model_name, y, prob_oof, pred_oof
        predictions.append(out)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "model_metrics.csv", index=False)
    write_latex_metrics_table(summary_frame, args.output_dir / "model_metrics_table.tex")
    pd.DataFrame(folds).to_csv(args.output_dir / "fold_metrics_by_subject.csv", index=False)
    pd.DataFrame(folds).to_csv(args.output_dir / "results.csv", index=False)
    pd.DataFrame(confusions).to_csv(args.output_dir / "confusion_matrices.csv", index=False)
    pd.DataFrame(tuning).to_csv(args.output_dir / "nested_rf_hyperparameters.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    report = [
        "VALIDATION OF THE AMPLITUDE-DECLINE FATIGUE PROXY",
        "Sole positive target: inherited sustained terminal amplitude decline. Time/repetition thirds appear in OOF output only as descriptive approximations; they never define the target.",
        "Outer validation: leave-one-participant-out. Hyperparameter tuning for RF is nested in outer training participants using GroupKFold and macro-F1. Scaling is fitted within training folds. No resampling is performed; RF class weights and LDA priors are compared while test prevalence remains natural.",
        "Models: majority baseline; simple RMS+MDF LDA reference; LDA with empirical/equal priors; RF balanced/unweighted for spectral-only, amplitude-only, and combined features. Probability threshold is fixed at 0.5.",
        "Guideline metrics: accuracy, balanced accuracy, macro precision/recall/F1, class-specific precision/recall/F1/support, PR-AUC (average precision), ROC-AUC, confusion matrices, and per-participant scores with mean and SD.",
        "results.csv contains one row per model x held-out participant, with every fold metric and participant identity. model_metrics.csv gives pooled out-of-fold metrics and mean +/- SD across participants. Feature-family comparisons (spectral-only, amplitude-only, combined) are the available ablation; because amplitude defines the target, only the spectral family is less circular.",
        "Primary less-circular analysis is spectral-only, but MDF/MNF are unfiltered and acquisition/filter/sample calibration are unresolved. RMS/amplitude-only/combined scores are circular because target construction uses RMS decline.",
        "Critical limitation: only recordings with a qualifying terminal decline contribute; target 0 means active pre-decline windows within those selected recordings. Other recordings/windows are unknown, not verified non-fatigue. This selected-window evaluation cannot establish clinical sensitivity/specificity or physiological fatigue validity.",
        "Windows are 250 ms with 125 ms overlap and are correlated. Splitting by participant reduces participant leakage but windows are not independent observations.",
        "Files: results.csv (one row per model x held-out participant); model_metrics.csv (pooled and fold mean +/- SD); model_metrics_table.tex (all model mean +/- SD metrics ready for the report); fold_metrics_by_subject.csv; confusion_matrices.csv; nested_rf_hyperparameters.csv; feature_importance_final_rf.csv (descriptive final-fit impurity importance, not held-out); oof_predictions.csv (out-of-fold predictions and descriptive thirds); final_models/ contains serialized spectral-only LDA/RF and metadata fitted after CV.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("\nRESULTS SUMMARY")
    shown = pd.DataFrame(summaries)
    print(shown[["model", "positive_prevalence", "pooled_accuracy", "pooled_balanced_accuracy",
        "pooled_f1_macro", "pooled_precision_amplitude_decline", "pooled_recall_amplitude_decline",
        "pooled_f1_amplitude_decline", "pooled_average_precision", "pooled_roc_auc"]].to_string(index=False))

    # Reproducible deployable artifacts are fitted only after cross-validation
    # reporting. They do not alter the held-out estimates above.
    model_dir = args.output_dir / "final_models"
    model_dir.mkdir(exist_ok=True)
    primary_columns = FEATURE_SETS["spectral_only_primary"]
    final_data = merged.dropna(subset=primary_columns).copy()
    X_all = final_data[primary_columns].to_numpy(float)
    y_all = final_data["fatigue_target"].to_numpy(int)
    group_all = final_data["subject_id"].astype(str).to_numpy()
    inner_n = min(3, len(np.unique(group_all)))
    final_search = GridSearchCV(
        RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1),
        RF_GRID, scoring="f1_macro", cv=GroupKFold(n_splits=inner_n),
        refit=True, n_jobs=1, error_score="raise",
    ).fit(X_all, y_all, groups=group_all)
    joblib.dump(final_search.best_estimator_, model_dir / "random_forest_spectral_balanced.joblib")
    pd.DataFrame({
        "feature": primary_columns,
        "impurity_importance": final_search.best_estimator_.feature_importances_,
        "interpretation_limit": "descriptive importance from final all-subject fit; not an independent or held-out importance estimate",
    }).to_csv(args.output_dir / "feature_importance_final_rf.csv", index=False)
    final_lda = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(priors=[0.5, 0.5]))
    final_lda.fit(X_all, y_all)
    joblib.dump(final_lda, model_dir / "lda_spectral_equal_prior.joblib")
    model_info = {
        "target": "inherited sustained terminal amplitude-decline operational proxy; not independently verified physiological fatigue",
        "features": primary_columns,
        "training_subjects": sorted(np.unique(group_all).tolist()),
        "training_windows": int(len(y_all)),
        "positive_windows": int(y_all.sum()),
        "decision_threshold": 0.5,
        "random_seed": 42,
        "random_forest_parameters_selected_by_grouped_cv": final_search.best_params_,
        "rf_inner_best_macro_f1": float(final_search.best_score_),
        "validation_metrics_source": "out-of-fold results.csv; final artifacts were fitted on all labeled rows after CV",
        "warning": "Use only to reproduce the operational amplitude-decline detector. Features are unfiltered MDF/MNF; verify acquisition and filtering before physiological interpretation.",
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(model_info, indent=2), encoding="utf-8")
    print(f"Final models and metadata saved in {model_dir}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
