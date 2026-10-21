"""Manual validation of the automatic repetition segmentation.

Step 1 (make):  draws one random recording per participant (seeded), saves a plot of the
                raw signal, the RMS envelope and the detected repetition peaks/cycles,
                and writes a checklist CSV to fill in by eye.
Step 2 (score): reads the filled checklist and reports detection precision, recall,
                count agreement and how often the last peak is followed by a decline.

Plots and the checklist use pseudonymous IDs only (SUBJxx_Rn_side).

Usage (from the project root):
    python src/validate_repetition_segmentation.py make
    ... open results/segmentation_validation/index.html and fill validation_checklist.csv ...
    python src/validate_repetition_segmentation.py score
"""
from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FILL_COLUMNS = ["false_peaks", "missed_reps", "last_peak_followed_by_decline"]


def wilson(successes: float, total: float, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson confidence interval for a proportion (honest uncertainty with few recordings)."""
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return float(centre - half), float(centre + half)


def record_key(frame: pd.DataFrame) -> pd.Series:
    return frame[["subject_id", "repetition_id", "side"]].astype(str).agg("_".join, axis=1)


def make(args: argparse.Namespace) -> None:
    root = args.root
    features = pd.read_csv(args.dataset)
    reps = pd.read_csv(args.candidates)
    features["record_key"] = record_key(features)
    available = sorted(set(reps["record_id"].astype(str)) & set(features["record_key"]))
    if args.exclude is not None:
        # keep a held-out sample: skip recordings that were already used (and tuned on) in an earlier checklist
        seen = pd.read_csv(args.exclude, sep=None, engine="python", encoding="utf-8-sig")
        seen_ids = set(seen["record_id"].astype(str).str.strip())
        available = [key for key in available if key not in seen_ids]
        print(f"Excluding {len(seen_ids)} recordings already reviewed in {args.exclude.name}")
    rng = np.random.default_rng(args.seed)
    chosen: list[str] = []
    for subject in sorted(features["subject_id"].unique()):
        pool = [key for key in available if key.startswith(f"{subject}_")]
        if pool:
            chosen += list(rng.choice(pool, size=min(args.per_subject, len(pool)), replace=False))

    checklist_path = args.output_dir / "validation_checklist.csv"
    if checklist_path.exists() and not args.force:
        existing = pd.read_csv(checklist_path)
        filled = existing[[c for c in FILL_COLUMNS if c in existing.columns]].notna().any(axis=None)
        if filled:
            raise SystemExit(
                f"{checklist_path} already contains filled answers; running 'make' again would erase them.\n"
                "Copy the file somewhere safe, or pass --force (or another --output-dir) to regenerate."
            )

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    rows, cards = [], []
    for key in chosen:
        rec = features[features["record_key"] == key].sort_values("start_time_s")
        cand = reps[reps["record_id"].astype(str) == key].sort_values("candidate_rep")
        t0 = float(rec["start_time_s"].iloc[0])
        mid = (rec["start_time_s"].to_numpy(float) + rec["end_time_s"].to_numpy(float)) / 2 - t0

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
        raw_path = root / str(rec["source_file"].iloc[0])
        if raw_path.exists():
            raw = pd.read_csv(raw_path, usecols=["Relative_Time_s", "Voltage_mV"])
            axes[0].plot(raw["Relative_Time_s"].to_numpy(float) - t0, raw["Voltage_mV"].to_numpy(float),
                         color="#7a8b99", linewidth=0.4, rasterized=True)
        else:
            axes[0].text(0.5, 0.5, "raw file not found", ha="center", va="center", transform=axes[0].transAxes)
        axes[0].set_ylabel("Voltage_mV")
        axes[1].plot(mid, rec["rms"].to_numpy(float), color="#294c60", linewidth=0.9, label="RMS (250 ms windows)")
        for ax in axes:
            for edge in np.unique(np.r_[cand["cycle_start_s"].to_numpy(float), cand["cycle_end_s"].to_numpy(float)]):
                ax.axvline(edge, color="#999999", linestyle=":", linewidth=0.7)
        peak_x = cand["peak_time_s"].to_numpy(float)
        peak_y = cand["peak_smoothed_rms"].to_numpy(float)
        axes[1].plot(peak_x, peak_y, "v", color="#b21f35", markersize=8, label="Detected peak")
        for number, x, y in zip(cand["candidate_rep"], peak_x, peak_y):
            axes[1].annotate(str(int(number)), (x, y), textcoords="offset points", xytext=(0, 8),
                             ha="center", fontsize=9, color="#b21f35")
        axes[1].set_ylabel("RMS")
        axes[1].set_xlabel("Time within recording (s)")
        axes[1].legend(frameon=False, loc="upper right")
        for ax in axes:
            ax.grid(alpha=0.2)
        fig.suptitle(f"{key} | {len(cand)} detected repetitions", fontsize=11)
        fig.savefig(plot_dir / f"{key}.png", dpi=110)
        plt.close(fig)

        rows.append({"record_id": key, "subject_id": cand["subject_id"].iloc[0], "side": cand["side"].iloc[0],
                     "detected_peaks": len(cand), **{col: "" for col in FILL_COLUMNS}, "notes": ""})
        cards.append(f'<article><h3>{html.escape(key)} — {len(cand)} detected</h3>'
                     f'<img src="plots/{html.escape(key)}.png" style="width:100%"></article>')

    pd.DataFrame(rows).to_csv(args.output_dir / "validation_checklist.csv", index=False)
    page = (
        "<!doctype html><html lang='es'><head><meta charset='utf-8'><title>Validación de segmentación</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "article{border-top:1px solid #bbb;margin-top:1.5rem;padding-top:1rem}.note{background:#fff7df;padding:1rem}"
        "</style></head><body><h1>Validación de repeticiones detectadas</h1><div class='note'>"
        "Para cada registro cuenta a ojo: <b>false_peaks</b> = picos rojos que NO son una repetición real; "
        "<b>missed_reps</b> = repeticiones reales sin pico; <b>last_peak_followed_by_decline</b> = si/no, "
        "el último pico va seguido de una bajada clara de la envolvente. Anota en validation_checklist.csv."
        "</div>" + "\n".join(cards) + "</body></html>"
    )
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"{len(rows)} recordings selected (seed {args.seed}).")
    print(f"Open: {args.output_dir / 'index.html'}")
    print(f"Fill: {args.output_dir / 'validation_checklist.csv'}")


def score(args: argparse.Namespace) -> None:
    path = args.output_dir / "validation_checklist.csv"
    # sep=None sniffs the delimiter, so files saved by Excel with ';' still load
    try:
        sheet = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except (csv.Error, pd.errors.ParserError):
        sheet = pd.read_csv(path, encoding="utf-8-sig")   # damaged file: lenient parse, bad rows get reported below
    sheet.columns = [str(c).strip().rstrip(";") for c in sheet.columns]
    for col in ("false_peaks", "missed_reps"):
        sheet[col] = pd.to_numeric(sheet[col], errors="coerce")
    done = sheet.dropna(subset=["false_peaks", "missed_reps"]).copy()
    if done.empty:
        raise SystemExit("No filled rows: complete false_peaks and missed_reps in validation_checklist.csv first.")
    pending = sheet.loc[~sheet.index.isin(done.index), "record_id"].astype(str).tolist()
    if pending:
        print(f"WARNING: {len(pending)} of {len(sheet)} rows are blank or unreadable and were NOT scored:")
        for record in pending:
            print(f"  - {record[:60]}")
        print("(Avoid commas, semicolons and quotes in 'notes'; save as CSV UTF-8 comma-delimited.)\n")
    done["true_positive"] = done["detected_peaks"] - done["false_peaks"]
    done["true_reps"] = done["true_positive"] + done["missed_reps"]
    done["count_error"] = done["detected_peaks"] - done["true_reps"]
    if (done["true_positive"] < 0).any():
        raise SystemExit("false_peaks cannot exceed detected_peaks in any row; please check the checklist.")
    tp, fp, fn = done["true_positive"].sum(), done["false_peaks"].sum(), done["missed_reps"].sum()
    decline = done["last_peak_followed_by_decline"].astype(str).str.strip().str.lower()
    answered = decline.isin(["si", "sí", "s", "yes", "y", "no", "n"])
    yes = decline.isin(["si", "sí", "s", "yes", "y"])
    summary = pd.DataFrame([{
        "recordings_reviewed": len(done),
        "detected_peaks": int(done["detected_peaks"].sum()), "true_repetitions": int(done["true_reps"].sum()),
        "precision": tp / (tp + fp) if tp + fp else np.nan, "recall": tp / (tp + fn) if tp + fn else np.nan,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else np.nan,
        "precision_ci95_low": wilson(tp, tp + fp)[0], "precision_ci95_high": wilson(tp, tp + fp)[1],
        "recall_ci95_low": wilson(tp, tp + fn)[0], "recall_ci95_high": wilson(tp, tp + fn)[1],
        "records_exact_count": int((done["count_error"] == 0).sum()),
        "records_within_1": int((done["count_error"].abs() <= 1).sum()),
        "median_count_error": float(done["count_error"].median()),
        "last_peak_decline_yes": int(yes.sum()), "last_peak_decline_answered": int(answered.sum()),
    }])
    done.to_csv(args.output_dir / "validation_scored_by_record.csv", index=False)
    summary.to_csv(args.output_dir / "validation_summary.csv", index=False)
    print(summary.round(3).T.to_string(header=False))
    print(f"Saved to {args.output_dir}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["make", "score"])
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--candidates", type=Path, default=root / "results" / "fatigue_candidates" / "candidate_repetitions.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "segmentation_validation")
    parser.add_argument("--per-subject", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude", type=Path, default=None,
                        help="Checklist CSV whose recordings must not be selected again (held-out validation)")
    parser.add_argument("--force", action="store_true", help="Regenerate even if the checklist already has answers")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make(args) if args.mode == "make" else score(args)


if __name__ == "__main__":
    main()
