"""Audit how terminal-decline candidate counts change across rule settings.

This script reads existing derived CSVs only. It does not edit raw recordings.
It creates a sensitivity grid and a pseudonymous RMS review gallery.
"""

from __future__ import annotations

import argparse
import html
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DURATIONS = [1.0, 1.5, 2.0, 2.5, 3.0]
RATIOS = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root / "data" / "features_dataset_corrected_unlabeled.csv")
    parser.add_argument("--record-summary", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_record_summary.csv")
    parser.add_argument("--window-labels", type=Path, default=root / "results" / "terminal_decline_candidates" / "terminal_decline_window_labels.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "terminal_decline_sensitivity")
    parser.add_argument("--primary-duration", type=float, default=1.0,
                        help="Minimum duration used for the currently selected primary rule")
    args = parser.parse_args()

    summary = pd.read_csv(args.record_summary)
    required = {"record_id", "side", "terminal_active_duration_s", "terminal_end_to_peak_rms_ratio", "terminal_decline_slope_rms_per_s"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"Record summary is missing columns: {sorted(missing)}")
    summary["terminal_decline_slope_rms_per_s"] = pd.to_numeric(summary["terminal_decline_slope_rms_per_s"], errors="coerce")
    summary["terminal_active_duration_s"] = pd.to_numeric(summary["terminal_active_duration_s"], errors="coerce")
    summary["terminal_end_to_peak_rms_ratio"] = pd.to_numeric(summary["terminal_end_to_peak_rms_ratio"], errors="coerce")
    summary["negative_slope"] = summary["terminal_decline_slope_rms_per_s"] < 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    grid_rows: list[dict[str, object]] = []
    record_grid: list[pd.DataFrame] = []
    for duration in DURATIONS:
        for ratio in RATIOS:
            passed = (
                summary["negative_slope"]
                & (summary["terminal_active_duration_s"] >= duration)
                & (summary["terminal_end_to_peak_rms_ratio"] <= ratio)
            ).fillna(False)
            grid_rows.append({
                "minimum_duration_s": duration,
                "maximum_end_to_peak_ratio": ratio,
                "qualifying_records": int(passed.sum()),
                "percent_of_140_records": float(passed.mean() * 100),
                "qualifying_right_records": int((passed & summary["side"].eq("right")).sum()),
                "qualifying_left_records": int((passed & summary["side"].eq("left")).sum()),
            })
            subset = summary[["subject_id", "record_id", "side", "repetition_code"]].copy()
            subset["minimum_duration_s"] = duration
            subset["maximum_end_to_peak_ratio"] = ratio
            subset["candidate_decline"] = passed.to_numpy(bool)
            record_grid.append(subset)
    grid = pd.DataFrame(grid_rows)
    grid.to_csv(args.output_dir / "threshold_sensitivity_summary.csv", index=False)
    pd.concat(record_grid, ignore_index=True).to_csv(args.output_dir / "threshold_sensitivity_by_record.csv", index=False)
    pivot = grid.pivot(index="minimum_duration_s", columns="maximum_end_to_peak_ratio", values="qualifying_records")
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", aspect="auto", vmin=0, vmax=len(summary))
    ax.set_xticks(np.arange(len(pivot.columns)), [f"{value:.2f}" for value in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), [f"{value:.1f}" for value in pivot.index])
    ax.set(xlabel="Maximum final-active RMS / peak RMS", ylabel="Minimum decline duration (s)",
           title="Records passing each provisional rule combination")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            ax.text(j, i, str(int(pivot.iloc[i, j])), ha="center", va="center", color="#222222", fontsize=9)
    fig.colorbar(image, ax=ax, label="Qualifying recordings (of 140)")
    fig.savefig(args.output_dir / "threshold_sensitivity_heatmap.png", dpi=160)
    plt.close(fig)

    lenient = summary["negative_slope"] & (summary["terminal_active_duration_s"] >= min(DURATIONS)) & (summary["terminal_end_to_peak_rms_ratio"] <= max(RATIOS))
    strict = summary["negative_slope"] & (summary["terminal_active_duration_s"] >= max(DURATIONS)) & (summary["terminal_end_to_peak_rms_ratio"] <= min(RATIOS))
    base = summary["negative_slope"] & (summary["terminal_active_duration_s"] >= args.primary_duration) & (summary["terminal_end_to_peak_rms_ratio"] <= 0.75)
    summary["sensitivity_class"] = np.select(
        [strict.fillna(False), lenient.fillna(False)],
        ["robust_across_grid", "threshold_sensitive"], default="fails_even_lenient_grid",
    )
    summary["passes_primary_rule"] = base.fillna(False)
    summary.to_csv(args.output_dir / "terminal_decline_sensitivity_by_record.csv", index=False)

    # Plot all signals as anonymized cards so pass/fail and near-boundary cases
    # can be visually inspected without exposing participant folder names.
    windows = pd.read_csv(args.window_labels)
    windows["record_key"] = windows["subject_id"].astype(str) + "_" + windows["repetition_id"].astype(str) + "_" + windows["side"].astype(str)
    summary_by_key = summary.set_index("record_id")
    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    cards: list[str] = []
    for record_key, record in windows.groupby("record_key", sort=True):
        if record_key not in summary_by_key.index:
            continue
        info = summary_by_key.loc[record_key]
        record = record.sort_values("start_time_s")
        t = ((record["start_time_s"].to_numpy(float) + record["end_time_s"].to_numpy(float)) / 2)
        t = t - float(record["start_time_s"].iloc[0])
        values = record["rms_smooth"].to_numpy(float)
        threshold = float(record["activity_cutoff"].iloc[0])
        peak_t = float(info.get("last_candidate_peak_time_s", np.nan))
        category = str(info["sensitivity_class"])
        passes = bool(info["passes_primary_rule"])
        fig, ax = plt.subplots(figsize=(9, 3.3), constrained_layout=True)
        ax.plot(t, values, color="#294c60", linewidth=1.0, label="Smoothed RMS")
        ax.axhline(threshold, color="#777777", linestyle="--", linewidth=1, label="Activity cutoff")
        if np.isfinite(peak_t):
            ax.axvline(peak_t, color="#d17a22", linestyle=":", linewidth=1.4, label="Last candidate peak")
            tail_color = "#c44536" if passes else ("#e09f3e" if category == "threshold_sensitive" else "#8a9a9a")
            end_t = peak_t + float(info["terminal_active_duration_s"])
            ax.axvspan(peak_t, end_t, color=tail_color, alpha=0.20, label="Active terminal tail")
        ax.set(title=f"{record_key} | {category} | primary rule: {'PASS' if passes else 'FAIL'}",
               xlabel="Time within file (s)", ylabel="RMS (recorded units; unverified)")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3, fontsize=8, loc="upper right")
        filename = f"{record_key}.png"
        fig.savefig(plot_dir / filename, dpi=120)
        plt.close(fig)
        escaped = html.escape(record_key)
        cards.append(
            f'<article data-class="{html.escape(category)}" data-pass="{str(passes).lower()}">'
            f'<h3>{escaped}</h3><p>{html.escape(category)} · primary rule: {"PASS" if passes else "FAIL"} · '
            f"duration {info['terminal_active_duration_s']:.2f} s · ratio {info['terminal_end_to_peak_rms_ratio']:.2f}</p>"
            f'<a href="plots/{html.escape(filename)}"><img loading="lazy" src="plots/{html.escape(filename)}" alt="RMS trace {escaped}"></a></article>'
        )
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Terminal-decline sensitivity review</title>
<style>body{font:15px system-ui,sans-serif;margin:2rem;color:#18252d}header{max-width:900px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:1rem}article{border:1px solid #ccd4d8;border-radius:8px;padding:.8rem}img{width:100%;height:auto}h3{margin:.2rem 0}select{padding:.5rem}p{color:#46565f}</style></head><body>
<header><h1>Terminal-decline sensitivity review</h1><p>All recordings are shown with pseudonymous IDs. Red marks recordings passing the selected primary duration / 0.75 rule; orange indicates sensitivity to the rule grid; gray fails even the lenient grid. Shading marks the active tail after the last detected peak, ending at the estimated activity cutoff.</p>
<label>Filter: <select id="filter"><option value="all">All</option><option value="robust_across_grid">Robust across grid</option><option value="threshold_sensitive">Threshold-sensitive</option><option value="fails_even_lenient_grid">Fails even lenient grid</option></select></label></header>
<main id="cards">__CARDS__</main><script>document.getElementById('filter').addEventListener('change',e=>{document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.class!==e.target.value)})</script></body></html>""".replace("__CARDS__", "\n".join(cards))
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")

    n = len(summary)
    report = [
        "TERMINAL-DECLINE RULE SENSITIVITY AUDIT",
        f"Recordings: {n}",
        f"Primary rule (>= {args.primary_duration:g} s, negative slope, end/peak <=0.75): {int(base.sum())}/{n}",
        f"Robust across all tested settings (>=3 s and end/peak <=0.50): {int(strict.sum())}/{n}",
        f"Pass under at least one lenient tested setting (>=1 s and end/peak <=0.90): {int(lenient.sum())}/{n}",
        f"Threshold-sensitive: {int((lenient & ~strict).sum())}/{n}",
        f"Fails even the lenient tested grid: {int((~lenient).sum())}/{n}",
        "Counts at ratio <=0.75 by minimum duration: " + "; ".join(
            f">={duration:g}s: {int(grid[(grid.minimum_duration_s == duration) & (grid.maximum_end_to_peak_ratio == 0.75)].qualifying_records.iloc[0])}"
            for duration in DURATIONS
        ),
        "Counts at duration >=2s by maximum ratio: " + "; ".join(
            f"<={ratio:.2f}: {int(grid[(grid.minimum_duration_s == 2.0) & (grid.maximum_end_to_peak_ratio == ratio)].qualifying_records.iloc[0])}"
            for ratio in RATIOS
        ),
        f"Primary rule (>= {args.primary_duration:g} s, negative slope, end/peak <=0.75): {int(base.sum())}/{n}",
        "The grid describes label sensitivity only; it does not estimate true fatigue precision or accuracy.",
        "Visual inspection is needed because an end-of-file decline may be fatigue, relaxation, stopping, electrode motion, or acquisition artifact.",
        "The derived CSVs and gallery use pseudonymous IDs; source recordings were read but not modified.",
    ]
    (args.output_dir / "README.txt").write_text("\n".join(report), encoding="utf-8")
    print("Sensitivity audit complete.")
    print("\n".join(report[1:7]))
    print(f"Grid: {args.output_dir / 'threshold_sensitivity_summary.csv'}")
    print(f"Review gallery: {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
