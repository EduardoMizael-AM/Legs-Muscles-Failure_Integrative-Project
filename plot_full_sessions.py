"""Plot every complete available raw recording grouped by participant.

Each CSV is drawn as its own time series. Files are not concatenated because
their relative clocks reset and no shared session/event clock is available.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


R_ORDER = ["R0", "R0_1", "R1", "R2", "R3", "R4", "R5"]
SIDE_ORDER = ["right", "left"]
SIDE_COLOR = {"right": "#31688e", "left": "#d17a22"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path,
        default=root / "data" / "features_dataset_corrected_unlabeled.csv",
        help="Corrected feature CSV used to map recordings to pseudonymous participants",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=root / "results" / "full_session_plots",
        help="Folder for participant figures and the HTML gallery",
    )
    parser.add_argument(
        "--signal-column", choices=["Voltage_mV", "ADC_Value"], default="Voltage_mV",
        help="Recorded, unfiltered signal column to plot",
    )
    args = parser.parse_args()

    frame = pd.read_csv(args.dataset)
    required = {"subject_id", "source_file", "repetition_id", "side"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Feature dataset is missing columns: {sorted(missing)}")
    records = frame[["subject_id", "source_file", "repetition_id", "side"]].drop_duplicates()
    records = records[records["repetition_id"].isin(R_ORDER)].copy()
    if records.empty:
        raise ValueError("No recordings with filename codes R0-R5 were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gallery: list[tuple[str, str, int]] = []
    manifest: list[dict[str, object]] = []
    for subject in sorted(records["subject_id"].unique()):
        subject_records = records[records["subject_id"] == subject].copy()
        reps = sorted(subject_records["repetition_id"].unique(), key=lambda rep: R_ORDER.index(rep))
        fig, axes = plt.subplots(
            len(reps), 2,
            figsize=(13, max(3.0, 2.25 * len(reps))),
            sharex=False,
            squeeze=False,
            constrained_layout=True,
        )
        count_files = 0
        durations: list[float] = []
        for row_index, rep in enumerate(reps):
            for col_index, side in enumerate(SIDE_ORDER):
                ax = axes[row_index, col_index]
                match = subject_records[
                    (subject_records["repetition_id"] == rep)
                    & (subject_records["side"] == side)
                ]
                if match.empty:
                    ax.text(0.5, 0.5, "No file", ha="center", va="center", transform=ax.transAxes, color="#777777")
                    ax.set_axis_off()
                    continue
                for _, item in match.iterrows():
                    raw_path = root / str(item["source_file"])
                    raw = pd.read_csv(raw_path, usecols=["Relative_Time_s", args.signal_column])
                    time = pd.to_numeric(raw["Relative_Time_s"], errors="raise").to_numpy(dtype=float).copy()
                    signal = pd.to_numeric(raw[args.signal_column], errors="raise").to_numpy(dtype=float).copy()
                    if len(time) != len(signal) or not len(time):
                        raise ValueError(f"Invalid signal arrays in {raw_path}")
                    dt = float(np.median(np.diff(time))) if len(time) > 1 else np.nan
                    gaps = np.flatnonzero(np.diff(time) > dt * 1.01) + 1 if np.isfinite(dt) else []
                    time = time - time[0]
                    time[gaps] = np.nan
                    signal[gaps] = np.nan
                    ax.plot(time, signal, color=SIDE_COLOR[side], linewidth=0.45, rasterized=True)
                    duration = float(pd.to_numeric(raw["Relative_Time_s"], errors="raise").iloc[-1] - pd.to_numeric(raw["Relative_Time_s"], errors="raise").iloc[0])
                    durations.append(duration)
                    count_files += 1
                    manifest.append({
                        "subject_id": subject,
                        "repetition_code": rep,
                        "side": side,
                        "duration_s": duration,
                        "sample_count": len(signal),
                    })
                ax.set_title(f"{rep} — {side}")
                ax.set_xlabel("Time within this file (s)")
                ax.set_ylabel(f"{args.signal_column}\n(unverified units)")
                ax.grid(alpha=0.18)

        mean_duration = float(np.mean(durations)) if durations else float("nan")
        fig.suptitle(
            f"{subject} — all available recordings | {count_files} files | "
            f"mean file duration {mean_duration:.1f} s\n"
            "Separate file clocks; no cross-file synchronization or fatigue labels",
            fontsize=11,
        )
        image_name = f"{subject}_full_recordings.png"
        fig.savefig(args.output_dir / image_name, dpi=130)
        plt.close(fig)
        gallery.append((subject, image_name, count_files))

    pd.DataFrame(manifest).to_csv(args.output_dir / "recording_manifest.csv", index=False)
    sections = []
    for subject, image_name, count in gallery:
        escaped_image = html.escape(image_name, quote=True)
        sections.append(
            f"<section><h2>{html.escape(subject)} — {count} archivos</h2>"
            f'<a href="{escaped_image}"><img loading="lazy" src="{escaped_image}" '
            f'alt="{html.escape(subject)} grabaciones completas" style="width:min(100%,1200px);height:auto"></a></section>'
        )
    document = """<!doctype html>
<html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Registros completos por participante</title>
<style>body{font:16px system-ui,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#222}
section{border-top:1px solid #bbb;margin-top:2rem;padding-top:1rem}img{border:1px solid #ddd}
.note{background:#fff7df;padding:1rem}</style></head><body>
<h1>Registros completos por participante</h1>
<p class="note">Cada panel conserva el archivo completo y su reloj relativo. Las filas muestran códigos del nombre (R0-R5), no clases confirmadas de repetición.
Los lados se grafican aparte. Las señales no se concatenan: los CSV reinician el tiempo y no hay sincronización global. El eje Y usa la columna registrada sin filtrado y sus unidades no están verificadas.</p>
""" + "\n".join(sections) + "\n</body></html>"
    (args.output_dir / "index.html").write_text(document, encoding="utf-8")
    print(f"Participants plotted: {len(gallery)}")
    print(f"Recordings included: {len(manifest)}")
    print(f"Gallery: {args.output_dir / 'index.html'}")
    print(f"Manifest: {args.output_dir / 'recording_manifest.csv'}")


if __name__ == "__main__":
    main()
