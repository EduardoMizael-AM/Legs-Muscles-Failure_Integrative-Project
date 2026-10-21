# Legs-Muscles-Failure_Integrative-Project
Análisis exploratorio de señales EMG del cuádriceps para estudiar posibles indicadores de fatiga muscular mediante procesamiento de señales, extracción de características y comparación de modelos. Los resultados son aproximaciones exploratorias, no una detección fisiológica validada.      10:21 p.m.
# EMG-Based Detection and Classification of Quadriceps Muscular Fatigue

The repository contains repeated EMG recordings in participant folders under `data/`. Raw files are expected to contain `Relative_Time_s`, `ADC_Value`, and `Voltage_mV` columns. Participant names in folder/file names are identifiable information; generated analysis tables use sorted pseudonymous subject IDs.

## Build the corrected exploratory window dataset

The previous `data/features_dataset.csv` must not be used: the old feature script selected the first numeric column (`Relative_Time_s`) as if it were the EMG signal and assigned labels from each recording's time quartile. The corrected pipeline reads `Voltage_mV` by default, derives the sample interval from timestamps, segments at timestamp discontinuities, and preserves source and window identifiers. It does not filter, interpolate, resample, or label the data.

From the repository root, run:

```powershell
python src/features.py
```

The default output is `data/features_dataset_corrected_unlabeled.csv`. The output path is protected against accidental overwrite. Use `--signal-column ADC_Value` to process the raw ADC column instead, or choose a different output with `--output PATH`. `--window-duration` and `--step-duration` are specified in seconds; defaults are 0.250 s and 0.125 s. These are analysis window settings, not documented acquisition settings.

The generated table contains RMS, MAV, variance, waveform length, plus exploratory full-band MDF/MNF columns computed without filtering. MDF/MNF are explicitly named as unfiltered; they are not classifier inputs or approved labels. `label_phase` remains blank. The repository includes exploratory proxy models and candidate repetition-stage analysis; neither is a validated physiological fatigue detector.

The pipeline and exploratory figures require Python with NumPy, pandas, and Matplotlib. Install them with `python -m pip install -r requirements.txt`. The corrected exploratory feature extraction does not need SciPy. A validated filter design and model training still require resolving acquisition parameters and adding their dependencies.

## Run exploratory analysis

After generating the corrected window dataset, run:

```powershell
python src/exploratory_analysis.py
```

The script saves descriptive signal and feature figures, per-record quality summaries, participant/side amplitude summaries, spectral trajectories, and a text report in `results/`. MDF/MNF are unfiltered exploratory measures, and the trajectory x-axis is time within each separate recording. It does not assign phases or interpret these measurements as confirmed physiological fatigue.

See `results/data_audit.txt` for the current audit and known data limitations.
See `results/protocol_data_status.md` for the comparison between the course-prescribed protocol, the local acquisition sketch, and the files actually available. The protocol guide is not treated as proof of how these recordings were acquired.

## Run the exploratory temporal-stage model

Install requirements with `python -m pip install -r requirements.txt`, then run from the repository root:

```powershell
python src/train_models.py
```

This prototype uses early/middle/late thirds of each CSV recording as a **temporal-stage proxy** and evaluates LDA and Random Forest with leave-one-subject-out validation. It uses RMS, MAV, variance, and waveform length; it excludes MDF/MNF from model inputs. These labels do not establish physiological fatigue, because recording boundaries and the meaning of each CSV are not confirmed. Results are written to `results/temporal_stage_model_metrics.txt` and `results/temporal_stage_oof_predictions.csv`.

## Plot every recording

To create a per-file view of the recorded signal, RMS, MDF, and MNF, run:

```powershell
python src/plot_recordings.py
```

The script reads the corrected window dataset and corresponding local raw CSVs. It writes one PNG per recording and a browsable `results/recording_plots/index.html` gallery. Plots use relative time within each file; MDF/MNF remain unfiltered exploratory measures, and the figures do not assign fatigue stages.

## Average amplitude by R-code

To calculate mean RMS amplitude over relative time for R1-R5, run:

```powershell
python src/average_repetitions.py
```

The script averages within recordings first and then across participants, keeping right and left sides separate. Time is binned in one-second intervals. It also reports mean file duration and mean RMS per R-code. Outputs are stored in `results/repetition_averages/`. R1-R5 are treated as filename codes; their experimental meaning is not confirmed. RMS is used instead of averaging signed raw samples, which could cancel across unsynchronized recordings.

## Plot complete recordings by participant

To create a full-duration raw trace for every available file, grouped into one figure per pseudonymous participant, run:

```powershell
python src/plot_full_sessions.py
```

Each figure has separate panels for each R-code and side; file clocks remain separate. The default signal is the unfiltered `Voltage_mV` column; use `--signal-column ADC_Value` to plot counts. The gallery and file manifest are saved to `results/full_session_plots/`.

## Explore repetition-based fatigue indicators

To detect candidate peaks in the smoothed RMS envelope and compare early, middle, and late candidate repetitions, run:

```powershell
python src/detect_fatigue_candidates.py
```

The script proposes an activity cutoff per file with Otsu's method, detects envelope peaks above that cutoff, and uses a provisional 2-second minimum peak spacing informed by the course-prescribed cadence. It exports per-repetition features, per-record counts/threshold diagnostics, paired early-to-late summaries, and diagnostic plots in `results/fatigue_candidates/`. Inspect the gallery before interpreting counts: peaks have not been checked against video or manual repetition annotations, the signal's RAW/envelope provenance is unresolved, and MDF/MNF remain unfiltered exploratory features. The output indicates candidate fatigue-associated trends, not validated physiological fatigue or a trained detector.

## Compare time-based and candidate-repetition stages

To compare the two operational stage definitions and evaluate how well LDA and Random Forest reconstruct each proxy label in participants held out from training, run:

```powershell
python src/compare_fatigue_methods.py
```

The script evaluates both label schemes on the same feature windows that could be matched to candidate repetition cycles (currently 82.9% coverage). It reports label agreement, Cohen's kappa, and leave-one-subject-out proxy-label metrics in `results/method_comparison/`. These metrics are **not fatigue-detection accuracy**: neither stage definition is independently annotated, and RMS is used both as a model feature and in candidate-peak detection. Independent repetition/fatigue labels are required before reporting physiological detection precision or validating a fatigue detector.

## Audit sustained terminal amplitude declines

To create provisional window labels for a sustained RMS decline after the last detected candidate peak, run:

```powershell
python src/label_terminal_decline.py
```

The current primary rule labels an active terminal tail as a candidate decline if it lasts at least 1 second, has a negative linear RMS trend, and ends at no more than 75% of its peak RMS. The 1-second duration was selected by the project team; the 75% ratio remains provisional. Only the active tail is labeled `candidate_terminal_decline`; earlier active windows, inactive windows, and uncertain windows stay distinct. Outputs are in `results/terminal_decline_candidates/`. A decline at the end of a recording can also indicate relaxation or stopping, so this label is operational and does not by itself prove fatigue onset.

To assess how sensitive candidate counts are to those rule settings and review all recordings, run:

```powershell
python src/analyze_terminal_threshold_sensitivity.py
```

The script varies minimum duration from 1 to 3 seconds and maximum final-active RMS / peak RMS from 0.50 to 0.90. The selected primary duration is 1 second. It writes a count heatmap, per-record decisions and a 140-record anonymized gallery in `results/terminal_decline_sensitivity/`. The tested grid is a sensitivity analysis, not a method for choosing thresholds by maximizing apparent model scores.

## Model the terminal-decline fatigue proxy

After generating the terminal-decline labels, run:

```powershell
python src/train_terminal_decline_model.py
```

This fits LDA and Random Forest with nested leave-one-participant-out validation, selecting probability thresholds on training participants only across several recall constraints (50%, 30%, 20%, 10%, 5%, 2%, and 1%) to show the precision–recall tradeoff. The stricter constraints test whether higher precision is possible while making the loss of detections explicit. It compares amplitude-derived, unfiltered spectral, and combined features. It includes only recordings with a qualifying terminal decline, comparing active pre-decline windows with candidate-decline windows; it excludes uncertain tails and does not count recordings without a qualifying tail as fatigue-negative. Metrics, out-of-fold predictions, per-record event results, and persistence summaries (1, 4, or 8 consecutive windows) are saved in `results/terminal_decline_model/`. They measure prediction of the operational label, not independent clinical or physiological ground truth. Overlapping windows are correlated; spectral measures are unfiltered, and the source signal's acquisition provenance is still unresolved.

## Compare proposed fatigue-label strategies

The previous terminal-amplitude rule and model outputs were copied before creating alternatives into `results/backup_20261018_pre_new_labels/`. This local backup includes the relevant source scripts and prior label/model outputs; it does not alter or publish the source data.

Create three separate target definitions and compare subject-held-out LDA and Random Forest models:

```powershell
python src/label_fatigue_strategies.py
python src/train_fatigue_strategy_models.py
```

The strategies are (1) active signal windows in the final third of recording time, (2) active windows assigned to automatically detected repetitions in the late third, and (3) the existing sustained terminal-amplitude-decline rule. Each is modeled separately. Outputs go to `results/proposed_fatigue_labels/` and `results/proposed_fatigue_models/`. These targets remain operational proxies: detected repetition cycles are unverified, and the final-third definitions assume progression in time/order corresponds to fatigue. Model results use leave-one-participant-out predictions and a fixed 0.5 threshold; they do not establish physiological validity.

For the expanded validation, run:

```powershell
python src/validate_fatigue_strategy_models.py
```

This creates a separate results directory, `results/proposed_fatigue_models_validated/`, preserving the initial model outputs. It adds a fold-specific majority-class baseline, a simple LDA baseline using RMS and MDF, class-wise and macro metrics, pooled confusion matrices, per-participant fold results, mean and standard deviation across participants, and a grouped inner search for Random Forest depth. The outer split still holds out one participant at a time; scaling is fitted only on each training fold, RF class weights handle imbalance, and RF search optimizes macro-F1 within training participants. The run uses a fixed 0.5 decision threshold. These scores evaluate the operational labels described above, not independent physiological fatigue truth.

To compare balancing choices without resampling the validation participants, run:

```powershell
python src/compare_class_balance_strategies.py
```

This compares Random Forest with balanced weights versus no class weights, and LDA with empirical versus equal class priors. It keeps the natural class prevalence in each held-out participant and writes results to `results/class_balance_comparison/`. It does not oversample overlapping windows.

## Primary operational fatigue label: terminal amplitude decline

The current analysis treats only the inherited sustained terminal RMS-amplitude decline as the positive fatigue proxy. Active windows before that decline, within the same recording that qualified, are the comparison class. Windows outside those periods and recordings without a qualifying decline remain unknown/excluded; they are not assumed to be non-fatigued. The last third of recording time and last third of candidate repetitions are retained only as descriptive approximations and do not define the target.

Regenerate the separate labels and validation as follows:

```powershell
python src/label_primary_amplitude_fatigue.py
python src/validate_primary_amplitude_fatigue.py
```

Labels and descriptive thirds are written to `results/primary_amplitude_fatigue_labels/`. Validation outputs go to `results/primary_amplitude_fatigue_validation/`, including `results.csv` (one model x held-out subject per row), pooled and fold-mean/SD model metrics, per-subject metrics, confusion matrices, nested RF parameters, feature-family ablation, feature importance, and out-of-fold predictions. The script includes majority and RMS+MDF LDA baselines; LDA equal/empirical priors; balanced/unweighted RF; nested participant-grouped tuning; accuracy, macro and per-class precision/recall/F1, ROC-AUC, PR-AUC, Cohen's kappa, and per-subject dispersion. Scaling and tuning remain inside training folds. Final spectral-only LDA and balanced RF artifacts plus metadata are saved in `final_models/` after CV and are not used to calculate the reported out-of-fold estimates.

The spectral-only family is the primary, less-circular detector because its features do not directly define the amplitude target. MDF/MNF are still unfiltered and the acquisition provenance is unresolved. Amplitude-only and combined results are explicitly exploratory and circular because the target itself was constructed from RMS decline. The inherited decline threshold is a provisional operational rule, not an independently verified physiological fatigue onset. The feature-family comparison serves as an available ablation; it does not replace future feature importance analysis across richer feature families or bilateral features. These results cannot establish clinical detection performance.
