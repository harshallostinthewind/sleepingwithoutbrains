# Sleeping Minds (BMET3921 Team 17)

Verification and solution code for a six-stage pipeline that looks for sleep-derived
anxiety indicators in overnight polysomnography (PSG). Research software only: it is
not a certified medical device and must not be used for patient care.

## Files

| File | Purpose | Report section |
|---|---|---|
| `verify_pipeline.py` | Verification tests of every pipeline stage on public, technician-scored Sleep-EDF recordings (no anxiety labels) | Section 7, Appendix 12 |
| `run_pipeline.py` | Solution pipeline (Stages 1-5 and the clinician report) for the clinical go/no-go study (Validation 4) | Appendix 13 |
| `vv_report.html`, `vv_results.json`, `vv_figures/` | Outputs of the last verification run (18 Sleep-EDF nights) | Appendix 12A-12C |
| `CHANGES.md` | What changed between code versions and what is still not implemented | Appendix 12D, 13 |

## Install

```bash
pip install mne yasa neurokit2 scikit-learn scipy statsmodels matplotlib edfio joblib pandas pooch
```

Tested with Python 3.13, YASA 0.8.1, MNE 1.13.2, NeuroKit2 0.3.0 and scikit-learn 1.6.1 (Google Colab)
and 1.9.1. Or `pip install -r requirements.txt`.

Keep both `.py` files in the same folder and run them with plain `python3` (not `python3 -I`),
because `verify_pipeline.py` imports the REM onset rule and statistics from `run_pipeline.py`.

## Verification (`verify_pipeline.py`)

```bash
python3 verify_pipeline.py --n 20        # the 18 nights reported in Appendix 12 (about 1.4 GB download)
python3 verify_pipeline.py --synthetic   # no download: dry run on simulated nights (do not report these numbers)
```

Data downloads automatically from PhysioNet into `~/mne_data`. Nights come from every
third Sleep-EDF subject; subjects without a first-night recording (36, 39, 68, 69, 78, 79)
are skipped with a warning, so `--n 20` yields 18 nights. Outputs: `vv_report.html`,
`vv_results.json` and one PNG per figure in `vv_figures/`. A fixed random seed (2026)
makes the numbers reproducible; with only 18 nights, small shifts between library versions
are possible (see Test 10).

Results of the last run (17 Nov 2026, Google Colab, 18 nights, about 45 min): 11/15 checks
passed, which is 8 of the 12 tests in the report. Failed: automated staging in the 60+ group (V3, report Test 4), REM latency agreement
with the full montage (V4, Test 5), sleep efficiency plausibility (V5, Test 6) and the
random-label classifier check at n = 18 (S5b, Test 10). See `vv_report.html`.

## Solution pipeline (`run_pipeline.py`)

Run on a labelled cohort:

```bash
python3 run_pipeline.py analyse --edf-dir data/ --clinical clinical.csv --out results/
```

Inputs: one EDF per person (`<id>.edf`), an optional clinician hypnogram
(`<id>_hypnogram.csv` with one stage per 30-s epoch, `<id>-Hypnogram.edf`, or stage
annotations inside the EDF), and `clinical.csv` with columns `id, anxiety (1/0), age,
sex (F/M), ahi, med_class, cognitive_status (MCI/SCI)` and optional `lights_off_s,
lights_on_s`. At least 30 people are needed, with at least 10 with and 10 without anxiety;
each person appears once in the CSV.

Cohort allocation follows the design: 80% development / 20% independent test,
stratified by anxiety. Stages 2-4 use the development cohort only (a subset of it is
reserved for the Stage 4 replication), and the test cohort is used once, for the final
Stage 5 evaluation.

Clinician report for one new person, using a saved model:

```bash
python3 run_pipeline.py report --model results/model.joblib --edf data/P001.edf --clinical clinical.csv --id P001 --out reports/
```

Demo on simulated data (no real data needed; effects are planted, so the numbers mean
nothing clinically):

```bash
python3 run_pipeline.py demo            # 120 simulated people, written to results/demo_data and results/demo_results
```

Channel names default to the AASM montage (`C4-M1`, `E1-M2`, `Chin1-Chin2`, `ECG`) and
can be changed with `--eeg --eog --emg --ecg`.
