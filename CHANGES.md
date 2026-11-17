# What changed in the corrected code, and why

Both scripts keep their command-line interface, their outputs and the verification
logic. The fixes below remove defects, close gaps against the design in the report
(Section 6) and make the code easier to read. Items marked **(results)** can change a
number compared with the run archived in `vv_results.json` / Appendix 12; the others
cannot.

## verify_pipeline.py

### Bugs fixed

1. **Pass/Fail badges rendered as `True` / `False`.** The HTML table renderer tested
   `isinstance(c, bool)`, which is false for numpy booleans. Every V1b, V2 (non-edge) and
   V5 row in the archived `vv_report.html` shows the raw word instead of the PASS/FAIL
   badge. Flags are now cast to plain `bool` where they are produced and the renderer
   accepts both types.
2. **Stage 6 table printed `[np.float64(-111.0), np.float64(76.6)]`.** The 95% limits
   were a list of numpy floats whose `repr` leaks into the report. Values are cast to
   float and the renderer prints ranges as `-111 to 76.6`.
3. **Crash paths guarded.** `crop_to_night` indexed `sl[0]` on a night with no scored
   sleep; `s6_montages` would call YASA with an EOG name that is absent from the
   recording and divide by zero when fewer than three nights had REM; the planted-label
   draw in S5 could produce a class too small for `StratifiedGroupKFold`
   (`n_splits=1` is invalid); the V3 confusion matrix divided by zero for a stage the
   technician never scored. Each case now degrades with a clear message or is skipped.
   Nights that cannot be cropped are skipped instead of aborting the whole run.
4. **k-means search excluded k = 6** **(results)**. `range(2, min(6, n - 1))` tests
   k = 2 to 5, while design Section 6.3.3 compares k = 2 to 6. The upper bound is now
   inclusive (`MIN_K`, `MAX_K` constants). The archived run chose k = 4 from k = 2 to 5;
   re-running may pick a different k and so shift the Stage 4 calibration figures
   slightly.
5. **S3 pass criterion did not match the V&V plan.** Table X and Appendix 12A say the
   test passes if balanced accuracy is at least 50% *and* the autoencoder beats PCA; the
   code only checked the first part. Both are now required (the archived run passes
   both, so the verdict is unchanged).
6. **Hard-coded "40 cases"** in the report text is now taken from the actual case count.
7. **scikit-learn version parsing** used `split(".")`, which breaks on pre-release
   strings such as `1.9.0rc1`; it now uses the leading integers.

### Structure and readability

- `v3_v4_staging` split into `v3_staging` and `v4_rem_latency`; a `_bootstrap_mean`
  helper replaces the nested closure.
- Magic numbers replaced by named constants (`EPOCHS_PER_HOUR`, `MIN_PER_EPOCH`,
  `SLEEP_CODES`, `LATENT_DIM`, `MIN_K`/`MAX_K`, output paths, `SEED`).
- One statement per line throughout (the original packed several `;`-joined statements
  per line), docstrings on every test, unused `sys` import removed, `as_py` JSON
  serialiser replaces an inline lambda, software versions and the seed are recorded in
  `meta`.
- Module docstring now states why `--n 20` gives 18 nights (Sleep-EDF has no first night
  for subjects 36, 39, 68, 69, 78 and 79) and includes `pooch`, which MNE needs to
  download the data.

## run_pipeline.py

### Bugs and broken logic fixed

1. **Test-cohort labels leaked into feature selection** **(results)**. Stage 4
   replicated the cluster-anxiety association on the held-out test set and, if it
   replicated, added `latent_high_cluster` as a Stage 5 feature. The test cohort
   therefore influenced the model before its "independent" evaluation. The design
   (Section 6, Figure 3) reserves a subset of the *development* cohort for Stage 4 and
   keeps the test cohort for final evaluation only. The cohort is now split 80/20
   (development/test, stratified), and development is split again into Stage 3
   training and a Stage 4 validation subset. The archived demonstration used a 70/30
   split, so its numbers (AUROC 0.34 vs 0.67, sensitivity 64%, specificity 55%) will not
   reproduce; see the corrections file for the re-run values.
2. **Split fraction did not match the design.** `test_size=0.3` replaced by 0.2
   (`TEST_FRACTION`), as Section 6 specifies an 80/20 allocation.
3. **Results were not reproducible run to run.** `stage3_discovery` received the
   training ids as a `set`, whose iteration order changes between Python processes;
   k-means initialisation and the inner validation split therefore varied. Ids are now
   ordered lists and every split is sorted.
4. **Epoch-feature columns could be misaligned.** Nights were truncated to the smallest
   number of feature columns (`common`), so a night without EOG (columns EEG x5, EMG)
   would have its EMG column read as "EOG slow" by the autoencoder. Columns are now
   selected by name (`common_feature_names`, `select_features`) and the model stores the
   names it was trained on; a new patient missing one of them gets a clear error.
5. **Stage 4 skipped the confounder adjustment** required by design Section 6.4.3. A
   logistic regression of anxiety on high-cluster membership adjusted for age, sex,
   AHI, cognitive status and medication is now fitted; a cluster is validated only if
   the training permutation test, the adjusted test and the replication all pass.
6. **Cross-validation could crash on small or imbalanced cohorts.** `StratifiedKFold(5)`
   and `LogisticRegressionCV(cv=5)` fail when a class has fewer than five members; the
   fold count now adapts (`_n_splits`).
7. **Band-pass filter could exceed Nyquist.** A 35 Hz high cut is invalid for recordings
   sampled between 50 and 70 Hz (the minimum allowed is 50 Hz); the cut is now capped at
   Nyquist minus 1 Hz.
8. **Inner validation share** for choosing autoencoder vs PCA was 20%; the design
   (6.3.1/6.3.2) specifies 10% (`INNER_VAL_FRACTION`).
9. **k-means search excluded k = 6** (same off-by-one as the verification script).
10. **Report sorting could raise `TypeError`** when two measures had equal contribution
    and the tuple comparison fell through to a `None` vs tuple comparison; sorting now
    uses an explicit key.
11. **Duplicated latent-profile code** in `report_cmd` (hand-written copy of the Stage 3
    encoder) is replaced by the shared `stage3_profile`, so new patients are processed
    exactly as the training cohort was.
12. **Smaller robustness fixes**: `edf_native_rates` returns `{}` instead of raising on
    an unparseable header; `hrv_nrem` passes an integer sampling rate to NeuroKit2 and
    returns NaN rather than crashing when no beat pairs survive filtering; a hypnogram CSV
    with a header row no longer shifts every epoch by one; a lights-off/lights-on window
    shorter than four hours is rejected with a clear message; participant ids are
    stripped of whitespace; `report` refuses an id that is not in the clinical CSV.

### Design alignment and readability

- PR-AUC (average precision) is now reported alongside AUROC, as Section 6.5.3 lists it.
- `results.json` records the cohort split sizes, the adjusted Stage 4 test and the
  software versions; `features.csv` labels each person's subset.
- The lights-off/lights-on logic is simplified (one branch instead of two assignments to
  `lights`), the shared adjusted-logistic helper `adjusted_logit` is used by Stages 2
  and 4, constants are named (`TEST_FRACTION`, `STAGE4_VAL_FRACTION`, `N_PERM`,
  `N_BOOT`, ...), and multi-statement lines are expanded.

## README.md

The original file was two READMEs concatenated without a line break ("...is
included.# sleeping-minds-verification"). Rewritten as one document with the install
line that includes `pooch`, the reason `--n 20` yields 18 nights, and the cohort
allocation used by the solution pipeline.

## Not changed on purpose

- Verification test ids, pass criteria, random seed and the order of random draws are
  unchanged, so a re-run on the same 18 nights reproduces Tests 1-7 and 10-11 exactly;
  only the k-means range (item 4 above) can move Tests 8-9.
- `class_weight="balanced"` is kept in both scripts. Note that design Section 6.5.2 says
  models would *initially* be trained without imbalance correction; this discrepancy is
  flagged in the report corrections rather than changed here, because the archived
  verification results were produced with balanced weighting.
- The autoencoder remains a fully connected network on per-epoch spectral features
  (scikit-learn), not the convolutional network on raw signals described in Section
  6.3.2; that is a planned extension, and the report corrections say so.

---
## Audit pass (17 Nov 2026) — changes on top of the version above
Run_pipeline.py
1. Stage 2 now follows design 6.2: Welch t-test (Mann-Whitney U when a group is skewed or has extreme outliers), then the Eq. 4 linear regression (feature ~ anxiety + covariates), Benjamini-Hochberg on both, FDR 5% (was 10%, and an adjusted logistic model with the feature as predictor).
2. Stage 4 reports Cramér's V, uses Fisher's exact test when K = 2 and expected counts < 5, and applies the 6.4.4 replication rule (same direction, and significant or consistent in magnitude when the validation subset is under 30 people).
3. Input checks: duplicate participant IDs, anxiety not 0/1, missing age/sex/AHI/cognitive status, test cohort with one class, too few people per class.
4. Report and summary HTML escape participant IDs; summary flags when Stage 5 ran in exploratory mode (no feature survived FDR).
Verify_pipeline.py
5. Tests the shipped code: REM latency uses run_pipeline's three-epoch rule (the old copy used the first REM epoch), plus new V6 unit checks (BH vs statsmodels, DeLong vs sklearn, chi-square vs scipy, 80/20 split, Eq. 4 recovery).
6. S6 pass is judged on the pre-specified EEG + EOG set, not the best of three.
7. Keep both files in one folder and run with plain `python` (not `-I`), since verify imports run_pipeline.
Not changed / still open: spindle and slow-wave features (spec 6.1.4, spindles 12–15 Hz), HRV beyond RMSSD, the CNN autoencoder, latent-feature tests (6.4.2), interpretation and channel ablation (6.4.5), random forest, Stage 6 reduced-channel run, class_weight="balanced", imputation fitted once on the development cohort rather than per CV fold.
vv_results.json, vv_report.html and vv_figures/ were regenerated with this version on 17 Nov 2026 (Google Colab, 18 Sleep-EDF nights).
REM latency vs technician improved from bias -89.7 min / ICC 0.07 to -34.9 min / ICC 0.69 (full montage) and -1.8 min / ICC 0.97 (EEG + EOG).
S5b (random-label AUROC) now reads 0.43 and fails the 0.44-0.56 band; S5a shows no leakage, so this reflects the n = 18 sample.
