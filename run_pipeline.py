"""
Sleeping Minds - solution pipeline (Stages 1-5 and the clinician report)
=======================================================================

BMET3921 Team 17. Research software for the Brain and Mind Centre go/no-go study
(Validation 4). It is NOT a certified medical device and must not be used for
patient care until Validation 4 has passed and the IEC 62304 / TGA steps are done.

What it does
------------
Given overnight PSG recordings (EDF) and a spreadsheet of each person's anxiety label
and clinical details, it runs the five analysis stages of the design:

  Stage 1  Feature extraction: input checks, channel quality rule, sleep staging
           (clinician hypnogram if supplied, otherwise automated EEG + EOG staging),
           main sleep period, apnoea-epoch exclusion, three-epoch REM onset rule,
           sleep, spectral and heart-rate features.
  Stage 2  Literature testing: each published biomarker vs anxiety, adjusted for
           age, sex, AHI, medication class and cognitive status (Welch / Mann-Whitney and the
           Eq. 4 regression, Benjamini-Hochberg at 5%).
  Stage 3  Discovery: autoencoder (PCA fallback) learns latent sleep features
           without labels; k-means groups people by latent profile.
  Stage 4  Validation of discoveries: chi-square permutation test of cluster vs
           anxiety in the Stage 3 training participants, confounder-adjusted
           logistic regression, then replication in the reserved validation subset.
  Stage 5  Elastic-net logistic regression on surviving features plus clinical
           covariates, compared with a clinical-only model on the independent test
           cohort (AUROC, PR-AUC, DeLong test, sensitivity/specificity) and the go/no-go call.
  Report   One clinician report (HTML) per person: probability with 95% interval,
           review threshold, the measures that drove it, the hypnogram with the
           two epochs that define REM latency, recording quality and limits.

Cohort allocation (design Section 6, Figure 3)
----------------------------------------------
  * 80% development cohort / 20% independent test cohort, stratified by anxiety.
  * Stages 2-4 use the development cohort only. A subset of it is reserved for the
    Stage 4 replication of discovered clusters, so the test cohort is never seen
    before the final Stage 5 evaluation.

Design changes from verification (report Appendix 12C) are built in:
  * clinician-scored hypnograms are used where available (automated staging only
    when none is supplied, with low-confidence epochs flagged);
  * EOG / EMG recorded below 50 Hz native sampling are excluded and staging uses
    EEG + EOG;
  * REM onset = first run of at least 3 consecutive REM epochs (1.5 min);
  * sleep efficiency uses lights-off / lights-on times when supplied;
  * the main overnight sleep period is isolated before analysis.

Inputs
------
  --edf-dir   folder of PSG recordings named <id>.edf
              optional clinician hypnogram per person, either
                <id>_hypnogram.csv  (one stage per 30-s epoch: W, N1, N2, N3, R)
                <id>-Hypnogram.edf  (EDF+ annotations, Sleep-EDF style)
              or sleep-stage annotations inside <id>.edf itself.
              Respiratory events (apnoea / hypopnoea) are read from EDF annotations.
  --clinical  CSV with one row per person and columns
                id, anxiety (1 = clinically significant symptoms, 0 = not),
                age, sex (F/M), ahi, med_class, cognitive_status (MCI/SCI)
              optional: lights_off_s, lights_on_s (seconds from recording start)

How to run
----------
  pip install mne yasa neurokit2 scikit-learn scipy statsmodels matplotlib edfio joblib pandas
  python run_pipeline.py analyse --edf-dir data/ --clinical clinical.csv --out results/
  python run_pipeline.py report  --model results/model.joblib --edf data/P001.edf \
                                 --clinical clinical.csv --id P001 --out reports/
  python run_pipeline.py demo    # builds a simulated cohort and runs everything

Channel names default to the AASM montage (C4-M1, E1-M2, Chin1-Chin2, ECG) and can be
changed with --eeg --eog --emg --ecg.
"""
import argparse
import base64
import html as html_lib
import io
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal as sps, stats  # noqa: E402
from sklearn import __version__ as SKLEARN_VERSION  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV  # noqa: E402
from sklearn.metrics import average_precision_score, silhouette_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
import joblib  # noqa: E402
import mne  # noqa: E402
import statsmodels.api as sm  # noqa: E402

mne.set_log_level("ERROR")

# ============================================================================ settings
EPOCH = 30                      # seconds per sleep epoch (AASM)
MIN_PER_EPOCH = EPOCH / 60
MIN_HOURS = 4.0                 # shortest usable night
MIN_SF = 50                     # minimum native sampling rate for any channel used
MIN_RECORDINGS = 30             # smallest cohort the analysis will run on
REM_RUN = 3                     # design change: REM onset needs >= 3 consecutive REM epochs
LOW_CONF = 0.50                 # automated-staging confidence below this is flagged
FDR_Q = 0.05                    # Stage 2 Benjamini-Hochberg false-discovery rate (design Section 6.2.1-6.2.2)
REVIEW_THRESHOLD = 0.40         # default review threshold (overwritten by Stage 5 tuning)
TEST_FRACTION = 0.20            # independent test cohort (design: 80% development / 20% test)
STAGE4_VAL_FRACTION = 0.25      # share of the development cohort reserved for Stage 4 replication
INNER_VAL_FRACTION = 0.10       # share of Stage 3 training people used to choose autoencoder vs PCA
MIN_K, MAX_K = 2, 6             # k-means cluster counts compared by silhouette (design Section 6.3.3)
LATENT_DIM = 6
N_PERM = 2000                   # permutations in the Stage 4 test
N_BOOT = 100                    # bootstrap refits for the per-patient 95% interval
SEED = 2026
SLEEP_CODES = [1, 2, 3, 4]
STAGES = ["W", "N1", "N2", "N3", "REM"]
STAGE_MAP = {"W": 0, "WAKE": 0, "0": 0, "N1": 1, "1": 1, "N2": 2, "2": 2, "N3": 3, "3": 3,
             "N4": 3, "4": 3, "R": 4, "REM": 4, "5": 4,
             "SLEEP STAGE W": 0, "SLEEP STAGE 1": 1, "SLEEP STAGE 2": 2, "SLEEP STAGE 3": 3,
             "SLEEP STAGE 4": 3, "SLEEP STAGE R": 4}
RESP_WORDS = ("apnea", "apnoea", "hypopnea", "hypopnoea")
CLINICAL = ["age", "sex_m", "ahi", "cog_mci"]          # + medication dummies, added at run time

# Literature (Tier 1-3) biomarkers tested in Stage 2, with the direction reported in prior art
LITERATURE = {
    "SOL": ("Sleep onset latency", "min", +1),
    "WASO": ("Wake after sleep onset", "min", +1),
    "SE": ("Sleep efficiency", "%", -1),
    "REMLAT": ("REM latency", "min", +1),
    "REMp": ("REM sleep", "%", +1),
    "N3p": ("Deep sleep (N3)", "%", -1),
    "NREM_beta": ("NREM beta power (16-30 Hz)", "rel.", +1),
    "NREM_sigma": ("NREM sigma power (12-16 Hz)", "rel.", +1),
    "REM_EOG": ("REM eye-movement activity", "log uV2", +1),
    "REM_EMG": ("REM chin muscle tone", "log uV2", +1),
    "HRV_RMSSD": ("HRV (RMSSD) in NREM", "ms", -1),
}


def log(msg):
    print(msg, flush=True)


class PipelineError(Exception):
    """Raised with a clear message when a recording cannot be processed (SPR3.1)."""


def _sklearn_needs_penalty():
    """scikit-learn < 1.8 needs penalty='elasticnet'; later versions infer it from l1_ratio."""
    major, minor = (int(v) for v in re.findall(r"\d+", SKLEARN_VERSION)[:2])
    return (major, minor) < (1, 8)


# ============================================================================ Stage 1
def edf_native_rates(path):
    """Each channel's ORIGINAL sampling rate, read from the EDF header (MNE resamples on load).
    Returns {} when the header cannot be parsed, so callers fall back to the loaded rate."""
    try:
        with open(path, "rb") as f:
            hdr = f.read(256)
            ns = int(hdr[252:256].decode().strip())
            dur = float(hdr[244:252].decode().strip()) or 1.0
            rest = f.read(ns * 256)
        labels = [rest[i * 16:(i + 1) * 16].decode("latin1").strip() for i in range(ns)]
        off = ns * (16 + 80 + 8 + 8 + 8 + 8 + 8 + 80)
        nsamp = [int(rest[off + i * 8: off + (i + 1) * 8].decode().strip()) for i in range(ns)]
        return {lab: n / dur for lab, n in zip(labels, nsamp)}
    except Exception:
        return {}


def to_stage(v):
    return STAGE_MAP.get(str(v).strip().upper(), -1)


def hypnogram_from_annotations(annots, n_ep, t0=0.0):
    """30-s hypnogram from sleep-stage annotations; None when the annotations hold no stages."""
    hyp = -np.ones(n_ep, int)
    found = False
    for a in annots:
        st = to_stage(a["description"])
        if st < 0:
            continue
        found = True
        s = int(round((a["onset"] - t0) / EPOCH))
        d = max(1, int(round(a["duration"] / EPOCH)))
        hyp[max(s, 0): s + d] = st
    return hyp if found else None


def load_clinician_hypnogram(edf_path, raw, n_ep):
    """Design change: use the clinician-scored hypnogram whenever one exists."""
    pid = edf_path.stem
    csv = edf_path.with_name(f"{pid}_hypnogram.csv")
    if csv.exists():
        vals = pd.read_csv(csv, header=None).iloc[:, 0].tolist()
        if vals and to_stage(vals[0]) < 0 and len(vals) > 1 and to_stage(vals[1]) >= 0:
            vals = vals[1:]                                   # tolerate a header row
        hyp = np.array([to_stage(v) for v in vals])[:n_ep]
        return np.pad(hyp, (0, n_ep - len(hyp)), constant_values=-1), f"clinician-scored ({csv.name})"
    hfile = edf_path.with_name(f"{pid}-Hypnogram.edf")
    if hfile.exists():
        h = hypnogram_from_annotations(mne.read_annotations(hfile), n_ep)
        if h is not None:
            return h, f"clinician-scored ({hfile.name})"
    h = hypnogram_from_annotations(raw.annotations, n_ep)
    if h is not None:
        return h, "clinician-scored (annotations in EDF)"
    return None, None


def stage_automated(raw, ch):
    """YASA staging with EEG + EOG (EMG added only if it passed the quality rule).
    Returns the integer hypnogram and the per-epoch confidence (max class probability)."""
    import yasa
    sls = yasa.SleepStaging(raw, eeg_name=ch["eeg"], eog_name=ch.get("eog"), emg_name=ch.get("emg"))
    pred = sls.predict()
    labels = pred.hypno.astype(str).tolist() if hasattr(pred, "hypno") else list(pred)
    proba = getattr(pred, "proba", None)                       # YASA >= 0.6.4 returns a Hypnogram with .proba
    if proba is None and hasattr(sls, "predict_proba"):        # older YASA versions
        proba = sls.predict_proba()
    conf = proba.max(axis=1).to_numpy() if proba is not None else np.ones(len(labels))
    return np.array([to_stage(v) for v in labels]), conf


def main_sleep_period(hyp, max_gap=120):
    """Longest sleep run with no wake gap over `max_gap` epochs (60 min); excludes daytime naps."""
    sl = np.where(np.isin(hyp, SLEEP_CODES))[0]
    if len(sl) == 0:
        raise PipelineError("No sleep was detected in this recording.")
    brk = np.where(np.diff(sl) > max_gap)[0]
    starts, ends = np.r_[sl[0], sl[brk + 1]], np.r_[sl[brk], sl[-1]]
    i = int(np.argmax(ends - starts))
    return int(starts[i]), int(ends[i])


def rem_onset_epoch(hyp, run=REM_RUN):
    """Design change: first epoch of the first run of >= `run` consecutive REM epochs."""
    is_rem = (np.asarray(hyp) == 4).astype(int)
    if is_rem.sum() == 0:
        return None
    win = np.convolve(is_rem, np.ones(run, int), mode="valid")
    hits = np.where(win == run)[0]
    return int(hits[0]) if len(hits) else None


def sleep_measures(hyp, lights=None):
    """Sleep continuity and architecture, REM latency with the robust onset rule.
    `lights` = (lights-off epoch, lights-on epoch) in the hypnogram's own indexing."""
    hyp = np.asarray(hyp)
    sleep = np.isin(hyp, SLEEP_CODES)
    idx = np.where(sleep)[0]
    if lights is not None:                       # design change: SE from lights-off
        a, b = lights
        tib = (b - a) * MIN_PER_EPOCH
        sol = (idx[0] - a) * MIN_PER_EPOCH if len(idx) else np.nan
    else:
        tib = (hyp >= 0).sum() * MIN_PER_EPOCH
        sol = idx[0] * MIN_PER_EPOCH if len(idx) else np.nan
    tst = sleep.sum() * MIN_PER_EPOCH
    onset = int(idx[0]) if len(idx) else None
    rem0 = rem_onset_epoch(hyp)
    remlat = (rem0 - onset) * MIN_PER_EPOCH if (rem0 is not None and onset is not None and rem0 >= onset) else np.nan
    spt = hyp[idx[0]: idx[-1] + 1] if len(idx) else np.array([])
    return dict(TST=tst, SE=100 * tst / max(tib, 1e-6), SOL=sol, WASO=float(np.sum(spt == 0) * MIN_PER_EPOCH),
                N1p=100 * np.mean(hyp[sleep] == 1) if tst else np.nan,
                N2p=100 * np.mean(hyp[sleep] == 2) if tst else np.nan,
                N3p=100 * np.mean(hyp[sleep] == 3) if tst else np.nan,
                REMp=100 * np.mean(hyp[sleep] == 4) if tst else np.nan,
                REMLAT=remlat), dict(onset=onset, rem_onset=rem0)


def respiratory_epochs(annots, n_ep, t0=0.0):
    """Epochs that overlap a scored apnoea/hypopnoea event (excluded from spectral/HRV features)."""
    bad = np.zeros(n_ep, bool)
    for a in annots:
        if any(w in a["description"].lower() for w in RESP_WORDS):
            s = int((a["onset"] - t0) // EPOCH)
            e = int((a["onset"] - t0 + a["duration"]) // EPOCH)
            bad[max(s, 0): min(e + 1, n_ep)] = True
    return bad


def band_rel(data, sf, lo, hi):
    f, p = sps.welch(data, fs=sf, nperseg=int(sf * 4), axis=-1)
    tot = p[..., (f >= 0.5) & (f < 30)].sum(-1)
    return p[..., (f >= lo) & (f < hi)].sum(-1) / np.maximum(tot, 1e-12)


def epoch_matrix(raw, name, n_ep):
    """Signal `name` as an (epochs x samples) matrix in microvolts."""
    sf = raw.info["sfreq"]
    spe = int(sf * EPOCH)
    x = raw.get_data(picks=[name])[0][: n_ep * spe] * 1e6
    return x.reshape(n_ep, spe), sf


def epoch_spectra(raw, ch, n_ep):
    """Per-epoch log band powers on every usable signal (Stage 3 input)."""
    cols, names = [], []
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
    X, sf = epoch_matrix(raw, ch["eeg"], n_ep)
    f, p = sps.welch(X, fs=sf, nperseg=int(sf * 4), axis=-1)
    for b, (lo, hi) in bands.items():
        cols.append(np.log10(p[:, (f >= lo) & (f < hi)].mean(1) + 1e-6))
        names.append(f"EEG {b}")
    if ch.get("eog"):
        X, sf = epoch_matrix(raw, ch["eog"], n_ep)
        f, p = sps.welch(X, fs=sf, nperseg=int(sf * 4), axis=-1)
        for b, (lo, hi) in {"slow": (0.3, 2), "fast": (2, 8)}.items():
            cols.append(np.log10(p[:, (f >= lo) & (f < hi)].mean(1) + 1e-6))
            names.append(f"EOG {b}")
    if ch.get("emg"):
        X, _ = epoch_matrix(raw, ch["emg"], n_ep)
        cols.append(np.log10(X.var(1) + 1e-6))
        names.append("EMG power")
    return np.column_stack(cols), names


def hrv_nrem(raw, ecg, hyp, usable):
    """RMSSD (ms) from R-R intervals whose two beats both fall in usable NREM epochs."""
    import neurokit2 as nk
    sf = int(round(raw.info["sfreq"]))
    sig = raw.get_data(picks=[ecg])[0]
    _, info = nk.ecg_peaks(nk.ecg_clean(sig, sampling_rate=sf), sampling_rate=sf)
    pk = np.asarray(info["ECG_R_Peaks"])
    if len(pk) < 50:
        return np.nan
    ep = np.clip((pk / sf // EPOCH).astype(int), 0, len(hyp) - 1)
    ok = np.isin(hyp[ep], [2, 3]) & usable[ep]
    rr = np.diff(pk) / sf * 1000
    pair = ok[1:] & ok[:-1] & (rr > 300) & (rr < 2000)
    rr = rr[pair]
    if len(rr) < 30:
        return np.nan
    d = np.diff(rr)
    d = d[np.abs(d) < 200]                     # drop ectopic jumps
    return float(np.sqrt(np.mean(d ** 2))) if len(d) else np.nan


def stage1_night(edf_path, ch_req, clin_row=None):
    """Run Stage 1 on one recording. Returns features, hypnogram, epoch features and QC."""
    edf_path = Path(edf_path)
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, stim_channel=None)
    except Exception:
        raise PipelineError(f"{edf_path.name}: file could not be read as EDF/EDF+.")
    rates = edf_native_rates(edf_path)
    qc = dict(excluded=[], unavailable=[])
    if ch_req["eeg"] not in raw.ch_names:
        raise PipelineError(f"{edf_path.name}: required EEG channel '{ch_req['eeg']}' is missing.")
    if rates.get(ch_req["eeg"], raw.info["sfreq"]) < MIN_SF:
        raise PipelineError(f"{edf_path.name}: EEG sampled below {MIN_SF} Hz.")
    hours = raw.times[-1] / 3600
    if hours < MIN_HOURS:
        raise PipelineError(f"{edf_path.name}: recording is {hours:.1f} h; at least {MIN_HOURS:.0f} h is required.")

    ch = {"eeg": ch_req["eeg"]}
    for k in ("eog", "emg", "ecg"):                 # design change: native-rate rule
        name = ch_req.get(k)
        if not name or name not in raw.ch_names:
            qc["unavailable"].append(f"{k.upper()} missing")
            continue
        if rates.get(name, raw.info["sfreq"]) < MIN_SF:
            qc["excluded"].append(f"{name} (native {rates[name]:g} Hz)")
            continue
        ch[k] = name
    raw.pick(list(ch.values()))
    nyq = raw.info["sfreq"] / 2
    raw.filter(0.3, min(35, nyq - 1), picks=[ch["eeg"]] + ([ch["eog"]] if "eog" in ch else []))
    if "emg" in ch:
        raw.filter(10, min(45, nyq - 1), picks=[ch["emg"]])

    n_ep = int(raw.times[-1] // EPOCH)
    hyp, source = load_clinician_hypnogram(edf_path, raw, n_ep)
    conf = np.ones(n_ep)
    if hyp is None:
        stage_ch = {k: ch.get(k) for k in ("eeg", "eog", "emg")}
        hyp, conf = stage_automated(raw, stage_ch)
        hyp, conf = hyp[:n_ep], conf[:n_ep]
        source = "automated (YASA, " + " + ".join(k.upper() for k in ("eeg", "eog", "emg") if stage_ch.get(k)) + ")"
    qc["staging_source"] = source
    qc["clinician_scored"] = not source.startswith("automated")

    # Analysis window: lights-off to lights-on if supplied, otherwise the main sleep period +/- 30 min
    lights = None
    has_lights = (clin_row is not None and pd.notna(clin_row.get("lights_off_s", np.nan))
                  and pd.notna(clin_row.get("lights_on_s", np.nan)))
    if has_lights:
        s0 = max(0, int(clin_row["lights_off_s"] // EPOCH))
        s1 = min(n_ep, int(clin_row["lights_on_s"] // EPOCH))
        if (s1 - s0) * EPOCH < MIN_HOURS * 3600:
            raise PipelineError(f"{edf_path.name}: lights-off to lights-on is shorter than {MIN_HOURS:.0f} h.")
        qc["window"] = "lights-off to lights-on"
        lights = (0, s1 - s0)
    else:
        a, b = main_sleep_period(hyp)
        s0, s1 = max(0, a - 60), min(n_ep, b + 61)
        qc["window"] = "main sleep period +/- 30 min (no lights-off markers)"
    resp = respiratory_epochs(raw.annotations, n_ep)
    hyp, conf, resp = hyp[s0:s1], conf[s0:s1], resp[s0:s1]
    raw.crop(tmin=s0 * EPOCH, tmax=s1 * EPOCH - 1 / raw.info["sfreq"])
    n_ep = s1 - s0

    meas, marks = sleep_measures(hyp, lights)
    usable = (hyp >= 0) & ~resp & (conf >= LOW_CONF)
    qc.update(n_epochs=int(n_ep), usable_pct=round(100 * usable.mean(), 1),
              resp_excluded_pct=round(100 * resp.mean(), 1),
              low_conf_pct=round(100 * (conf < LOW_CONF).mean(), 1),
              mean_conf=None if qc["clinician_scored"] else round(float(conf.mean()), 2))

    # Spectral micro-architecture on usable NREM / REM epochs
    E, sf = epoch_matrix(raw, ch["eeg"], n_ep)
    nrem = usable & np.isin(hyp, [2, 3])
    rem = usable & (hyp == 4)
    meas["NREM_beta"] = float(band_rel(E[nrem], sf, 16, 30).mean()) if nrem.sum() > 10 else np.nan
    meas["NREM_sigma"] = float(band_rel(E[nrem], sf, 12, 16).mean()) if nrem.sum() > 10 else np.nan
    if "eog" in ch and rem.sum() > 5:
        O, _ = epoch_matrix(raw, ch["eog"], n_ep)
        meas["REM_EOG"] = float(np.log10(O[rem].var(1).mean() + 1e-6))
    else:
        meas["REM_EOG"] = np.nan
    if "emg" in ch and rem.sum() > 5:
        M, _ = epoch_matrix(raw, ch["emg"], n_ep)
        meas["REM_EMG"] = float(np.log10(M[rem].var(1).mean() + 1e-6))
    else:
        meas["REM_EMG"] = np.nan
    meas["HRV_RMSSD"] = hrv_nrem(raw, ch["ecg"], hyp, usable) if "ecg" in ch else np.nan
    for k in LITERATURE:
        if k not in meas or not np.isfinite(meas[k]):
            qc["unavailable"].append(f"{LITERATURE[k][0]} could not be computed")

    X_ep, ep_names = epoch_spectra(raw, ch, n_ep)
    return dict(id=edf_path.stem, features=meas, hyp=hyp, conf=conf, usable=usable, resp=resp,
                marks=marks, X_ep=X_ep, ep_names=ep_names, qc=qc, signals=sorted(ch.keys()))


# ============================================================================ clinical table
def load_clinical(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    need = {"id", "age", "sex", "ahi", "med_class", "cognitive_status"}
    miss = need - set(df.columns)
    if miss:
        raise PipelineError(f"Clinical CSV is missing columns: {', '.join(sorted(miss))}")
    df["id"] = df["id"].astype(str).str.strip()
    dup = df["id"][df["id"].duplicated()].unique().tolist()
    if dup:                                                   # design: one recording per participant
        raise PipelineError(f"Clinical CSV lists these participants more than once: {', '.join(dup[:10])}.")
    bad = [c for c in ("age", "sex", "ahi", "cognitive_status") if df[c].isna().any()]
    if bad:
        raise PipelineError(f"Clinical CSV has missing values in: {', '.join(bad)}. Complete or remove those rows.")
    for c in ("age", "ahi"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
        if df[c].isna().any():
            raise PipelineError(f"Clinical CSV column '{c}' must be numeric.")
    if "anxiety" in df.columns:
        a = pd.to_numeric(df["anxiety"], errors="coerce")
        if a.isna().any() or not a.isin([0, 1]).all():
            raise PipelineError("Clinical CSV column 'anxiety' must contain only 0 or 1 for every row.")
        df["anxiety"] = a.astype(int)
    df["sex_m"] = (df["sex"].astype(str).str.upper().str[0] == "M").astype(int)
    df["cog_mci"] = (df["cognitive_status"].astype(str).str.upper() == "MCI").astype(int)
    df["med_class"] = df["med_class"].fillna("none").astype(str).str.lower()
    return df.set_index("id")


def clinical_matrix(df, med_levels):
    X = df[CLINICAL].astype(float).copy()
    for m in med_levels:
        X[f"med_{m}"] = (df["med_class"] == m).astype(float)
    return X


def stratified_split(ids, y, fraction, seed):
    """Stratified split that falls back to a plain split when a class is too small to stratify."""
    try:
        return train_test_split(list(ids), test_size=fraction, stratify=np.asarray(y), random_state=seed)
    except ValueError:
        return train_test_split(list(ids), test_size=fraction, random_state=seed)


# ============================================================================ Stage 2
def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(na + nb - 2, 1))
    g = (a.mean() - b.mean()) / sp if sp > 0 else 0.0
    return g * (1 - 3 / (4 * (na + nb) - 9))


def bh_fdr(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def adjusted_logit(x, C, y):
    """Logistic regression of anxiety on one standardised predictor plus the clinical covariates.
    Used by Stage 4 for cluster membership (design 6.4.3). Returns (OR, CI low, CI high, p); NaNs on failure."""
    Cv = np.asarray(C, float)
    Cv = Cv[:, Cv.std(0) > 0]                              # drop covariates with no variation
    cols = [np.asarray(x, float)] + ([StandardScaler().fit_transform(Cv)] if Cv.shape[1] else [])
    Xd = sm.add_constant(np.column_stack(cols), has_constant="add")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sm.Logit(np.asarray(y, int), Xd).fit(disp=0, maxiter=200)
        ci = np.exp(np.asarray(fit.conf_int())[1])
        return float(np.exp(fit.params[1])), float(ci[0]), float(ci[1]), float(fit.pvalues[1])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def adjusted_ols(x, C, y):
    """Design Eq. 4: feature = b0 + b1*anxiety + covariates. Returns (b1, CI low, CI high, p) for anxiety."""
    Cv = np.asarray(C, float)
    Cv = Cv[:, Cv.std(0) > 0]
    cols = [np.asarray(y, float)] + ([Cv] if Cv.shape[1] else [])
    Xd = sm.add_constant(np.column_stack(cols), has_constant="add")
    try:
        fit = sm.OLS(np.asarray(x, float), Xd).fit()
        ci = np.asarray(fit.conf_int())[1]
        return float(fit.params[1]), float(ci[0]), float(ci[1]), float(fit.pvalues[1])
    except Exception:
        return np.nan, np.nan, np.nan, np.nan


def group_test(a, b):
    """Design 6.2.1: Welch's t-test; Mann-Whitney U instead when either group is clearly skewed
    (|skewness| > 1) or has extreme outliers (beyond 3 IQR). Returns (test name, p, mean diff, CI low, CI high)."""
    def unsuitable(v):
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        return (len(v) > 2 and abs(stats.skew(v)) > 1) or (iqr > 0 and np.any((v < q1 - 3 * iqr) | (v > q3 + 3 * iqr)))
    t = stats.ttest_ind(a, b, equal_var=False)
    diff = float(a.mean() - b.mean())
    ci = t.confidence_interval(0.95)
    lo, hi = float(ci.low), float(ci.high)
    if unsuitable(a) or unsuitable(b):
        return "Mann-Whitney U", float(stats.mannwhitneyu(a, b).pvalue), diff, lo, hi
    return "Welch t", float(t.pvalue), diff, lo, hi


def stage2_literature(F, C, y):
    """Each literature biomarker vs anxiety (design 6.2): group comparison (Welch / Mann-Whitney) and the
    covariate-adjusted regression of Eq. 4, each with Benjamini-Hochberg control. A feature is carried
    forward when its adjusted anxiety coefficient survives BH at FDR_Q."""
    rows = []
    for k, (label, unit, expect) in LITERATURE.items():
        x = F[k].to_numpy(float)
        ok = np.isfinite(x)
        if ok.sum() < 20 or min((y[ok] == 1).sum(), (y[ok] == 0).sum()) < 3:
            rows.append(dict(feature=k, label=label, n=int(ok.sum()), note="too few values"))
            continue
        a, b = x[ok & (y == 1)], x[ok & (y == 0)]
        test, p_raw, diff, d_lo, d_hi = group_test(a, b)
        beta, b_lo, b_hi, p_adj = adjusted_ols(x[ok], C.to_numpy(float)[ok], y[ok])
        sd = x[ok].std(ddof=1)
        rows.append(dict(feature=k, label=label, unit=unit, n=int(ok.sum()), expected=("higher" if expect > 0 else "lower"),
                         mean_anx=float(a.mean()), mean_non=float(b.mean()), hedges_g=float(hedges_g(a, b)),
                         test=test, raw_p=p_raw, diff=diff, diff_lo=d_lo, diff_hi=d_hi,
                         beta=beta, beta_lo=b_lo, beta_hi=b_hi, std_beta=(beta / sd if sd > 0 else np.nan), adj_p=p_adj))
    T = pd.DataFrame(rows)
    for pcol, qcol in (("raw_p", "q_raw"), ("adj_p", "q")):
        T[qcol] = np.nan
        has = T[pcol].notna() if pcol in T else pd.Series(False, index=T.index)
        if has.any():
            T.loc[has, qcol] = bh_fdr(T.loc[has, pcol])
    T["carried_forward"] = T["q"] < FDR_Q
    return T


# ============================================================================ Stage 3
def common_feature_names(nights):
    """Epoch-feature columns available in every recording (a night without EOG lacks the EOG columns)."""
    return [nm for nm in nights[0]["ep_names"] if all(nm in n["ep_names"] for n in nights)]


def select_features(night, names):
    """Epoch-feature columns of one night, selected by name so that columns always line up."""
    try:
        idx = [night["ep_names"].index(nm) for nm in names]
    except ValueError as e:
        raise PipelineError(f"{night['id']}: cannot compute the latent profile, a required signal feature is missing ({e}).")
    return night["X_ep"][:, idx]


def stage3_encode(model, X):
    Z = model["scaler"].transform(X)
    if model["use_ae"]:
        ae = model["ae"]
        a = np.tanh(Z @ ae.coefs_[0] + ae.intercepts_[0])
        return np.tanh(a @ ae.coefs_[1] + ae.intercepts_[1])
    return model["pca"].transform(Z)


def stage3_profile(model, night):
    """Per-person latent profile: mean latent vector in N2, N3 and REM (usable epochs only)."""
    L = stage3_encode(model, select_features(night, model["features"]))
    base = night["usable"] if night["usable"].sum() >= 5 else np.ones(len(L), bool)
    row = []
    for st in (2, 3, 4):
        m = night["usable"] & (night["hyp"] == st)
        row.append(L[m].mean(0) if m.sum() >= 5 else L[base].mean(0))
    return np.concatenate(row)


def stage3_discovery(nights, train_ids, latent_dim=LATENT_DIM):
    """Autoencoder on per-epoch features of training nights; PCA if the AE does not beat it.
    `train_ids` must be an ordered sequence so the result is reproducible run to run."""
    rng = np.random.default_rng(SEED)
    train_ids = list(train_ids)
    features = common_feature_names(nights)
    by_id = {n["id"]: n for n in nights}
    Xtr, gtr = [], []
    for i, pid in enumerate(train_ids):
        n = by_id[pid]
        m = n["usable"]
        Xtr.append(select_features(n, features)[m])
        gtr.append(np.full(m.sum(), i))
    Xtr, gtr = np.vstack(Xtr), np.concatenate(gtr)
    sc = StandardScaler().fit(Xtr)
    Z = sc.transform(Xtr)
    subj = np.unique(gtr)
    inner = rng.choice(subj, max(2, int(round(INNER_VAL_FRACTION * len(subj)))), replace=False)
    te = np.isin(gtr, inner)

    def new_ae():
        return MLPRegressor(hidden_layer_sizes=(32, latent_dim, 32), activation="tanh", max_iter=300,
                            early_stopping=True, random_state=0)

    ae = new_ae().fit(Z[~te], Z[~te])
    pca = PCA(latent_dim).fit(Z[~te])
    mse_ae = float(np.mean((ae.predict(Z[te]) - Z[te]) ** 2))
    mse_pca = float(np.mean((pca.inverse_transform(pca.transform(Z[te])) - Z[te]) ** 2))
    use_ae = mse_ae <= mse_pca
    if use_ae:
        ae = new_ae().fit(Z, Z)
    else:
        pca = PCA(latent_dim).fit(Z)
    model = dict(scaler=sc, ae=ae if use_ae else None, pca=None if use_ae else pca, use_ae=use_ae,
                 features=features, latent_dim=latent_dim)

    prof = {n["id"]: stage3_profile(model, n) for n in nights}
    P_tr = np.array([prof[i] for i in train_ids])
    psc = StandardScaler().fit(P_tr)
    Ptr = psc.transform(P_tr)
    k_values = range(MIN_K, min(MAX_K, len(train_ids) - 1) + 1)     # silhouette needs 2 <= k <= n - 1
    sil = {k: float(silhouette_score(Ptr, KMeans(k, n_init=20, random_state=0).fit_predict(Ptr))) for k in k_values}
    k = max(sil, key=sil.get)
    km = KMeans(k, n_init=20, random_state=0).fit(Ptr)
    clusters = {pid: int(km.predict(psc.transform(prof[pid][None]))[0]) for pid in prof}
    info = dict(method="autoencoder" if use_ae else "PCA fallback", mse_ae=round(mse_ae, 4),
                mse_pca=round(mse_pca, 4), k=int(k), silhouette={str(a): round(b, 3) for a, b in sil.items()},
                n_training=len(train_ids), features=features)
    model.update(prof_scaler=psc, kmeans=km)
    return clusters, info, model


# ============================================================================ Stage 4
def chi2_stat(c, y, k):
    tab = np.zeros((k, 2))
    np.add.at(tab, (c, y.astype(int)), 1)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nansum((tab - exp) ** 2 / np.where(exp == 0, np.nan, exp))), tab


def perm_test(c, y, k, n=N_PERM):
    rng = np.random.default_rng(SEED)
    obs, tab = chi2_stat(c, y, k)
    null = np.array([chi2_stat(c, rng.permutation(y), k)[0] for _ in range(n)])
    return (np.sum(null >= obs) + 1) / (n + 1), obs, tab      # Phipson & Smyth correction


def stage4_validate(clusters, ids_tr, ids_val, y_tr, y_val, C_tr, k):
    """Cluster vs anxiety in the Stage 3 training people (permutation chi-square), confounder-adjusted
    logistic regression (design 6.4.3), then one-sided replication in the reserved validation subset (6.4.4)."""
    c_tr = np.array([clusters[i] for i in ids_tr])
    c_val = np.array([clusters[i] for i in ids_val])
    y_tr, y_val = np.asarray(y_tr, int), np.asarray(y_val, int)
    p_tr, chi_tr, tab_tr = perm_test(c_tr, y_tr, k)
    expc = tab_tr.sum(1, keepdims=True) * tab_tr.sum(0, keepdims=True) / max(tab_tr.sum(), 1)
    exact = "permutation chi-square (equivalent to Fisher-Freeman-Halton when counts are small)"
    if k == 2 and (expc < 5).any() and (tab_tr.sum(0) > 0).all() and (tab_tr.sum(1) > 0).all():
        p_tr, exact = float(stats.fisher_exact(tab_tr).pvalue), "Fisher exact (K = 2, expected count < 5)"   # design 6.4.1
    rate = tab_tr[:, 1] / np.maximum(tab_tr.sum(1), 1)
    hi = int(np.argmax(rate))                                 # the "high-anxiety" cluster
    adj_or, _, _, adj_p = adjusted_logit((c_tr == hi).astype(float), C_tr, y_tr)
    a = (c_val == hi).astype(int)
    tab = np.array([[np.sum((a == 1) & (y_val == 1)), np.sum((a == 1) & (y_val == 0))],
                    [np.sum((a == 0) & (y_val == 1)), np.sum((a == 0) & (y_val == 0))]])
    p_rep = stats.fisher_exact(tab, alternative="greater").pvalue if tab.sum() else 1.0
    # Design 6.4.4: retain only if the direction matches the training set and the association is significant
    # in the validation subset, or - when that subset is too small for significance - consistent in magnitude.
    rate_val_hi = float(y_val[a == 1].mean()) if (a == 1).any() else np.nan
    rate_val_lo = float(y_val[a == 0].mean()) if (a == 0).any() else np.nan
    diff_val = rate_val_hi - rate_val_lo
    diff_tr = float(y_tr[c_tr == hi].mean() - y_tr[c_tr != hi].mean()) if (c_tr != hi).any() else np.nan
    direction_ok = bool(np.isfinite(diff_val) and diff_val > 0)
    too_small = len(ids_val) < 30
    consistent = bool(too_small and direction_ok and np.isfinite(diff_tr) and diff_tr > 0 and diff_val >= 0.5 * diff_tr)
    replicated = bool(direction_ok and (p_rep < 0.05 or consistent))
    n_tot = tab_tr.sum()
    cramers_v = float(np.sqrt(chi_tr / (n_tot * max(1, min(tab_tr.shape) - 1)))) if n_tot else np.nan
    validated = bool(p_tr < 0.05 and np.isfinite(adj_p) and adj_p < 0.05 and adj_or > 1 and replicated)
    res = dict(n_train=int(len(ids_tr)), n_validation=int(len(ids_val)), train_p=round(float(p_tr), 4),
               train_chi2=round(chi_tr, 2), high_cluster=hi, cluster_rates=[round(float(r), 2) for r in rate],
               adjusted_or=None if not np.isfinite(adj_or) else round(adj_or, 3),
               adjusted_p=None if not np.isfinite(adj_p) else round(adj_p, 4),
               cramers_v=round(cramers_v, 3), train_test=exact, validation_p=round(float(p_rep), 4),
               replication=("significant" if p_rep < 0.05 else ("consistent in magnitude (validation subset < 30)" if consistent else "not replicated")),
               direction_matches=direction_ok, validated=validated)
    return res, hi


# ============================================================================ Stage 5
def delong(y, p1, p2):
    """DeLong test for two correlated AUROCs (fast version, Sun & Xu 2014)."""
    y = np.asarray(y)
    pos, neg = y == 1, y == 0

    def v10_01(p):
        x, z = p[pos], p[neg]
        v10 = np.array([(np.sum(xi > z) + 0.5 * np.sum(xi == z)) / len(z) for xi in x])
        v01 = np.array([(np.sum(x > zj) + 0.5 * np.sum(x == zj)) / len(x) for zj in z])
        return v10, v01

    a10, a01 = v10_01(p1)
    b10, b01 = v10_01(p2)
    auc1, auc2 = a10.mean(), b10.mean()
    s10 = np.cov(np.vstack([a10, b10]))
    s01 = np.cov(np.vstack([a01, b01]))
    var = (s10[0, 0] + s10[1, 1] - 2 * s10[0, 1]) / len(a10) + (s01[0, 0] + s01[1, 1] - 2 * s01[0, 1]) / len(a01)
    if var <= 0:                      # degenerate case (e.g. perfect or constant predictions)
        return float(auc1), float(auc2), (0.0 if auc1 != auc2 else 1.0)
    z = (auc1 - auc2) / np.sqrt(var)
    return float(auc1), float(auc2), float(2 * stats.norm.sf(abs(z)))


def _n_splits(y, max_splits=5):
    """Fold count that every class can support (at least 2)."""
    return int(max(2, min(max_splits, np.bincount(np.asarray(y, int)).min())))


def en_model(n_splits=5):
    """Elastic-net logistic regression; C and the L1/L2 mix chosen by stratified CV on AUROC."""
    kw = dict(solver="saga", l1_ratios=[0.2, 0.5, 0.8], Cs=10, cv=n_splits, scoring="roc_auc",
              max_iter=10000, class_weight="balanced", random_state=0)
    if _sklearn_needs_penalty():
        kw["penalty"] = "elasticnet"
    return LogisticRegressionCV(**kw)


def en_fixed(C, l1):
    kw = dict(C=C, l1_ratio=l1, solver="saga", max_iter=10000, class_weight="balanced")
    if _sklearn_needs_penalty():
        kw["penalty"] = "elasticnet"
    return LogisticRegression(**kw)


def fit_scaled(X, y, n_splits):
    sc = StandardScaler().fit(X)
    m = en_model(n_splits).fit(sc.transform(X), y)
    return sc, m


def pick_threshold(y, p, min_sens=0.80):
    """Highest threshold that still gives >= 80% sensitivity on cross-validated training predictions."""
    for t in np.sort(np.unique(p))[::-1]:
        if np.mean(p[y == 1] >= t) >= min_sens:
            return float(t)
    return REVIEW_THRESHOLD


def stage5_classifier(F, C, y, ids_tr, ids_te, sleep_feats):
    imp_med = F.loc[ids_tr, sleep_feats].median()
    S = F[sleep_feats].fillna(imp_med)
    Xc_tr, Xc_te = C.loc[ids_tr].to_numpy(float), C.loc[ids_te].to_numpy(float)
    Xf_tr = np.column_stack([Xc_tr, S.loc[ids_tr].to_numpy(float)])
    Xf_te = np.column_stack([Xc_te, S.loc[ids_te].to_numpy(float)])
    ytr, yte = y.loc[ids_tr].to_numpy(int), y.loc[ids_te].to_numpy(int)
    n_splits = _n_splits(ytr)

    sc_c, m_c = fit_scaled(Xc_tr, ytr, n_splits)
    sc_f, m_f = fit_scaled(Xf_tr, ytr, n_splits)
    C_best, l1_best = float(np.atleast_1d(m_f.C_)[0]), float(np.atleast_1d(m_f.l1_ratio_)[0])
    cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)
    p_cv = cross_val_predict(en_fixed(C_best, l1_best), sc_f.transform(Xf_tr), ytr, cv=cv,
                             method="predict_proba")[:, 1]
    thr = pick_threshold(ytr, p_cv)
    p_full = m_f.predict_proba(sc_f.transform(Xf_te))[:, 1]
    p_clin = m_c.predict_proba(sc_c.transform(Xc_te))[:, 1]
    auc_f, auc_c, p_delong = delong(yte, p_full, p_clin)
    prauc_f = float(average_precision_score(yte, p_full)) if yte.sum() else np.nan
    prauc_c = float(average_precision_score(yte, p_clin)) if yte.sum() else np.nan
    sens = float(np.mean(p_full[yte == 1] >= thr)) if (yte == 1).any() else np.nan
    spec = float(np.mean(p_full[yte == 0] < thr)) if (yte == 0).any() else np.nan
    checks = {"AUC gain >= 0.05": (auc_f - auc_c) >= 0.05, "DeLong p < 0.05": p_delong < 0.05,
              "sensitivity >= 80%": sens >= 0.80, "specificity >= 70%": spec >= 0.70}
    go = bool(all(checks.values()))

    rng = np.random.default_rng(SEED)
    boots = []                                              # for per-patient 95% intervals
    for _ in range(N_BOOT):
        b = rng.integers(0, len(ytr), len(ytr))
        if np.unique(ytr[b]).size < 2:
            continue
        sb = StandardScaler().fit(Xf_tr[b])
        boots.append((sb, en_fixed(C_best, l1_best).fit(sb.transform(Xf_tr[b]), ytr[b])))
    names = list(C.columns) + list(sleep_feats)
    coefs = dict(zip(names, m_f.coef_[0].round(3).tolist()))
    res = dict(n_train=int(len(ytr)), n_test=int(len(yte)), auc_full=round(auc_f, 3), auc_clinical=round(auc_c, 3),
               auc_gain=round(auc_f - auc_c, 3), delong_p=round(p_delong, 4),
               prauc_full=round(prauc_f, 3), prauc_clinical=round(prauc_c, 3), threshold=round(thr, 3),
               sensitivity=round(sens, 3), specificity=round(spec, 3), go=go,
               criteria={k: bool(v) for k, v in checks.items()}, coefficients=coefs,
               C=C_best, l1_ratio=l1_best, cv_folds=n_splits)
    model = dict(scaler=sc_f, model=m_f, names=names, sleep_feats=list(sleep_feats), impute=imp_med.to_dict(),
                 threshold=thr, boots=boots)
    return res, model, dict(zip(ids_te, p_full))


# ============================================================================ report
def fig_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def hypnogram_png(n):
    hyp, marks = n["hyp"], n["marks"]
    t = np.arange(len(hyp)) * EPOCH / 3600
    ylev = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    yv = np.array([ylev.get(int(v), np.nan) for v in hyp], float)
    fig, ax = plt.subplots(figsize=(10, 2.6))
    for i in np.where(n["resp"])[0]:
        ax.axvspan(t[i], t[i] + EPOCH / 3600, color="#f3dcd6", lw=0)
    for i in np.where(n["conf"] < LOW_CONF)[0]:
        ax.axvspan(t[i], t[i] + EPOCH / 3600, color="#e8e4f3", lw=0)
    ax.step(t, yv, where="post", color="#2d5f7c", lw=1.3)
    for key, col, lab in (("onset", "#1f8a5b", "Sleep onset"), ("rem_onset", "#c2413b", "REM onset (3-epoch rule)")):
        if marks.get(key) is not None:
            x = t[marks[key]]
            ax.axvline(x, color=col, lw=1.4, ls="--")
            ax.text(x, 4.35, lab, color=col, fontsize=7, ha="left")
    ax.set_yticks([4, 3, 2, 1, 0])
    ax.set_yticklabels(["Wake", "REM", "N1", "N2", "N3"])
    ax.set_xlabel("Hours from start of analysis window")
    ax.set_ylim(-0.4, 4.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig_b64(fig)


def reference_ranges(F, y, ids_dev):
    """Cohort reference: 10th-90th percentile of development participants without anxiety."""
    ref = F.loc[[i for i in ids_dev if y[i] == 0]]
    return {k: (float(ref[k].quantile(0.1)), float(ref[k].quantile(0.9))) for k in LITERATURE if ref[k].notna().sum() >= 5}


def patient_prediction(bundle, n, crow):
    m5 = bundle["stage5"]
    C = clinical_matrix(crow.to_frame().T, bundle["med_levels"])
    s = [n["features"].get(k, np.nan) for k in m5["sleep_feats"]]
    s = [m5["impute"][k] if not np.isfinite(v) else v for k, v in zip(m5["sleep_feats"], s)]
    x = np.r_[C.to_numpy(float)[0], np.array(s, float)][None]
    xs = m5["scaler"].transform(x)
    p = float(m5["model"].predict_proba(xs)[0, 1])
    bp = [lr.predict_proba(sb.transform(x))[0, 1] for sb, lr in m5["boots"]]
    lo, hi = (np.percentile(bp, [2.5, 97.5]) if bp else (p, p))
    contrib = dict(zip(m5["names"], (m5["model"].coef_[0] * xs[0]).tolist()))
    return p, float(lo), float(hi), contrib


def fmt(v):
    if v is None or not np.isfinite(v):
        return "–"
    return f"{v:.3f}" if abs(v) < 1 else (f"{v:.1f}" if abs(v) < 10 else f"{v:.0f}")


def write_report(bundle, n, crow, out_dir):
    p, lo, hi, contrib = patient_prediction(bundle, n, crow)
    thr = bundle["stage5"]["threshold"]
    ref = bundle["ref"]
    rows = []
    for k in bundle["stage5"]["sleep_feats"]:
        if k not in LITERATURE:
            continue
        v = n["features"].get(k, np.nan)
        label, unit, _ = LITERATURE[k]
        r = ref.get(k)
        flag = ""
        if r and np.isfinite(v):
            flag = "Above range" if v > r[1] else ("Below range" if v < r[0] else "")
        rows.append((abs(contrib.get(k, 0)), label, v, unit, r, flag, contrib.get(k, 0)))
    if "latent_high_cluster" in contrib:
        member = float(bundle["clusters"].get(n["id"], -1) == bundle["stage4"]["high_cluster"])
        rows.append((abs(contrib["latent_high_cluster"]), "Latent sleep profile (Stage 3 cluster)",
                     member, "member", None, "", contrib["latent_high_cluster"]))
    rows.sort(key=lambda r: r[0], reverse=True)
    mx = max([r[0] for r in rows] + [1e-6])
    tr = "".join(
        f"<tr><td>{lab}{f'<br><span class=flag>{fl}</span>' if fl else ''}</td>"
        f"<td class=num>{fmt(v)} {u}</td>"
        f"<td class=num>{'' if not rg else fmt(rg[0]) + '–' + fmt(rg[1])}</td>"
        f"<td><div class=bar style='width:{100 * a / mx:.0f}%;background:{'#2d5f7c' if c > 0 else '#9fb6c6'}'></div></td></tr>"
        for a, lab, v, u, rg, fl, c in rows[:6])
    qc = n["qc"]
    verdict = ("<b>Above the review threshold.</b> Sleep findings support a closer clinical look at anxiety symptoms."
               if p >= thr else "<b>Below the review threshold.</b> Sleep findings do not add support for further anxiety review.")
    excl = ", ".join(qc["excluded"]) or "None"
    unav = "; ".join(qc["unavailable"]) or "None"
    s5 = bundle["stage5res"]
    html = f"""<!doctype html><html><head><meta charset=utf-8><title>Sleep report {html_lib.escape(str(n['id']))}</title>
<style>body{{font-family:Arial,Helvetica,sans-serif;color:#1b2a38;margin:0;background:#fff}}
.wrap{{max-width:980px;margin:24px auto;border:1px solid #dfe5eb;border-radius:8px;overflow:hidden}}
.head{{background:#eef2f6;padding:18px 24px;display:flex;justify-content:space-between}}
.kicker{{font-family:monospace;letter-spacing:.1em;font-size:12px;color:#5b6b7c}}
h1{{margin:4px 0 0;font-size:24px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:24px;padding:20px 24px}}
.big{{font-size:52px;font-weight:700}} .muted{{color:#5b6b7c}} table{{width:100%;border-collapse:collapse;font-size:14px}}
td,th{{padding:7px 4px;border-bottom:1px solid #e6ebf0;text-align:left;vertical-align:top}} th{{font-family:monospace;font-size:11px;color:#5b6b7c}}
.num{{font-family:monospace;white-space:nowrap}} .flag{{display:inline-block;margin-top:4px;border:1px solid #b7791f;color:#b7791f;border-radius:10px;padding:1px 8px;font-size:12px}}
.bar{{height:8px;border-radius:4px;margin-top:6px}} .track{{position:relative;height:14px;background:#eef2f6;border-radius:7px;margin:14px 0 4px}}
.band{{position:absolute;top:2px;height:10px;background:#9fb6c6;border-radius:5px}} .pt{{position:absolute;top:-3px;width:4px;height:20px;background:#2d5f7c}}
.thr{{position:absolute;top:-6px;height:26px;border-left:2px dashed #5b6b7c}} .note{{margin:0 24px 20px;padding:12px 16px;background:#eef2f6;border-left:4px solid #b7791f}}
.sec{{padding:0 24px 16px}} .kv td{{border:none;padding:3px 4px}}</style></head><body><div class=wrap>
<div class=head><div><div class=kicker>SLEEP-DERIVED ANXIETY INDICATORS · SUPPORTING REPORT</div><h1>{html_lib.escape(str(n['id']))}</h1></div>
<table class=kv style="width:auto"><tr><td class=muted>Age</td><td>{float(crow['age']):.0f}</td><td class=muted>Cognitive status</td><td>{html_lib.escape(str(crow['cognitive_status']))}</td></tr>
<tr><td class=muted>AHI</td><td>{float(crow['ahi']):.0f} /h</td><td class=muted>Medication</td><td>{html_lib.escape(str(crow['med_class']))}</td></tr></table></div>
<div class=grid><div><b>Estimated probability of clinician-assessed anxiety</b>
<div><span class=big>{100 * p:.0f}%</span> <span class=muted>(95% interval {100 * lo:.0f}–{100 * hi:.0f}%)</span></div>
<div class=track><div class=band style="left:{100 * lo:.0f}%;width:{max(1, 100 * (hi - lo)):.0f}%"></div><div class=pt style="left:{100 * p:.0f}%"></div><div class=thr style="left:{100 * thr:.0f}%"></div></div>
<div class=muted style="font-size:12px;display:flex;justify-content:space-between"><span>0%</span><span>review threshold {100 * thr:.0f}%</span><span>100%</span></div>
<p>{verdict}</p><table class=kv><tr><td class=muted>Signals evaluated</td><td>{len(n['signals'])} ({', '.join(s.upper() for s in n['signals'])})</td></tr>
<tr><td class=muted>Sleep staging</td><td>{qc['staging_source']}</td></tr></table></div>
<div><b>What drove this estimate</b> <span class=muted style="font-size:12px">— dark bar raises the estimate, light bar lowers it</span><table><tr><th>MEASURE</th><th>VALUE</th><th>COHORT RANGE</th><th>CONTRIBUTION</th></tr>{tr}</table></div></div>
<div class=sec><b>The night</b> <span class=muted style="font-size:12px">— dashed lines mark the two epochs that define REM latency;
pink = apnoea-related epochs excluded; lilac = low-confidence automated staging</span><br>
<img style="width:100%" src="data:image/png;base64,{hypnogram_png(n)}"></div>
<div class=grid><div><b>Recording quality</b><table class=kv>
<tr><td class=muted>Usable epochs</td><td>{qc['usable_pct']}%</td></tr>
<tr><td class=muted>Apnoea-related epochs excluded</td><td>{qc['resp_excluded_pct']}%</td></tr>
<tr><td class=muted>Staging confidence (mean)</td><td>{'clinician-scored' if qc['clinician_scored'] else qc['mean_conf']}</td></tr>
<tr><td class=muted>Analysis window</td><td>{qc['window']}</td></tr>
<tr><td class=muted>Channels excluded</td><td>{excl}</td></tr>
<tr><td class=muted>Not computed</td><td>{unav}</td></tr></table></div>
<div><b>Method and limits</b><p class=muted>Probability from an elastic-net logistic regression combining sleep measures with age, sex,
cognitive status, AHI and medication class (research model, {bundle['trained_on']}). REM latency uses the first run of
three consecutive REM epochs. Held-out performance: AUROC {s5['auc_full']} vs {s5['auc_clinical']} for clinical
information alone. The model has not passed clinical validation.</p></div></div>
<div class=note><b>Not a diagnosis.</b> This report supports, and does not replace, clinical assessment. Read it alongside the
patient's history, medications and questionnaire results.</div></div></body></html>"""
    out = Path(out_dir) / f"report_{n['id']}.html"
    out.write_text(html, encoding="utf-8")
    return out, p


# ============================================================================ commands
def channels(args):
    return dict(eeg=args.eeg, eog=args.eog, emg=args.emg, ecg=args.ecg)


def analyse(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clin = load_clinical(args.clinical)
    if "anxiety" not in clin.columns:
        raise PipelineError("Clinical CSV needs an 'anxiety' column (1/0) for the analysis.")
    nights, failed = [], {}
    for edf in sorted(Path(args.edf_dir).glob("*.edf")):
        if edf.stem.endswith("-Hypnogram") or edf.stem not in clin.index:
            continue
        try:
            log(f"Stage 1  {edf.stem}")
            nights.append(stage1_night(edf, channels(args), clin.loc[edf.stem]))
        except PipelineError as e:
            failed[edf.stem] = str(e)
            log(f"         skipped: {e}")
    if len(nights) < MIN_RECORDINGS:
        raise PipelineError(f"Only {len(nights)} usable recordings; at least {MIN_RECORDINGS} are needed for the analysis.")
    ids = [n["id"] for n in nights]
    y_all = clin.loc[ids, "anxiety"].astype(int)
    if y_all.sum() < 10 or (1 - y_all).sum() < 10:
        raise PipelineError(f"Need at least 10 people with and 10 without anxiety; found {int(y_all.sum())} and "
                            f"{int((1 - y_all).sum())}.")
    F = pd.DataFrame([n["features"] for n in nights], index=ids)
    y = clin.loc[ids, "anxiety"].astype(int)
    med_levels = sorted(m for m in clin["med_class"].unique() if m != "none")
    C = clinical_matrix(clin.loc[ids], med_levels)

    # Cohort allocation: 80% development / 20% independent test; a validation subset inside development for Stage 4
    ids_dev, ids_te = stratified_split(ids, y.loc[ids], TEST_FRACTION, SEED)
    ids_s3, ids_val = stratified_split(ids_dev, y.loc[ids_dev], STAGE4_VAL_FRACTION, SEED)
    ids_dev, ids_te, ids_s3, ids_val = (sorted(x) for x in (ids_dev, ids_te, ids_s3, ids_val))
    if y.loc[ids_te].nunique() < 2:
        raise PipelineError("The independent test cohort contains only one anxiety class, so AUROC is undefined. Add recordings.")
    log(f"Split: {len(ids_dev)} development ({len(ids_s3)} Stage 3 training + {len(ids_val)} Stage 4 validation) "
        f"/ {len(ids_te)} independent test")

    log("Stage 2  literature biomarkers")
    T2 = stage2_literature(F.loc[ids_dev], C.loc[ids_dev], y.loc[ids_dev].to_numpy())
    survivors = T2.loc[T2["carried_forward"] == True, "feature"].tolist()  # noqa: E712
    fallback = not survivors
    sleep_feats = survivors or [k for k in LITERATURE if F[k].notna().mean() > 0.8]
    if fallback:
        log("         WARNING: no literature feature survived FDR control; Stage 5 uses all computable features (exploratory).")

    log("Stage 3  discovery")
    clusters, s3, m3 = stage3_discovery(nights, ids_s3)
    log("Stage 4  validation of discovered groups")
    s4, hi = stage4_validate(clusters, ids_s3, ids_val, y.loc[ids_s3], y.loc[ids_val], C.loc[ids_s3], s3["k"])
    if s4["validated"]:
        F["latent_high_cluster"] = [float(clusters[i] == hi) for i in ids]
        for n in nights:
            n["features"]["latent_high_cluster"] = float(clusters[n["id"]] == hi)
        sleep_feats = sleep_feats + ["latent_high_cluster"]

    log("Stage 5  classifier and go/no-go")
    s5, m5, p_te = stage5_classifier(F, C, y, ids_dev, ids_te, sleep_feats)

    bundle = dict(stage5=m5, stage5res=s5, stage3=m3, stage4=s4, clusters=clusters, med_levels=med_levels,
                  ref=reference_ranges(F, y, ids_dev), trained_on=f"trained on {len(ids_dev)} people", channels=channels(args))
    joblib.dump(bundle, out / "model.joblib")
    T2.to_csv(out / "stage2_literature.csv", index=False)
    subset = {i: "development (Stage 3 training)" for i in ids_s3}
    subset.update({i: "development (Stage 4 validation)" for i in ids_val})
    subset.update({i: "independent test" for i in ids_te})
    F.assign(anxiety=y, set=[subset[i] for i in ids]).to_csv(out / "features.csv")
    s5["exploratory_features"] = bool(fallback)
    summary = dict(n_recordings=len(nights), failed=failed,
                   split=dict(development=len(ids_dev), stage3_training=len(ids_s3), stage4_validation=len(ids_val),
                              independent_test=len(ids_te)),
                   stage2=dict(carried_forward=survivors, fallback_all_features=fallback),
                   stage3=s3, stage4=s4, stage5=dict(s5), qc={n["id"]: n["qc"] for n in nights},
                   software=dict(sklearn=SKLEARN_VERSION, mne=mne.__version__, seed=SEED))
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    rep = out / "reports"
    rep.mkdir(exist_ok=True)
    for n in nights:
        if n["id"] in ids_te:
            write_report(bundle, n, clin.loc[n["id"]], rep)
    write_summary_html(out, summary, T2)
    log("")
    log(f"Stage 5: AUROC {s5['auc_full']} (sleep + clinical) vs {s5['auc_clinical']} (clinical only), "
        f"gain {s5['auc_gain']:+.3f}, DeLong p = {s5['delong_p']}; PR-AUC {s5['prauc_full']} vs {s5['prauc_clinical']}")
    log(f"         sensitivity {s5['sensitivity']:.0%}, specificity {s5['specificity']:.0%} at threshold {s5['threshold']:.2f}")
    log(f"Decision (Validation 4 criteria): {'GO' if s5['go'] else 'NO-GO'}")
    for k, v in s5["criteria"].items():
        log(f"         {'met' if v else 'NOT met'}: {k}")
    log(f"Results in {out}/ (summary.html, results.json, stage2_literature.csv, features.csv, reports/)")
    return summary


def write_summary_html(out, S, T2):
    s5, s4, s3, sp = S["stage5"], S["stage4"], S["stage3"], S["split"]
    t2 = T2.copy()
    cols = [c for c in ["label", "n", "expected", "hedges_g", "test", "raw_p", "q_raw", "beta", "adj_p", "q", "carried_forward"] if c in t2]
    adj = "n/a" if s4["adjusted_p"] is None else f"{s4['adjusted_p']} (OR {s4['adjusted_or']})"
    html = (f"<!doctype html><meta charset=utf-8><title>Sleeping Minds analysis</title>"
            f"<style>body{{font-family:Arial;max-width:960px;margin:24px auto;color:#1b2a38}}table{{border-collapse:collapse;font-size:13px}}"
            f"td,th{{border-bottom:1px solid #dfe5eb;padding:5px 8px;text-align:left}}.go{{font-size:22px;font-weight:700}}</style>"
            f"<h1>Sleeping Minds — analysis summary</h1><p>{S['n_recordings']} recordings analysed; {len(S['failed'])} skipped. "
            f"Development cohort {sp['development']} ({sp['stage3_training']} Stage 3 training + {sp['stage4_validation']} Stage 4 validation); "
            f"independent test cohort {sp['independent_test']}.</p>"
            f"<p class=go>Decision: {'GO' if s5['go'] else 'NO-GO'}</p>"
            + ("<p><b>Exploratory:</b> no literature feature survived FDR control, so Stage 5 used every computable feature.</p>"
               if s5.get("exploratory_features") else "") +
            f"<p>Held-out AUROC {s5['auc_full']} (sleep + clinical) vs {s5['auc_clinical']} (clinical only); gain {s5['auc_gain']:+.3f};"
            f" DeLong p = {s5['delong_p']}; PR-AUC {s5['prauc_full']} vs {s5['prauc_clinical']};"
            f" sensitivity {s5['sensitivity']:.0%}, specificity {s5['specificity']:.0%} at threshold {s5['threshold']:.2f}.</p>"
            f"<h2>Stage 2 — literature biomarkers (development cohort, adjusted, FDR q &lt; {FDR_Q})</h2>"
            + t2[cols].round(3).to_html(index=False) +
            f"<h2>Stage 3 — discovery</h2><p>{s3['method']} (rebuild error {s3['mse_ae']} vs PCA {s3['mse_pca']}); k = {s3['k']} groups.</p>"
            f"<h2>Stage 4 — validation of discovered groups</h2><p>Training permutation p = {s4['train_p']}; "
            f"Cramér's V = {s4['cramers_v']}; confounder-adjusted p = {adj}; validation-subset replication p = {s4['validation_p']} ({s4['replication']});"
            f" {'validated and carried into Stage 5' if s4['validated'] else 'not validated, so not used in Stage 5'}.</p>")
    (out / "summary.html").write_text(html, encoding="utf-8")


def report_cmd(args):
    bundle = joblib.load(args.model)
    clin = load_clinical(args.clinical)
    if args.id not in clin.index:
        raise PipelineError(f"Participant '{args.id}' is not in the clinical CSV.")
    n = stage1_night(Path(args.edf), bundle["channels"], clin.loc[args.id])
    m3 = bundle["stage3"]
    prof = stage3_profile(m3, n)
    bundle["clusters"][n["id"]] = int(m3["kmeans"].predict(m3["prof_scaler"].transform(prof[None]))[0])
    n["features"]["latent_high_cluster"] = float(bundle["clusters"][n["id"]] == bundle["stage4"]["high_cluster"])
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path, p = write_report(bundle, n, clin.loc[args.id], args.out)
    log(f"{args.id}: probability {p:.0%}; report written to {path}")


# ============================================================================ demo cohort
def make_demo_cohort(folder, n=72, hours=7.0, sf=100):
    """Simulated cohort for demonstrating the pipeline. NOT real data; effects are planted."""
    import neurokit2 as nk
    rng = np.random.default_rng(SEED)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    n_ep = int(hours * 3600 / EPOCH)
    t = np.arange(EPOCH * sf) / sf
    desc = {0: "Sleep stage W", 1: "Sleep stage 1", 2: "Sleep stage 2", 3: "Sleep stage 3", 4: "Sleep stage R"}
    amp_by_stage = {0: (5, 6, 18, 2, 8), 1: (12, 18, 6, 2, 4), 2: (22, 14, 4, 10, 3), 3: (70, 10, 2, 3, 2), 4: (12, 16, 5, 2, 3)}
    emg_sd = {0: 20, 1: 10, 2: 8, 3: 7, 4: 2}
    for s in range(n):
        pid = f"P{s + 1:03d}"
        age = int(rng.uniform(60, 88))
        sex = rng.choice(["F", "M"])
        cog = rng.choice(["MCI", "SCI"], p=[0.67, 0.33])
        ahi = float(np.round(rng.gamma(2.0, 6.0), 1))
        med = rng.choice(["none", "ssri", "benzodiazepine"], p=[0.6, 0.25, 0.15])
        lat = rng.normal()                                      # latent anxiety tendency
        # anxiety depends partly on clinical factors and partly on the latent tendency the sleep
        # signals carry, so neither the clinical-only nor the sleep model is perfect
        anx = int(1 / (1 + np.exp(-(2.0 * lat - 1.0 + 0.9 * (med == "benzodiazepine") + 0.6 * (cog == "MCI")))) > rng.random())
        # hypnogram: sleep onset and REM latency lengthened by anxiety (planted effect)
        sol = int(max(2, rng.normal(18 + 7 * lat, 8)) * 2)
        rem_lat = int(max(60, rng.normal(90 + 18 * lat, 20)) * 2)
        hyp = np.zeros(n_ep, int)
        i = sol
        cyc = 0
        while i < n_ep:
            first = cyc == 0
            nrem = rem_lat if first else int(rng.normal(150, 25))
            seq = ([1] * 4 + [2] * (nrem // 2 - 4) + [3] * max(0, nrem // 2 - cyc * 12 - int(ahi / 2)) + [2] * 8)[:max(nrem, 10)]
            for v in seq + [4] * int(rng.normal(30 + 8 * cyc, 6)):
                if i >= n_ep:
                    break
                hyp[i] = v if rng.random() > 0.03 + 0.01 * anx else 0
                i += 1
            cyc += 1
        sigs = {"C4-M1": [], "E1-M2": [], "Chin1-Chin2": []}
        beta_boost = max(0.5, 1 + 0.3 * lat + 0.2 * rng.normal())   # NREM beta power raised with anxiety (planted)
        for st in hyp:
            a = amp_by_stage[st]
            ph = rng.uniform(0, 6.3, 5)
            eeg = (a[0] * np.sin(2 * np.pi * 1.5 * t + ph[0]) + a[1] * np.sin(2 * np.pi * 6 * t + ph[1]) +
                   a[2] * np.sin(2 * np.pi * 10 * t + ph[2]) + a[3] * np.sin(2 * np.pi * 13 * t + ph[3]) +
                   a[4] * (beta_boost if st in (2, 3) else 1) * np.sin(2 * np.pi * 20 * t + ph[4]) + 3 * rng.normal(0, 1, t.size))
            sigs["C4-M1"].append(eeg)
            sigs["E1-M2"].append(rng.normal(0, 4, t.size) + (40 * np.sign(np.sin(2 * np.pi * 0.4 * t + ph[4])) if st == 4
                                                               else 10 * np.sin(2 * np.pi * 0.2 * t)))
            sigs["Chin1-Chin2"].append(rng.normal(0, emg_sd[st], t.size))
        hr = 62 + 6 * anx + rng.normal(0, 4)
        ecg = nk.ecg_simulate(duration=int(hours * 3600), sampling_rate=sf, heart_rate=hr,
                              heart_rate_std=max(2.0, 7 - 3 * anx), method="simple", random_state=s)[: n_ep * EPOCH * sf]
        data = np.vstack([np.concatenate(sigs[c]) * 1e-6 for c in sigs] + [ecg * 1e-3])
        info = mne.create_info(list(sigs) + ["ECG"], sf, ["eeg", "eog", "emg", "ecg"])
        raw = mne.io.RawArray(data, info)
        events = []
        for _ in range(int(ahi * hours)):                       # scored apnoea events
            on = rng.uniform(sol * EPOCH, n_ep * EPOCH - 60)
            events.append((on, rng.uniform(10, 30), "Obstructive apnea"))
        # Clinician scoring supplied three ways, to exercise each input path; every 6th person has
        # none, so automated staging runs (on simulated signals it is not meaningful - see Test 4
        # for its accuracy on real recordings).
        if s % 6 in (0, 2, 4):
            events += [(k * EPOCH, EPOCH, desc[v]) for k, v in enumerate(hyp)]
        elif s % 6 in (1, 3):
            pd.Series([STAGES[v] if v != 4 else "R" for v in hyp]).to_csv(folder / f"{pid}_hypnogram.csv", index=False, header=False)
        if events:
            on, du, de = zip(*events)
            raw.set_annotations(mne.Annotations(on, du, de))
        mne.export.export_raw(folder / f"{pid}.edf", raw, fmt="edf", overwrite=True, physical_range="auto")
        rows.append(dict(id=pid, anxiety=anx, age=age, sex=sex, ahi=ahi, med_class=med, cognitive_status=cog))
        log(f"  simulated {pid}")
    pd.DataFrame(rows).to_csv(folder / "clinical.csv", index=False)
    return folder / "clinical.csv"


def demo(args):
    folder = Path(args.out) / "demo_data"
    if not (folder / "clinical.csv").exists():
        log("Building a simulated cohort (not real patients)...")
        make_demo_cohort(folder, n=args.n)
    args.edf_dir, args.clinical = str(folder), str(folder / "clinical.csv")
    args.eeg, args.eog, args.emg, args.ecg = "C4-M1", "E1-M2", "Chin1-Chin2", "ECG"
    args.out = str(Path(args.out) / "demo_results")
    analyse(args)


def main():
    ap = argparse.ArgumentParser(description="Sleeping Minds solution pipeline (research use only)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("analyse", "report", "demo"):
        p = sub.add_parser(name)
        p.add_argument("--out", default="results" if name != "report" else "reports")
        p.add_argument("--eeg", default="C4-M1")
        p.add_argument("--eog", default="E1-M2")
        p.add_argument("--emg", default="Chin1-Chin2")
        p.add_argument("--ecg", default="ECG")
        if name == "analyse":
            p.add_argument("--edf-dir", required=True)
            p.add_argument("--clinical", required=True)
        if name == "report":
            p.add_argument("--model", required=True)
            p.add_argument("--edf", required=True)
            p.add_argument("--clinical", required=True)
            p.add_argument("--id", required=True)
        if name == "demo":
            p.add_argument("--n", type=int, default=120, help="number of simulated people")
    args = ap.parse_args()
    try:
        {"analyse": analyse, "report": report_cmd, "demo": demo}[args.cmd](args)
    except PipelineError as e:
        sys.exit(f"Error: {e}")


if __name__ == "__main__":
    main()
