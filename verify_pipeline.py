"""
Sleeping Minds - verification of the six-stage pipeline on REAL, unlabelled sleep data.

What it does
------------
Downloads overnight recordings from the public Sleep-EDF Expanded database (PhysioNet,
via MNE), runs a verification test for every stage of the pipeline, and writes:

    vv_report.html     a visual V&V report (open in any browser)
    vv_results.json    every number in the report
    vv_figures/*.png   each figure on its own, ready to paste into the Word report

Sleep-EDF has technician sleep scoring but NO anxiety labels. So this script verifies that
the software works correctly on real recordings. It does not validate any anxiety biomarker.
Stages 4 and 5 are checked with synthetic labels on the real features (null and planted
effects), which is the standard way to verify statistical machinery without outcome data.

How to run (Mac / Windows / Linux, Python 3.10+)
------------------------------------------------
    pip install mne yasa neurokit2 scikit-learn scipy statsmodels matplotlib pooch edfio joblib pandas
    python verify_pipeline.py --n 20          # the 18 nights reported in Appendix 12 (~1.4 GB download)
    python verify_pipeline.py                 # about 26 nights, roughly 1.8 GB download
    python verify_pipeline.py --synthetic     # no download: dry run on simulated nights

Nights are taken from every third Sleep-EDF subject (0, 3, 6, ...). Subjects for which
PhysioNet holds no first night (36, 39, 68, 69, 78, 79) are skipped with a warning, which is
why ``--n 20`` yields 18 nights. The first run downloads the data to ~/mne_data and reuses it.
"""
import argparse
import base64
import io
import json
import re
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import signal as sps, stats  # noqa: E402
from sklearn import __version__ as SKLEARN_VERSION  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

warnings.filterwarnings("ignore")
import mne  # noqa: E402
import yasa  # noqa: E402

mne.set_log_level("ERROR")

import run_pipeline as rp  # noqa: E402  (the shipped pipeline; keep this file next to run_pipeline.py)

# ----------------------------------------------------------------------------- settings
EEG, EEG2, EOG, EMG = "EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal", "EMG submental"
REQUIRED = [EEG, EOG, EMG]          # channels the full pipeline expects (EEG is mandatory)
STAGES = ["W", "N1", "N2", "N3", "REM"]
SLEEP_CODES = [1, 2, 3, 4]          # integer codes that count as sleep (W = 0, unscored = -1)
EPOCH = 30                          # seconds per epoch (AASM)
EPOCHS_PER_HOUR = 3600 // EPOCH
MIN_PER_EPOCH = EPOCH / 60
SEED = 2026
RNG = np.random.default_rng(SEED)
OUT = Path("vv_figures")
RESULTS_JSON = Path("vv_results.json")
REPORT_HTML = Path("vv_report.html")
MIN_K, MAX_K = 2, 6                 # k-means cluster counts compared by silhouette (design Section 6.3.3)
LATENT_DIM = 6                      # autoencoder bottleneck size

# Pass criteria, copied from the V&V plan (Table: Verification Test Plans)
CRIT = dict(acc=0.80, kappa=0.70, rem_f1=0.80, ba_bias=5.0, ba_loa=15.0, icc=0.75)

# V5: broad plausibility ranges for older adults, informed by the age trends in
# Ohayon et al. 2004, Sleep 27(7):1255-73 (cited in the report as [Ohayon 2004]).
PLAUSIBLE_60PLUS = {"TST (min)": (300, 450), "SE (%)": (65, 95), "REM (%)": (12, 25),
                    "N3 (%)": (0, 20), "REM latency (min)": (50, 150)}

COL = dict(ink="#1b2a38", sub="#5b6b7c", acc="#1d6b8c", good="#1f8a5b", bad="#c2413b",
           warn="#b7791f", grid="#dfe5eb", stages=["#c9861a", "#8fb3d9", "#2f6fd6", "#173f8a", "#d24b4b"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": COL["sub"], "axes.labelcolor": COL["ink"],
                     "xtick.color": COL["sub"], "ytick.color": COL["sub"], "figure.dpi": 130})

FIGS = {}                           # figure name -> base64 PNG, embedded in the HTML report


# ----------------------------------------------------------------------------- small helpers
def save(fig, name):
    """Save a figure to vv_figures/<name>.png and keep a base64 copy for the HTML report."""
    OUT.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=160)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    FIGS[name] = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def as_py(value):
    """Convert numpy scalars/arrays to plain Python so they serialise and print cleanly."""
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


# ----------------------------------------------------------------------------- pipeline functions
class PipelineError(Exception):
    """Raised with a clear message when a recording cannot be processed."""


def validate_recording(raw, need_annotations=True, min_hours=4.0, min_sf=50):
    """Stage 1 input checks (SPR2.1, SPR2.5, SPR3.1). Returns the list of features that
    cannot be computed; raises PipelineError when the night cannot be staged at all."""
    if raw is None:
        raise PipelineError("File could not be read as EDF/EDF+.")
    if EEG not in raw.ch_names:
        raise PipelineError(f"Required EEG channel '{EEG}' is missing: sleep staging is impossible.")
    if raw.info["sfreq"] < min_sf:
        raise PipelineError(f"Sampling rate {raw.info['sfreq']:.0f} Hz is below the {min_sf} Hz minimum.")
    hours = raw.times[-1] / 3600
    if hours < min_hours:
        raise PipelineError(f"Recording is {hours:.1f} h; at least {min_hours:.0f} h is required.")
    if need_annotations and len(raw.annotations) == 0:
        raise PipelineError("No sleep-stage annotations found: technician comparison is impossible.")
    missing = [c for c in REQUIRED if c not in raw.ch_names]
    unavailable = []
    if EOG in missing:
        unavailable += ["REM density (EOG)", "EOG-assisted staging"]
    if EMG in missing:
        unavailable += ["REM muscle activity (EMG)", "EMG-assisted staging"]
    return unavailable


def load_edf(path):
    """Read an EDF/EDF+ file; return None (instead of raising) when it is not readable."""
    try:
        return mne.io.read_raw_edf(path, preload=True, stim_channel=None)
    except Exception:
        return None


# Stage labels as written by technicians (Sleep-EDF / R&K) and by YASA, mapped to W0 N1 N2 N3 REM4.
_MAP = {"W": 0, "WAKE": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 3, "R": 4, "REM": 4,
        "Sleep stage W": 0, "Sleep stage 1": 1, "Sleep stage 2": 2, "Sleep stage 3": 3,
        "Sleep stage 4": 3, "Sleep stage R": 4}


def to_int(h):
    """Convert a YASA Hypnogram, list of strings or ints to an int array (W0 N1 N2 N3 REM4, -1 unknown)."""
    vals = h.hypno.astype(str).tolist() if hasattr(h, "hypno") else list(h)
    return np.array([v if isinstance(v, (int, np.integer)) else _MAP.get(str(v), -1) for v in vals])


def rem_latency(hyp):
    """Minutes from sleep onset (first non-wake epoch) to REM onset (three-epoch rule). None if no REM or no sleep."""
    hyp = np.asarray(hyp)
    sleep = np.where(np.isin(hyp, SLEEP_CODES))[0]
    rem0 = rp.rem_onset_epoch(hyp)          # the shipped rule: first run of >= 3 consecutive REM epochs
    if len(sleep) == 0 or rem0 is None or rem0 < sleep[0]:
        return None
    return (rem0 - sleep[0]) * MIN_PER_EPOCH


def sleep_stats(hyp):
    """Continuity and architecture measures from a 30-s hypnogram (Stage 1 features)."""
    hyp = np.asarray(hyp)
    valid = hyp >= 0
    sleep = np.isin(hyp, SLEEP_CODES)
    idx = np.where(sleep)[0]
    tib = valid.sum() * MIN_PER_EPOCH
    tst = sleep.sum() * MIN_PER_EPOCH
    if len(idx) == 0:
        return dict(TST=0.0, SE=0.0, SOL=np.nan, WASO=np.nan, N1p=np.nan, N2p=np.nan, N3p=np.nan,
                    REMp=np.nan, REMLAT=np.nan, N1TR=np.nan)
    spt = hyp[idx[0]: idx[-1] + 1]                                   # sleep period time
    n1tr = np.sum((hyp[1:] == 1) & (hyp[:-1] != 1) & (hyp[:-1] >= 0))  # transitions into N1
    rl = rem_latency(hyp)
    return dict(TST=tst, SE=100 * tst / tib, SOL=idx[0] * MIN_PER_EPOCH, WASO=np.sum(spt == 0) * MIN_PER_EPOCH,
                N1p=100 * np.mean(hyp[sleep] == 1), N2p=100 * np.mean(hyp[sleep] == 2),
                N3p=100 * np.mean(hyp[sleep] == 3), REMp=100 * np.mean(hyp[sleep] == 4),
                REMLAT=np.nan if rl is None else rl, N1TR=n1tr / max(tst / 60, 1e-6))


def stage_with_yasa(raw, eeg, eog=None, emg=None):
    """Automated staging. EOG/EMG are used only when present in the recording."""
    eog = eog if eog in raw.ch_names else None
    emg = emg if emg in raw.ch_names else None
    sls = yasa.SleepStaging(raw, eeg_name=eeg, eog_name=eog, emg_name=emg)
    return to_int(sls.predict())


def epoch_features(raw, n_ep):
    """Per-epoch spectral features across all four signals (input to the Stage 3 autoencoder)."""
    sf = raw.info["sfreq"]
    picks = [c for c in (EEG, EEG2, EOG, EMG) if c in raw.ch_names]
    data = raw.get_data(picks=picks) * 1e6
    spe = int(sf * EPOCH)
    n_ep = min(n_ep, data.shape[1] // spe)
    data = data[:, : n_ep * spe].reshape(len(picks), n_ep, spe)
    f, p = sps.welch(data, fs=sf, nperseg=int(sf * 4), axis=-1)
    feats, names = [], []
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 16), "beta": (16, 30)}
    for ci, ch in enumerate(picks):
        if ch in (EEG, EEG2):
            for b, (lo, hi) in bands.items():
                m = (f >= lo) & (f < hi)
                feats.append(np.log10(p[ci][:, m].mean(1) + 1e-6))
                names.append(f"{ch.split()[-1]} {b}")
        elif ch == EOG:
            for b, (lo, hi) in {"slow": (0.3, 2), "fast": (2, 8)}.items():
                m = (f >= lo) & (f < hi)
                feats.append(np.log10(p[ci][:, m].mean(1) + 1e-6))
                names.append(f"EOG {b}")
        else:
            feats.append(np.log10(data[ci].var(1) + 1e-6))
            names.append("EMG power")
    return np.column_stack(feats), names


def icc_a1(x, y):
    """ICC(A,1): two-way random effects, absolute agreement, single measurement."""
    m = np.column_stack([x, y])
    n, k = m.shape
    gm = m.mean()
    msr = k * np.sum((m.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((m.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def chi2_stat(clusters, labels, k):
    tab = np.zeros((k, 2))
    np.add.at(tab, (clusters, labels.astype(int)), 1)
    exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.nansum((tab - exp) ** 2 / np.where(exp == 0, np.nan, exp))


def perm_p(clusters, labels, k, n=500):
    """Permutation p-value for the chi-square statistic (Phipson & Smyth 2010 correction)."""
    obs = chi2_stat(clusters, labels, k)
    null = np.array([chi2_stat(clusters, RNG.permutation(labels), k) for _ in range(n)])
    return (np.sum(null >= obs) + 1) / (n + 1), obs


def elastic_net():
    """Elastic-net logistic regression (scaled inputs). scikit-learn >= 1.8 selects the elastic
    net from l1_ratio alone and deprecates ``penalty``; older versions still need it."""
    kw = dict(solver="saga", C=1.0, l1_ratio=0.5, max_iter=5000, class_weight="balanced")
    major, minor = (int(v) for v in re.findall(r"\d+", SKLEARN_VERSION)[:2])
    if (major, minor) < (1, 8):
        kw["penalty"] = "elasticnet"
    return Pipeline([("s", StandardScaler()), ("m", LogisticRegression(**kw))])


# ----------------------------------------------------------------------------- data
def parse_age_sex(psg_path):
    """Age and sex from the EDF 'local patient identification' header field (bytes 8-88)."""
    try:
        hdr = open(psg_path, "rb").read(88)[8:88].decode("latin1")
        age = re.search(r"(\d+)\s*yr", hdr)
        low = hdr.lower()
        sex = "F" if "female" in low else ("M" if "male" in low else "?")
        return (int(age.group(1)) if age else None), sex
    except Exception:
        return None, "?"


def tech_hypnogram(raw):
    """Technician scoring from the EDF+ annotations as a 30-s integer hypnogram (-1 = unscored)."""
    n = int(raw.times[-1] // EPOCH)
    hyp = -np.ones(n, int)
    for a in raw.annotations:
        s = int(round(a["onset"] / EPOCH))
        d = int(round(a["duration"] / EPOCH))
        hyp[max(s, 0): s + d] = _MAP.get(a["description"], -1)
    return hyp


def crop_to_night(raw, hyp, pad=60):
    """Keep 30 minutes either side of the sleep period, the convention for Sleep-EDF."""
    sl = np.where(np.isin(hyp, SLEEP_CODES))[0]
    if len(sl) == 0:
        raise PipelineError("No sleep in the technician scoring: the night cannot be cropped.")
    s0, s1 = max(0, sl[0] - pad), min(len(hyp), sl[-1] + pad + 1)
    raw.crop(tmin=s0 * EPOCH, tmax=s1 * EPOCH - 1 / raw.info["sfreq"])
    return raw, hyp[s0:s1]


def load_sleep_edf(subjects):
    from mne.datasets.sleep_physionet.age import fetch_data
    files = fetch_data(subjects=subjects, recording=[1], on_missing="warn")
    nights = []
    for i, (psg, hyp_f) in enumerate(files):
        sid = Path(psg).name[:6]
        log(f"Loading {sid} ({i + 1}/{len(files)})")
        raw = mne.io.read_raw_edf(psg, preload=True, stim_channel="Event marker")
        raw.set_annotations(mne.read_annotations(hyp_f), emit_warning=False)
        age, sex = parse_age_sex(psg)
        try:
            raw, tech = crop_to_night(raw, tech_hypnogram(raw))
        except PipelineError as e:
            log(f"Skipping {sid}: {e}")
            continue
        nights.append(dict(id=sid, age=age, sex=sex, raw=raw, tech=tech, path=psg))
    if not nights:
        raise PipelineError("No usable Sleep-EDF nights were loaded.")
    return nights


def synthetic_nights(n=12):
    """Dry-run data: simulated nights with stage-dependent spectra. Not for reporting."""
    nights, sf = [], 100
    amp_by_stage = {0: (5, 6, 18, 2, 8), 1: (12, 18, 6, 2, 4), 2: (22, 14, 4, 10, 3), 3: (70, 10, 2, 3, 2), 4: (12, 16, 5, 2, 3)}
    emg_sd = {0: 20, 1: 10, 2: 8, 3: 7, 4: 2}
    desc = {0: "Sleep stage W", 1: "Sleep stage 1", 2: "Sleep stage 2", 3: "Sleep stage 3", 4: "Sleep stage R"}
    for s in range(n):
        age = int(RNG.uniform(25, 95))
        h = to_int(yasa.simulate_hypnogram(tib=560, seed=s))
        h = np.where((h == 3) & (RNG.random(len(h)) < (age - 20) / 120), 2, h)   # less N3 with age
        t = np.arange(EPOCH * sf) / sf
        sigs = {c: [] for c in (EEG, EEG2, EOG, EMG)}
        for st in h:
            amp = amp_by_stage[max(st, 0)]
            ph = RNG.uniform(0, 6.3, 5)
            eeg = (amp[0] * np.sin(2 * np.pi * 1.5 * t + ph[0]) + amp[1] * np.sin(2 * np.pi * 6 * t + ph[1]) +
                   amp[2] * np.sin(2 * np.pi * 10 * t + ph[2]) + amp[3] * np.sin(2 * np.pi * 13 * t + ph[3]) +
                   amp[4] * RNG.normal(0, 1, t.size))
            sigs[EEG].append(eeg)
            sigs[EEG2].append(0.7 * eeg + RNG.normal(0, 3, t.size))
            eog = RNG.normal(0, 4, t.size) + (40 * np.sign(np.sin(2 * np.pi * 0.4 * t + ph[4])) if st == 4
                                              else 10 * np.sin(2 * np.pi * 0.2 * t))
            sigs[EOG].append(eog)
            sigs[EMG].append(RNG.normal(0, emg_sd[max(st, 0)], t.size))
        data = np.vstack([np.concatenate(sigs[c]) for c in sigs]) * 1e-6
        info = mne.create_info(list(sigs), sf, ["eeg", "eeg", "eog", "emg"])
        raw = mne.io.RawArray(data, info)
        raw.set_annotations(mne.Annotations(np.arange(len(h)) * EPOCH, EPOCH, [desc[max(v, 0)] for v in h]))
        nights.append(dict(id=f"SIM{s:02d}", age=age, sex="?", raw=raw, tech=h.copy(), path=None))
    return nights


# ----------------------------------------------------------------------------- tests
R = {"tests": [], "meta": {}}


def record(test_id, stage, reqs, risks, what, result, passed, status=None):
    status = status or ("PASS" if bool(passed) else "FAIL")
    R["tests"].append(dict(id=test_id, stage=stage, reqs=reqs, risks=risks, what=what, result=result, status=status))
    log(f"{test_id}: {status} - {result}")


def _not_edf():
    OUT.mkdir(exist_ok=True)
    p = OUT / "not_an_edf.edf"
    p.write_text("this is a text file")
    return p


def v1_integration(nights):
    """V1: every real night runs on default settings; six deliberately broken inputs fail safely."""
    ok = 0
    for nt in nights:
        try:
            validate_recording(nt["raw"])
            ok += 1
        except PipelineError:
            pass
    base = nights[0]["raw"]
    cases = {
        "Missing EEG": lambda: base.copy().drop_channels([EEG]),
        "Missing EOG and EMG": lambda: base.copy().drop_channels([c for c in (EOG, EMG) if c in base.ch_names]),
        "Cut-short night (1 h)": lambda: base.copy().crop(0, 3600),
        "Wrong sampling rate (20 Hz)": lambda: base.copy().resample(20),
        "No annotations": lambda: base.copy().set_annotations(None),
        "File is not EDF": lambda: load_edf(_not_edf()),
    }
    rows = []
    for name, make in cases.items():
        try:
            unavailable = validate_recording(make())
            if unavailable:
                rows.append([name, "Ran, and named the features it could not compute: " + "; ".join(unavailable), True])
            else:
                rows.append([name, "Ran without any warning", False])
        except PipelineError as e:
            rows.append([name, f"Stopped with a clear error: {e}", True])
    R["v1_rows"] = rows
    n_handled = sum(r[2] for r in rows)
    passed = ok == len(nights) and n_handled == len(rows)
    record("V1", "Stage 1", "SPR2.1, SPR3.1, SPR1.5", "RSK1.5",
           f"{len(nights)} real nights on default settings, plus {len(rows)} deliberately broken inputs",
           f"{ok}/{len(nights)} valid nights accepted; {n_handled}/{len(rows)} broken inputs handled with a clear message", passed)


def v1b_hrv():
    """V1b: the NeuroKit2-based ECG/HRV module recovers a known heart rate from reference ECGs."""
    import neurokit2 as nk
    sf = 256
    rows, okc = [], 0
    for hr in (55, 70, 90):
        ecg = nk.ecg_simulate(duration=300, sampling_rate=sf, heart_rate=hr, heart_rate_std=3, random_state=hr)
        _, info = nk.ecg_process(ecg, sampling_rate=sf)
        peaks = np.asarray(info["ECG_R_Peaks"])
        est = 60 / np.mean(np.diff(peaks) / sf)
        hrv = nk.hrv_time(peaks, sampling_rate=sf)
        good = bool(abs(est - hr) <= 2)
        okc += good
        rows.append([hr, round(float(est), 1), round(float(hrv["HRV_RMSSD"].iloc[0]), 1), good])
        if hr == 70:
            fig, ax = plt.subplots(figsize=(8, 2.2))
            t = np.arange(len(ecg)) / sf
            m = t < 8
            ax.plot(t[m], ecg[m], color=COL["bad"], lw=0.8)
            pk = peaks[peaks < 8 * sf]
            ax.plot(pk / sf, ecg[pk], "o", color=COL["ink"], ms=4, label="Detected R-peaks")
            ax.set_xlabel("Time (s)")
            ax.set_yticks([])
            ax.legend(frameon=False, loc="upper right")
            ax.set_title("ECG module: R-peak detection on a reference signal (70 bpm)", loc="left", fontsize=10)
            save(fig, "v1b_ecg_peaks")
    R["hrv_rows"] = rows
    record("V1b", "Stage 1", "SPR2.1, SPR3.4", "RSK1.5",
           "ECG/HRV module on reference ECGs with known heart rate (Sleep-EDF has no ECG; repeat on HMC)",
           f"Detected heart rate within 2 bpm of truth in {okc}/{len(rows)} signals", okc == len(rows))


def v6_shipped_code():
    """V6: unit checks on the functions that run_pipeline.py actually ships (statistics and cohort split)."""
    from sklearn.metrics import roc_auc_score as auc
    from statsmodels.stats.multitest import multipletests
    r = np.random.default_rng(SEED)
    checks = []
    p = r.uniform(0, 1, 40) ** 2
    checks.append(("Benjamini-Hochberg equals statsmodels", bool(np.allclose(rp.bh_fdr(p), multipletests(p, method="fdr_bh")[1]))))
    y = np.r_[np.ones(30), np.zeros(50)].astype(int)
    p1, p2 = r.normal(size=80) + 1.0 * y, r.normal(size=80) + 0.3 * y
    a1, a2, pd_ = rp.delong(y, p1, p2)
    checks.append(("DeLong AUCs equal sklearn", bool(abs(a1 - auc(y, p1)) < 1e-9 and abs(a2 - auc(y, p2)) < 1e-9)))
    checks.append(("DeLong p is 1 for identical predictors", bool(rp.delong(y, p1, p1)[2] == 1.0)))
    c = r.integers(0, 3, 90)
    yy = (r.random(90) < 0.4).astype(int)
    chi, tab = rp.chi2_stat(c, yy, 3)
    from scipy.stats import chi2_contingency
    checks.append(("Chi-square equals scipy", bool(abs(chi - chi2_contingency(tab, correction=False)[0]) < 1e-9)))
    ids = [f"P{i:03d}" for i in range(100)]
    lab = (np.arange(100) % 4 == 0).astype(int)
    dev, te = rp.stratified_split(ids, lab, 0.2, SEED)
    dev2, te2 = rp.stratified_split(ids, lab, 0.2, SEED)
    ymap = dict(zip(ids, lab))
    checks.append(("Split: 80/20, disjoint, repeatable, stratified", bool(
        len(te) == 20 and not set(dev) & set(te) and (dev, te) == (dev2, te2)
        and sum(ymap[i] for i in te) == 5)))
    x = r.normal(size=60) + 0.8 * (np.arange(60) < 30)
    beta, lo, hi, pv = rp.adjusted_ols(x, r.normal(size=(60, 2)), (np.arange(60) < 30).astype(int))
    checks.append(("Eq. 4 regression recovers a planted effect", bool(lo < beta < hi and pv < 0.05)))
    checks.append(("Three-epoch rule ignores a lone REM epoch", rp.rem_onset_epoch([2, 2, 4, 2, 4, 4, 4]) == 4))
    R["v6_rows"] = [[n, ok] for n, ok in checks]
    record("V6", "Stages 1-5", "SPR3.2", "RSK1.1, RSK1.3", "Unit checks on the shipped statistics and split code",
           f"{sum(ok for _, ok in checks)}/{len(checks)} checks passed", all(ok for _, ok in checks))


def v2_unit():
    """V2: REM latency on hand-built hypnograms with a known answer, including edge cases."""
    cases = []
    for i in range(34):                                  # random but known hypnograms
        onset, lat = int(RNG.integers(2, 60)), int(RNG.integers(0, 260))
        h = [0] * onset + list(RNG.choice([1, 2, 3], lat)) + [4] * 10 + list(RNG.choice([0, 1, 2, 4], 200))
        cases.append((f"random {i + 1}", h, lat * MIN_PER_EPOCH))
    cases += [("no REM at all", [0] * 10 + [2] * 300, None),
              ("REM straight after onset", [0] * 10 + [4] * 5 + [2] * 50, 0.0),
              ("wake before first REM", [0] * 4 + [2] * 40 + [0] * 20 + [4] * 6, 30.0),
              ("R&K stage 4 present", [0] * 2 + [2] * 10 + [3] * 30 + [4] * 4, 20.0),
              ("never falls asleep", [0] * 100, None),
              ("unscored epochs at start", [-1] * 6 + [0] * 4 + [1] * 8 + [4] * 3, 4.0),
              ("single REM epoch ignored (3-epoch rule)", [0] * 4 + [2] * 10 + [4] + [2] * 10 + [4] * 5, 10.5),
              ("two REM epochs only", [0] * 4 + [2] * 40 + [4] * 2 + [2] * 10, None)]
    rows, ok = [], 0
    for name, h, truth in cases:
        got = rem_latency(h)
        good = bool((got is None and truth is None) or
                    (got is not None and truth is not None and abs(got - truth) < 1e-9))
        ok += good
        if not name.startswith("random") or not good:
            rows.append([name, "none" if truth is None else f"{truth:.1f}", "none" if got is None else f"{got:.1f}", good])
    R["v2_rows"] = rows
    R["v2_n_cases"] = len(cases)
    record("V2", "Stage 1", "SPR3.4, SPR1.3", "RSK1.1, RSK1.2, RSK1.5",
           f"{len(cases)} hand-built hypnograms with known REM latency, including edge cases",
           f"{ok}/{len(cases)} exact to the epoch; nights without REM return 'none'", ok == len(cases))


def _bootstrap_mean(vals, n_boot=2000):
    b = [np.mean(RNG.choice(vals, len(vals))) for _ in range(n_boot)]
    return float(np.mean(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def v3_staging(nights):
    """V3: automated staging (YASA) vs technician scoring. Pass/fail is judged on the 60+ group."""
    for nt in nights:
        log(f"YASA staging {nt['id']}")
        pipe = stage_with_yasa(nt["raw"], EEG, EOG, EMG)
        n = min(len(pipe), len(nt["tech"]))
        nt["pipe"], nt["tech"] = pipe[:n], nt["tech"][:n]
    rows, yt, yp = [], [], []
    for nt in nights:
        m = nt["tech"] >= 0
        t, p = nt["tech"][m], nt["pipe"][m]
        yt += list(t)
        yp += list(p)
        rows.append(dict(id=nt["id"], age=nt["age"], acc=float(accuracy_score(t, p)),
                         kappa=float(cohen_kappa_score(t, p)), rem_f1=float(f1_score(t == 4, p == 4, zero_division=0))))
    R["v3_rows"] = rows
    older = [r for r in rows if r["age"] and r["age"] >= 60] or rows
    summ = {key: _bootstrap_mean([r[key] for r in older]) for key in ("acc", "kappa", "rem_f1")}
    R["v3_summary"] = {k: [round(x, 3) for x in v] for k, v in summ.items()}
    R["v3_summary_all"] = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("acc", "kappa", "rem_f1")}
    passed = summ["acc"][0] >= CRIT["acc"] and summ["kappa"][0] >= CRIT["kappa"] and summ["rem_f1"][0] >= CRIT["rem_f1"]
    record("V3", "Stage 1", "SPR2.1, SPR3.4", "RSK1.1, RSK1.2",
           f"YASA staging vs technician scoring, {len(older)} nights aged 60+ ({len(rows)} nights in total)",
           f"Agreement {summ['acc'][0]:.1%} (95% CI {summ['acc'][1]:.1%}-{summ['acc'][2]:.1%}), "
           f"kappa {summ['kappa'][0]:.2f}, REM F1 {summ['rem_f1'][0]:.2f}", passed)

    # figure 1: example hypnogram for the oldest participant
    ex = max(nights, key=lambda d: d["age"] or 0)
    fig, ax = plt.subplots(figsize=(9, 2.6))
    rowmap = {0: 4, 4: 3, 1: 2, 2: 1, 3: 0}
    for h, c, lab, lw in ((ex["tech"], COL["ink"], "Technician", 1.4), (ex["pipe"], COL["acc"], "Pipeline (YASA)", 1.0)):
        y = [rowmap.get(v, np.nan) for v in h]
        ax.step(np.arange(len(h)) / EPOCHS_PER_HOUR, y, where="post", color=c, lw=lw, label=lab, alpha=0.85)
    ax.set_yticks(range(5))
    ax.set_yticklabels(["N3", "N2", "N1", "REM", "W"])
    ax.set_xlabel("Hours from start of recording window")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    ax.set_title(f"V3 - Hypnogram, {ex['id']} (age {ex['age']}): pipeline vs technician", loc="left", fontsize=10)
    save(fig, "v3_hypnogram_example")

    # figure 2: confusion matrix over all nights
    cm = confusion_matrix(yt, yp, labels=range(5))
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=7,
                    color="white" if cmn[i, j] > 0.6 else COL["ink"])
    ax.set_xticks(range(5))
    ax.set_xticklabels(STAGES)
    ax.set_yticks(range(5))
    ax.set_yticklabels(STAGES)
    ax.set_xlabel("Pipeline")
    ax.set_ylabel("Technician")
    ax.set_title("V3 - Confusion matrix (all nights)", loc="left", fontsize=10)
    save(fig, "v3_confusion_matrix")
    yt, yp = np.array(yt), np.array(yp)
    R["v3_per_stage_f1"] = {s: round(float(f1_score(yt == i, yp == i, zero_division=0)), 3) for i, s in enumerate(STAGES)}

    # figure 3: per-night agreement against age
    fig, axs = plt.subplots(1, 2, figsize=(8, 2.8))
    ages = [r["age"] or 0 for r in rows]
    for ax, key, lab in ((axs[0], "acc", "Epoch agreement"), (axs[1], "kappa", "Cohen's kappa")):
        ax.scatter(ages, [r[key] for r in rows], color=COL["acc"], s=18)
        ax.axhline(CRIT[key], color=COL["bad"], ls="--", lw=1)
        ax.text(min(ages), CRIT[key] + 0.01, "pass line", color=COL["bad"], fontsize=7)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(lab)
        ax.set_ylim(0.3, 1)
    axs[0].set_title("V3 - Agreement for each night by age", loc="left", fontsize=10)
    save(fig, "v3_by_age")


def v4_rem_latency(nights):
    """V4: Bland-Altman agreement of REM latency, pipeline vs technician (uses the V3 staging)."""
    pairs = [(rem_latency(nt["tech"]), rem_latency(nt["pipe"]), nt["id"]) for nt in nights]
    pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
    if len(pairs) < 3:
        record("V4", "Stage 1", "SPR3.4, SPR1.3", "RSK1.1, RSK1.2", "REM latency from pipeline vs technician scoring",
               f"Only {len(pairs)} nights with REM in both scorings; at least 3 are needed", False, status="NOT RUN")
        return
    t = np.array([p[0] for p in pairs])
    p = np.array([p[1] for p in pairs])
    d = p - t
    mean = (p + t) / 2
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    loa = (bias - 1.96 * sd, bias + 1.96 * sd)
    icc = float(icc_a1(t, p))
    outliers = [f"{pairs[i][2]} ({d[i]:+.0f} min)" for i in np.where(np.abs(d) > CRIT["ba_loa"])[0]]
    R["v4"] = dict(n=len(pairs), bias=round(bias, 1), loa=[round(loa[0], 1), round(loa[1], 1)], icc=round(icc, 3), outliers=outliers)
    passed = abs(bias) <= CRIT["ba_bias"] and loa[0] >= -CRIT["ba_loa"] and loa[1] <= CRIT["ba_loa"] and icc >= CRIT["icc"]
    record("V4", "Stage 1", "SPR3.4, SPR1.3", "RSK1.1, RSK1.2",
           f"REM latency from pipeline vs technician scoring, {len(pairs)} nights",
           f"Bias {bias:+.1f} min, 95% limits {loa[0]:.1f} to {loa[1]:.1f} min, ICC {icc:.2f}; "
           f"{len(outliers)} nights >{CRIT['ba_loa']:.0f} min out", passed)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.axhspan(-CRIT["ba_loa"], CRIT["ba_loa"], color=COL["good"], alpha=0.08, label=f"acceptance band (+/-{CRIT['ba_loa']:.0f} min)")
    ax.scatter(mean, d, color=COL["acc"], s=20)
    ax.axhline(bias, color=COL["ink"], lw=1.2, label=f"bias {bias:+.1f}")
    for v in loa:
        ax.axhline(v, color=COL["warn"], ls="--", lw=1)
    ax.set_xlabel("Mean REM latency of the two methods (min)")
    ax.set_ylabel("Pipeline - technician (min)")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.set_title("V4 - Bland-Altman: REM latency", loc="left", fontsize=10)
    save(fig, "v4_bland_altman")


def v5_plausibility(nights):
    """V5: group means of the pipeline's sleep measures (60+ group) against reference ranges."""
    for nt in nights:
        nt["stats"] = sleep_stats(nt["pipe"])
    older = [nt for nt in nights if nt["age"] and nt["age"] >= 60]
    if not older:
        record("V5", "Stage 1", "SPR3.4", "RSK1.5", "Sleep measures vs published ranges for adults 60+",
               "No nights aged 60+ in this run", False, status="NOT RUN")
        return
    keys = {"TST (min)": "TST", "SE (%)": "SE", "REM (%)": "REMp", "N3 (%)": "N3p", "REM latency (min)": "REMLAT"}
    rows, ok = [], 0
    for lab, k in keys.items():
        v = float(np.nanmean([nt["stats"][k] for nt in older]))
        lo, hi = PLAUSIBLE_60PLUS[lab]
        good = bool(lo <= v <= hi)
        ok += good
        rows.append([lab, round(v, 1), f"{lo}-{hi}", good])
    R["v5_rows"] = rows
    record("V5", "Stage 1", "SPR3.4", "RSK1.5", f"Pipeline sleep measures, {len(older)} nights aged 60+, vs reference ranges",
           f"{ok}/{len(rows)} group means inside the reference range (ranges informed by Ohayon 2004)", ok == len(rows))


def s2_positive_control(nights):
    """S2: the Stage 2 statistics recover known effects of age on sleep (a positive control)."""
    ages = np.array([nt["age"] or np.nan for nt in nights], float)
    ok_mask = ~np.isnan(ages)
    out, figs = [], []
    for key, lab, expect in (("N3p", "Deep sleep N3 (% of sleep)", -1), ("SE", "Sleep efficiency (%)", -1),
                             ("WASO", "Wake after sleep onset (min)", +1)):
        y = np.array([nt["stats"][key] for nt in nights], float)
        m = ok_mask & ~np.isnan(y)
        rho, p = stats.spearmanr(ages[m], y[m])
        good = bool(np.sign(rho) == expect and p < 0.05)
        out.append([lab, "falls" if expect < 0 else "rises", round(float(rho), 2), float(p), good])
        figs.append((lab, ages[m], y[m], rho, p))
    R["s2_rows"] = out
    passed = sum(r[4] for r in out) >= 2
    record("S2", "Stage 2", "SPR3.4", "RSK1.3",
           "Positive control: does the statistics code recover known effects of age on sleep?",
           "; ".join(f"{r[0].split(' (')[0]} {r[1]} with age (rho {r[2]:+.2f}, p {r[3]:.3g})" for r in out), passed)
    fig, axs = plt.subplots(1, 3, figsize=(9.5, 2.8))
    for ax, (lab, x, y, rho, p) in zip(axs, figs):
        ax.scatter(x, y, s=16, color=COL["acc"])
        if len(x) > 2:
            b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 10)
            ax.plot(xs, np.polyval(b, xs), color=COL["ink"], lw=1)
        ax.set_xlabel("Age (years)")
        ax.set_ylabel(lab, fontsize=8)
        ax.set_title(f"rho {rho:+.2f}, p {p:.2g}", fontsize=8, loc="left")
    fig.suptitle("Stage 2 - Positive control: known age effects recovered from real nights", x=0.01, ha="left", fontsize=10)
    save(fig, "s2_positive_control")


def s3_discovery(nights):
    """S3: an autoencoder trained without labels learns sleep structure; k-means groups the nights."""
    X, y, g = [], [], []
    names = []
    for i, nt in enumerate(nights):
        f, names = epoch_features(nt["raw"], len(nt["tech"]))
        X.append(f)
        y.append(nt["tech"][: len(f)])
        g.append(np.full(len(f), i))
    X, y, g = np.vstack(X), np.concatenate(y), np.concatenate(g)
    keep = np.isfinite(X).all(1)
    X, y, g = X[keep], y[keep], g[keep]
    test_subj = RNG.choice(len(nights), max(2, len(nights) // 4), replace=False)
    te = np.isin(g, test_subj)
    tr = ~te
    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X)
    ae = MLPRegressor(hidden_layer_sizes=(32, LATENT_DIM, 32), activation="tanh", max_iter=400,
                      early_stopping=True, random_state=0).fit(Xs[tr], Xs[tr])

    def encode(Z):
        a = np.tanh(Z @ ae.coefs_[0] + ae.intercepts_[0])
        return np.tanh(a @ ae.coefs_[1] + ae.intercepts_[1])

    mse_ae = float(np.mean((ae.predict(Xs[te]) - Xs[te]) ** 2))
    pca = PCA(LATENT_DIM).fit(Xs[tr])
    mse_pca = float(np.mean((pca.inverse_transform(pca.transform(Xs[te])) - Xs[te]) ** 2))
    L = encode(Xs)
    lab = y >= 0
    knn = KNeighborsClassifier(15).fit(L[tr & lab], y[tr & lab])
    bacc = float(balanced_accuracy_score(y[te & lab], knn.predict(L[te & lab])))

    # per-person latent profile (mean latent vector in N2, N3 and REM)
    prof = []
    for i in range(len(nights)):
        row = []
        for st in (2, 3, 4):
            m = (g == i) & (y == st)
            row.append(L[m].mean(0) if m.sum() >= 5 else L[g == i].mean(0))
        prof.append(np.concatenate(row))
    P = StandardScaler().fit_transform(np.array(prof))
    k_values = range(MIN_K, min(MAX_K, len(nights) - 1) + 1)       # silhouette needs 2 <= k <= n - 1
    sil = {k: float(silhouette_score(P, KMeans(k, n_init=20, random_state=0).fit_predict(P))) for k in k_values}
    best_k = max(sil, key=sil.get)
    clusters = KMeans(best_k, n_init=20, random_state=0).fit_predict(P)
    choice = "autoencoder kept" if mse_ae <= mse_pca else "PCA fallback used, as the design specifies"
    R["s3"] = dict(n_epochs=int(len(X)), features=names, latent_dim=LATENT_DIM, mse_ae=round(mse_ae, 4),
                   mse_pca=round(mse_pca, 4), stage_bacc=round(bacc, 3), sil={str(k): round(v, 3) for k, v in sil.items()},
                   best_k=int(best_k), choice=choice)
    passed = bacc >= 0.5 and mse_ae <= mse_pca            # pass criteria from the V&V plan: >= 50% and beats PCA
    record("S3", "Stage 3", "SPR2.2", "RSK1.3",
           f"Autoencoder ({LATENT_DIM}-number bottleneck) trained on {int(tr.sum())} epochs from "
           f"{len(nights) - len(test_subj)} nights, no labels; tested on {len(test_subj)} unseen nights",
           f"Latent features predict sleep stage with balanced accuracy {bacc:.0%} (chance 20%); "
           f"rebuild error {mse_ae:.3f} vs PCA {mse_pca:.3f}, so {choice}; k-means chose k={best_k}", passed)
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.3), gridspec_kw=dict(width_ratios=[1.6, 1]))
    xy = PCA(2).fit_transform(L[lab])
    yy = y[lab]
    sub = RNG.choice(len(xy), min(6000, len(xy)), replace=False)
    for st in range(5):
        m = yy[sub] == st
        axs[0].scatter(xy[sub][m, 0], xy[sub][m, 1], s=3, color=COL["stages"][st], label=STAGES[st], alpha=0.5)
    axs[0].legend(frameon=False, markerscale=4, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axs[0].set_xticks([])
    axs[0].set_yticks([])
    axs[0].set_title("Stage 3 - Latent space, coloured by technician stage\n(the network never saw these labels)", loc="left", fontsize=9)
    axs[1].bar(["Autoencoder", "PCA"], [mse_ae, mse_pca], color=[COL["acc"], COL["sub"]])
    axs[1].set_ylabel("Rebuild error on unseen nights (MSE)")
    axs[1].set_title("Rebuild error", loc="left", fontsize=9)
    save(fig, "s3_latent_space")
    return P, clusters, best_k


def s4_calibration(P, clusters, k):
    """S4: random labels should give ~5% false positives; a planted effect should be found.
    Checked at the real sample size and at a study-sized cohort (people resampled to n=120)."""
    sizes = np.bincount(clusters, minlength=k)
    target = int(np.argmin(np.abs(sizes - len(clusters) / 2)))      # most balanced cluster carries the effect
    res = {}
    for label, n in (("real", len(clusters)), ("study", 120)):
        cl = clusters if label == "real" else clusters[RNG.integers(0, len(clusters), n)]
        null_p = [perm_p(cl, (RNG.random(n) < 0.33).astype(int), k, n=300)[0] for _ in range(300)]
        det = [perm_p(cl, (RNG.random(n) < np.where(cl == target, 0.65, 0.15)).astype(int), k, n=300)[0] < 0.05
               for _ in range(200)]
        res[label] = dict(n=n, fpr=float(np.mean(np.array(null_p) < 0.05)), power=float(np.mean(det)), null_p=null_p)
    R["s4"] = {kk: dict(n=v["n"], fpr=round(v["fpr"], 3), power=round(v["power"], 3)) for kk, v in res.items()}
    R["s4"]["k"] = int(k)
    st, rl = res["study"], res["real"]
    record("S4a", "Stage 4", "SPR3.2", "RSK1.3", "300 sets of RANDOM labels on the real cluster structure (no true effect)",
           f"False-positive rate {st['fpr']:.1%} at n=120 ({rl['fpr']:.1%} at the real n={rl['n']}); "
           f"must not exceed 5% (allowing simulation error, at most 8%)", st["fpr"] <= 0.08)
    record("S4b", "Stage 4", "SPR3.2", "RSK1.1", "200 sets of labels with a PLANTED effect (one cluster 65% vs 15% anxious)",
           f"Detected in {st['power']:.0%} of runs at n=120 ({rl['power']:.0%} at the real n={rl['n']}, "
           f"which shows why the clinical study needs a properly sized cohort)", st["power"] >= 0.8)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 2.8))
    axs[0].hist(st["null_p"], bins=20, range=(0, 1), color=COL["sub"])
    axs[0].axvline(0.05, color=COL["bad"], ls="--")
    axs[0].set_xlabel("Permutation p-value (random labels, n=120)")
    axs[0].set_title(f"Stage 4 - No effect: {st['fpr']:.0%} false positives", loc="left", fontsize=9)
    axs[1].bar([f"Real n={rl['n']}", "Study n=120"], [rl["power"], st["power"]], color=[COL["sub"], COL["good"]])
    axs[1].axhline(0.8, color=COL["bad"], ls="--", lw=1)
    axs[1].set_ylim(0, 1)
    axs[1].set_ylabel("Share of runs detecting the effect")
    axs[1].set_title("Planted effect: detection rate", loc="left", fontsize=9)
    save(fig, "s4_calibration")


def s5_classifier(nights, P):
    """S5: the elastic-net classifier has no leakage, finds nothing in random labels and finds a planted effect."""
    keys = ["SOL", "REMLAT", "SE", "WASO", "N3p", "REMp"]
    F = np.array([[nt["stats"][k] for k in keys] for nt in nights], float)
    F = np.where(np.isnan(F), np.nanmedian(F, 0), F)
    ages = np.array([nt["age"] or 60 for nt in nights], float)
    X = np.column_stack([F, ages, PCA(min(3, P.shape[1])).fit_transform(P)])
    groups = np.arange(len(nights))
    z = StandardScaler().fit_transform(X)

    def cv_auc(y, reps=5):
        aucs, leak = [], 0
        n_splits = max(2, min(5, int(np.bincount(y).min())))
        p = np.zeros(len(y))
        for r in range(reps):
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=r)
            p = np.zeros(len(y))
            for tr, te in cv.split(X, y, groups):
                leak += len(set(groups[tr]) & set(groups[te]))
                p[te] = elastic_net().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y, p))
        return float(np.mean(aucs)), p, leak

    null = []
    for _ in range(60):
        y = (RNG.random(len(X)) < 0.4).astype(int)
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        null.append(cv_auc(y, reps=1)[0])
    logit = 2.2 * (-z[:, 2]) + 1.4 * z[:, 0] - 0.4        # planted link: lower SE and longer SOL -> anxious
    for _ in range(20):                                    # redraw only if a class is too small to cross-validate
        y_pl = (RNG.random(len(X)) < 1 / (1 + np.exp(-logit))).astype(int)
        if y_pl.sum() >= 3 and (1 - y_pl).sum() >= 3:
            break
    auc_pl, p_pl, leak = cv_auc(y_pl)
    null_mean = float(np.mean(null))
    R["s5"] = dict(null_mean=round(null_mean, 3), null_sd=round(float(np.std(null)), 3), planted_auc=round(auc_pl, 3),
                   leak=int(leak), n=len(X))
    record("S5a", "Stage 5", "SPR3.2", "RSK2.1", "Leakage check: participants in training vs test folds",
           f"{leak} participants appeared in both training and test folds", leak == 0)
    record("S5b", "Stage 5", "SPR3.2", "RSK1.2", f"Elastic-net model with {len(null)} sets of RANDOM labels (no true effect)",
           f"Mean cross-validated AUROC {null_mean:.2f} (expected 0.50)", abs(null_mean - 0.5) <= 0.06)
    record("S5c", "Stage 5", "SPR3.2", "RSK1.1", "Elastic-net model with a PLANTED link to sleep efficiency and sleep onset",
           f"Cross-validated AUROC {auc_pl:.2f}", auc_pl >= 0.7)
    fig, axs = plt.subplots(1, 2, figsize=(8.5, 3))
    axs[0].hist(null, bins=15, color=COL["sub"])
    axs[0].axvline(0.5, color=COL["ink"], ls="--")
    axs[0].set_xlabel("AUROC with random labels")
    axs[0].set_title("Stage 5 - No effect: AUROC centres on 0.5", loc="left", fontsize=9)
    fpr, tpr, _ = roc_curve(y_pl, p_pl)
    axs[1].plot(fpr, tpr, color=COL["acc"], lw=2)
    axs[1].plot([0, 1], [0, 1], color=COL["grid"], ls="--")
    axs[1].set_xlabel("1 - specificity")
    axs[1].set_ylabel("Sensitivity")
    axs[1].set_title(f"Planted effect recovered: AUROC {auc_pl:.2f}", loc="left", fontsize=9)
    save(fig, "s5_classifier")


def s6_montages(nights):
    """S6: sensors removed in software (wearable-style sets); re-staged and compared with the technician."""
    montages = {"EEG + EOG (headband)": (EEG, EOG, None), "Frontal EEG only": (EEG, None, None),
                "Parietal EEG only": (EEG2, None, None)}
    rows, ba = [], {}
    for name, (e, o, m) in montages.items():
        ks, tl, pl = [], [], []
        for nt in nights:
            if e not in nt["raw"].ch_names:
                continue
            h = stage_with_yasa(nt["raw"], e, o, m)[: len(nt["tech"])]
            tech = nt["tech"][: len(h)]
            msk = tech >= 0
            ks.append(cohen_kappa_score(tech[msk], h[msk]))
            a, b = rem_latency(tech), rem_latency(h)
            if a is not None and b is not None:
                tl.append(a)
                pl.append(b)
        if not ks:
            log(f"S6 montage {name}: no night has channel {e}; skipped")
            continue
        tl, pl = np.array(tl, float), np.array(pl, float)
        d = pl - tl
        if len(d) >= 3:
            bias, sd = float(d.mean()), float(d.std(ddof=1))
            loa = (bias - 1.96 * sd, bias + 1.96 * sd)
            icc = float(icc_a1(tl, pl))
        else:
            bias, loa, icc = float(d.mean()) if len(d) else np.nan, (np.nan, np.nan), np.nan
        rows.append([name, round(float(np.mean(ks)), 3), round(bias, 1), [round(float(loa[0]), 1), round(float(loa[1]), 1)], round(icc, 3)])
        ba[name] = ((pl + tl) / 2, d)
        log(f"S6 montage {name}: kappa {np.mean(ks):.2f}")
    full_k = float(np.mean([r["kappa"] for r in R["v3_rows"]]))
    R["s6_rows"] = rows
    R["s6_full_kappa"] = round(full_k, 3)
    best = max(rows, key=lambda r: r[1])
    primary = next((r for r in rows if r[0].startswith("EEG + EOG")), best)    # pre-specified reduced set
    record("S6", "Stage 6", "SPR3.5", "RSK2.1", "Sensors removed in software from the same real nights; re-staged and compared with the technician",
           "; ".join(f"{r[0]} kappa {r[1]:.2f}" for r in rows) + f" (full montage {full_k:.2f}). Pass judged on the pre-specified set ({primary[0]}); best set: {best[0]}",
           primary[1] >= CRIT["kappa"])
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.1), gridspec_kw=dict(width_ratios=[1, 1.2]))
    names = ["Full PSG"] + [r[0] for r in rows]
    vals = [full_k] + [r[1] for r in rows]
    axs[0].barh(names[::-1], vals[::-1], color=[COL["acc"] if v >= CRIT["kappa"] else COL["sub"] for v in vals[::-1]])
    axs[0].axvline(CRIT["kappa"], color=COL["bad"], ls="--", lw=1)
    axs[0].set_xlim(0, 1)
    axs[0].set_xlabel("Cohen's kappa vs technician")
    axs[0].set_title("Stage 6 - Staging agreement by sensor set", loc="left", fontsize=9)
    m, d = ba[best[0]]
    axs[1].axhspan(-CRIT["ba_loa"], CRIT["ba_loa"], color=COL["good"], alpha=0.08)
    axs[1].scatter(m, d, s=16, color=COL["acc"])
    axs[1].axhline(d.mean(), color=COL["ink"])
    for v in best[3]:
        if np.isfinite(v):
            axs[1].axhline(v, color=COL["warn"], ls="--")
    axs[1].set_xlabel("Mean REM latency (min)")
    axs[1].set_ylabel("Reduced - technician (min)")
    axs[1].set_title(f"REM latency, {best[0]}", loc="left", fontsize=9)
    save(fig, "s6_montages")


# ----------------------------------------------------------------------------- report
def html_report(nights, synthetic):
    def badge(s):
        c = {"PASS": "#1f8a5b", "FAIL": "#c2413b"}.get(s, "#b7791f")
        return (f'<span style="color:{c};border:1px solid {c};border-radius:999px;padding:1px 9px;'
                f'font-weight:600;font-size:12px">{s}</span>')

    def cell(c):
        """Render one table cell: pass/fail flags become badges, numbers and ranges print cleanly."""
        if isinstance(c, (bool, np.bool_)):
            return badge("PASS" if c else "FAIL")
        if isinstance(c, (list, tuple)):
            return " to ".join(cell(x) for x in c)
        if isinstance(c, (float, np.floating)):
            return "n/a" if not np.isfinite(c) else f"{float(c):g}"
        if isinstance(c, np.integer):
            return str(int(c))
        return str(c)

    def img(n, cap):
        return (f'<figure><img src="data:image/png;base64,{FIGS[n]}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
                if n in FIGS else "")

    def table(head, rows):
        return ("<table><tr>" + "".join(f"<th>{h}</th>" for h in head) + "</tr>" +
                "".join("<tr>" + "".join(f"<td>{cell(c)}</td>" for c in r) + "</tr>" for r in rows) + "</table>")

    ages = [nt["age"] for nt in nights if nt["age"]]
    n_pass = sum(t["status"] == "PASS" for t in R["tests"])
    warn = ('<p class="warn"><b>Synthetic dry run.</b> These nights are simulated to test that the script runs. '
            'Do not report these numbers. Run without --synthetic for real results.</p>') if synthetic else ""
    v3 = R.get("v3_summary", {})
    data_desc = "SIMULATED dry run" if synthetic else "Sleep-EDF Expanded (sleep-cassette, PhysioNet), technician-scored, no anxiety labels"
    age_desc = f"{min(ages)}-{max(ages)} ({sum(a >= 60 for a in ages)} aged 60+)" if ages else "unknown"
    n_v2 = R.get("v2_n_cases", 0)
    per_stage = ", ".join(f"{k} {v:.2f}" for k, v in R.get("v3_per_stage_f1", {}).items())
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Sleeping Minds V&amp;V Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{{font:14px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;color:#1b2a38;background:#f6f8fa;margin:0}}
main{{max-width:1050px;margin:0 auto;padding:28px 18px 60px}} h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:18px;margin:34px 0 8px;border-top:1px solid #dfe5eb;padding-top:18px}}
.sub{{color:#5b6b7c}} table{{border-collapse:collapse;width:100%;background:#fff;font-size:13px;margin:8px 0}} th,td{{border-bottom:1px solid #dfe5eb;padding:7px 8px;text-align:left;vertical-align:top}}
th{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#5b6b7c}} figure{{margin:12px 0;background:#fff;border:1px solid #dfe5eb;border-radius:8px;padding:10px}}
img{{max-width:100%}} figcaption{{font-size:12px;color:#5b6b7c;margin-top:4px}} .warn{{background:#fff4e0;border-left:3px solid #b7791f;padding:10px 12px}}
.kpis{{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}} .kpi{{background:#fff;border:1px solid #dfe5eb;border-radius:8px;padding:10px 14px;min-width:150px}} .kpi b{{font-size:20px;display:block}}</style></head><body><main>
<h1>Sleeping Minds - Verification report</h1>
<p class="sub">Generated {time.strftime('%d %b %Y %H:%M')} · Data: {data_desc} · {len(nights)} nights, ages {age_desc}</p>
{warn}
<div class="kpis"><div class="kpi"><b>{n_pass}/{len(R['tests'])}</b>tests passed</div>
<div class="kpi"><b>{v3.get('acc', [0])[0]:.1%}</b>staging agreement (60+)</div><div class="kpi"><b>{v3.get('kappa', [0])[0]:.2f}</b>Cohen's kappa (60+)</div>
<div class="kpi"><b>{R.get('s4', {}).get('study', {}).get('fpr', 0):.0%}</b>false positives, Stage 4</div></div>
<p>This report verifies that each stage of the software works correctly on real overnight recordings. Sleep-EDF has no anxiety labels, so Stages 4 and 5 are tested with synthetic labels on the real features: random labels (the tests should find nothing) and planted effects (the tests should find them). Validation of anxiety biomarkers remains the deferred clinical study (Validation 4).</p>
<h2>Summary: Verification Test Plan with results</h2>
<table><tr><th>Test</th><th>Stage</th><th>Reqs.</th><th>Risks</th><th>What was done</th><th>Result</th><th>Pass/Fail</th></tr>
{''.join(f"<tr><td><b>{t['id']}</b></td><td>{t['stage']}</td><td>{t['reqs']}</td><td>{t['risks']}</td><td>{t['what']}</td><td>{t['result']}</td><td>{badge(t['status'])}</td></tr>" for t in R['tests'])}</table>
<h2>Stage 1 - Feature extraction</h2>
<p><b>V1 Integration.</b> Real nights on default settings, then deliberately broken inputs:</p>{table(['Broken input', 'What the pipeline did', 'Pass/Fail'], R.get('v1_rows', []))}
<p><b>V1b ECG/HRV module.</b> Reference ECGs with known heart rate:</p>{table(['True HR (bpm)', 'Detected HR', 'RMSSD (ms)', 'Pass/Fail'], R.get('hrv_rows', []))}
{img('v1b_ecg_peaks', 'R-peaks detected by the NeuroKit2-based ECG module.')}
<p><b>V2 REM latency unit test.</b> Edge cases shown (all {n_v2} cases were checked):</p>{table(['Case', 'True (min)', 'Pipeline (min)', 'Pass/Fail'], R.get('v2_rows', []))}
{img('v3_hypnogram_example', 'V3: pipeline staging (blue) overlaid on technician scoring (black) for the oldest participant.')}
{img('v3_confusion_matrix', 'V3: confusion matrix, all nights. Rows are technician stages; values are the share of each technician stage.')}
<p>Per-stage F1: {per_stage}</p>
{img('v3_by_age', 'V3: agreement for each night against age, with the pass line.')}
{img('v4_bland_altman', 'V4: Bland-Altman agreement for REM latency. Shaded: ±15 min acceptance band.')}
<p>V4 nights more than 15 min apart: {', '.join(R.get('v4', {}).get('outliers', [])) or 'none'}</p>
<p><b>V5 Plausibility (60+).</b></p>{table(['Measure', 'Group mean', 'Reference range', 'Pass/Fail'], R.get('v5_rows', []))}
<p><b>V6 Shipped-code checks.</b></p>{table(['Check', 'Pass/Fail'], R.get('v6_rows', []))}
<h2>Stage 2 - Testing literature features (positive control)</h2>
{img('s2_positive_control', 'Known effects of ageing recovered by the Stage 2 statistics code from the real nights.')}
<h2>Stage 3 - Discovery</h2>
{img('s3_latent_space', 'Latent features learned without labels separate the sleep stages; the autoencoder rebuilds unseen nights at least as well as PCA.')}
<h2>Stage 4 - Validating discoveries (synthetic labels)</h2>
{img('s4_calibration', 'With random labels the permutation test flags no more than about 5% of runs; with a planted effect it detects it at a study-sized cohort.')}
<h2>Stage 5 - Classification (synthetic labels)</h2>
{img('s5_classifier', 'With random labels AUROC centres on 0.5; with a planted effect the elastic-net model recovers it. No participant appears in both training and test folds.')}
<h2>Stage 6 - Wearable translation</h2>
{table(['Sensor set', 'Kappa vs technician', 'REM latency bias (min)', '95% limits (min)', 'ICC'], R.get('s6_rows', []))}
{img('s6_montages', 'Staging agreement for each reduced sensor set, and REM-latency agreement for the best one.')}
<h2>What this does and does not show</h2>
<p>Shows: the software reads real PSG, stages sleep, measures features, learns latent features without labels, and its statistical tests behave correctly (about 5% false positives, planted effects found, no train/test leakage). Does not show: any link between these features and anxiety, or performance in people with MCI/SCI. That requires the clinicians' labelled cohort (Validation 4). Sleep-EDF has no ECG, so the HRV module was checked on reference signals; repeat V1b on HMC recordings.</p>
</main></body></html>"""
    REPORT_HTML.write_text(html, encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Sleeping Minds verification pipeline (public Sleep-EDF data, no anxiety labels)")
    ap.add_argument("--n", type=int, default=28, help="number of Sleep-EDF subjects to request (every third subject)")
    ap.add_argument("--subjects", type=str, default="", help="comma-separated Sleep-EDF subject numbers (0-82)")
    ap.add_argument("--synthetic", action="store_true", help="dry run on simulated nights, no download")
    a = ap.parse_args()
    t0 = time.time()
    if a.synthetic:
        nights = synthetic_nights(12)
    else:
        subs = [int(s) for s in a.subjects.split(",")] if a.subjects else list(range(0, 83, 3))[: a.n]
        nights = load_sleep_edf(subs)
    log(f"{len(nights)} nights loaded")
    v1_integration(nights)
    v1b_hrv()
    v2_unit()
    v6_shipped_code()
    v3_staging(nights)
    v4_rem_latency(nights)
    v5_plausibility(nights)
    s2_positive_control(nights)
    P, clusters, k = s3_discovery(nights)
    s4_calibration(P, clusters, k)
    s5_classifier(nights, P)
    s6_montages(nights)
    R["meta"] = dict(synthetic=a.synthetic, nights=[dict(id=n["id"], age=n["age"], sex=n["sex"]) for n in nights],
                     runtime_min=round((time.time() - t0) / 60, 1), yasa=yasa.__version__, mne=mne.__version__,
                     sklearn=SKLEARN_VERSION, seed=SEED)
    RESULTS_JSON.write_text(json.dumps(R, indent=1, default=as_py))
    html_report(nights, a.synthetic)
    log(f"Done in {R['meta']['runtime_min']} min. Open {REPORT_HTML}; figures are in {OUT}/")


if __name__ == "__main__":
    main()
