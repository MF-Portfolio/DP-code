"""Spoločné definície: načítanie dát, cieľová premenná, prediktory, metriky."""
import numpy as np, pandas as pd
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss

import os
from pathlib import Path
ROOT = Path(os.environ.get('DP_ROOT', Path(__file__).resolve().parent.parent))
DATA = str(ROOT / 'data' / 'df.pkl')
OUTDIR = ROOT / 'vysledky'; OUTDIR.mkdir(exist_ok=True)
OUT_ = str(OUTDIR) + '/'
import json as _json
CFG = _json.load(open(Path(__file__).resolve().parent / 'config.json', encoding='utf-8'))
SEED = CFG['seed']

LAB_EXCL = ()  # filtrovanie podľa chýbajúcich hodnôt sa robí vnútri pipeline
COMORB = ['Hypertenzia', 'Diabetes mellitus', 'Kardiovaskulárne ochorenia',
          'Chronické respiračné ochorenia', 'Pečeňové ochorenia',
          'Onkologické ochorenia', 'Imunosupresia']
SYMPT = ['kasel', 'hnacka', 'triaska_zimnica', 'zvracanie', 'nechutenstvo']


def load(transferred='exclude'):
    """transferred: 'exclude' (hlavná analýza), 'nonevent' alebo 'event' (citlivostné analýzy)."""
    df = pd.read_pickle(DATA).copy()
    st = df['Stav prepustenia']
    df['y'] = st.isin(['exitus letalis', 'ARO/OAIM']).astype(int)
    if transferred == 'event':          # druhá citlivostná analýza: preložený = udalosť
        df.loc[st == 'preložený', 'y'] = 1
    if transferred == 'death':          # citlivostná analýza C: udalosť = iba úmrtie; preložení a ARO/OAIM vylúčení
        df = df[~st.isin(['preložený', 'ARO/OAIM'])].copy()
        df['y'] = (df['Stav prepustenia'] == 'exitus letalis').astype(int)
    if transferred == 'exclude':
        df = df[st != 'preložený'].copy()
    df['muz'] = (df['Pohlavie'] == 'Muž').astype(int)
    df['vek_kat'] = pd.cut(df['Vek'], [-1, 49, 65, 80, 200],
                           labels=['<50', '50–65', '66–80', '>80'])
    return df.reset_index(drop=True)


def feature_list(df):
    labs = [c for c in df.columns if c.endswith('_prva_hodnota')]
    vit = ['sat_bez_o2_prva', 'sat_s_o2_prva', 'temp_first']
    return ['Vek', 'muz'] + COMORB + ['Vakcinácia'] + SYMPT + vit + labs


def X_of(df, feats):
    X = df[feats].copy()
    for c in X.columns:
        if X[c].dtype == bool:
            X[c] = X[c].astype(int)
    return X.astype(float)


# ---------------- metriky ----------------
def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def citl(y, p):
    lp = _logit(p)
    m = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp).fit()
    return float(m.params[0])


def slope(y, p):
    lp = _logit(p)
    m = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    return float(m.params[1])


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    return dict(AUROC=roc_auc_score(y, p), CITL=citl(y, p), sklon=slope(y, p),
                Brier=brier_score_loss(y, p), n=len(y), udalosti=int(y.sum()),
                pozorovane=y.mean(), predikovane=p.mean())


def bootstrap(y, p, B=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    y = np.asarray(y); p = np.asarray(p); n = len(y)
    out = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].min() == y[i].max():
            continue
        out.append(metrics(y[i], p[i]))
    return pd.DataFrame(out)
