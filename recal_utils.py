"""Pomocné funkcie pre skupinové analýzy (H2, H3) a rekalibráciu (H4)."""
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

# vopred určené porovnania: názov -> (stĺpec, skupina A, skupina B); rozdiel = A − B
COMPARISONS = {
    'muži − ženy': ('pohl', 'muži', 'ženy'),
    'vek > 80 − vek 66–80': ('vek_kat', '>80', '66–80'),
    'DM áno − DM nie': ('dm', 'áno', 'nie'),
}


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def citl_fast(y, p, it=50):
    """CITL = intercept α v modeli logit P(Y=1) = α + logit(p̂) (ofset); Newtonova metóda.
    Výsledok je zhodný s GLM odhadom v common.citl (overené v 10_reprodukovatelnost.py)."""
    y = np.asarray(y, float); lp = logit(p); a = 0.0
    for _ in range(it):
        mu = expit(a + lp); g = (y - mu).sum(); h = (mu * (1 - mu)).sum()
        step = g / max(h, 1e-12); a += step
        if abs(step) < 1e-10:
            break
    return float(a)


def holm(p):
    """Holmova–Bonferroniho korekcia (upravené p-hodnoty)."""
    p = np.asarray(p, float); m = len(p); o = np.argsort(p)
    adj = np.empty(m); run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj


def boot_p(r):
    """Dvojstranná bootstrapová p-hodnota pre H0: parameter = 0."""
    r = np.asarray(r); B = len(r)
    lo = ((r <= 0).sum() + 1) / (B + 1); hi = ((r >= 0).sum() + 1) / (B + 1)
    return float(min(1.0, 2 * min(lo, hi)))


def attach_groups(P, df):
    """Doplní skupinové atribúty k uloženým predikciám (poradie riadkov = poradie pacientov vo vlne)."""
    parts = []
    for (m, w), q in P.groupby(['model', 'vlna'], sort=False):
        d = df[df.vlna == w]
        assert len(d) == len(q) and (d.y.values == q.y.values).all() and (d.Vek.values == q.vek.values).all()
        q = q.copy()
        q['pohl'] = np.where(d.muz.values == 1, 'muži', 'ženy')
        q['vek_kat'] = d.vek_kat.astype(str).values
        q['dm'] = np.where(d['Diabetes mellitus'].astype(bool).values, 'áno', 'nie')
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


# ---------------- rekalibrácia ----------------
def fit_recal(y, p, method):
    """Vráti (a, b): rekalibrované logit = a + b · logit(p̂)."""
    if method == 'none':
        return 0.0, 1.0
    if method == 'intercept':
        return citl_fast(y, p), 1.0
    if method == 'intercept_slope':
        lr = LogisticRegression(C=np.inf, solver='lbfgs', max_iter=1000)
        lr.fit(logit(p).reshape(-1, 1), y)
        return float(lr.intercept_[0]), float(lr.coef_[0, 0])
    raise ValueError(method)


def apply_recal(p, ab):
    a, b = ab
    return expit(a + b * logit(p))
