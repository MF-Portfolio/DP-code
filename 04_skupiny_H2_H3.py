"""VO2/H2 a VO3/H3: skupinové CITL pevných modelov na vlnách 2–4, rozdiely medzi skupinami,
rozdiely časových zmien a Holmova korekcia.

Rodiny testov (vopred určené):
  H2 – pre každý model samostatne 9 testov: 3 porovnania × 3 testovacie vlny (2, 3, 4),
       H0: CITL_A − CITL_B = 0 v danej vlne.
  H3 – pre každý model samostatne 3 testy: 3 porovnania,
       H0: (CITL4 − CITL2)_A − (CITL4 − CITL2)_B = 0.
Hlavným modelom je XGBoost; logistická regresia je referencia (vlastné rodiny, rovnaký postup).
Použitie: python 04_skupiny_H2_H3.py [exclude|nonevent|event]
"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from common import CFG, load, SEED, OUT_, slope
from recal_utils import citl_fast, holm, boot_p, COMPARISONS, attach_groups

mode = sys.argv[1] if len(sys.argv) > 1 else 'exclude'
PFX = {'exclude': '', 'nonevent': 'sens_', 'event': 'sens2_', 'death': 'sens3_'}[mode]
B = CFG['bootstrap_B_skupiny']
df = load(mode)
P = attach_groups(pd.read_csv(OUT_ + PFX + 'predictions_waves2-4.csv'), df)
oof = pd.read_csv(OUT_ + PFX + 'oof_wave1.csv')


def youden_threshold(y, p):
    """Spoločný prah určený iba z OOF predikcií vlny 1 (bez testovacích dát)."""
    grid = np.unique(np.round(p, 4))
    best, thr = -1, 0.5
    for t in grid:
        tpr = ((p >= t) & (y == 1)).sum() / max((y == 1).sum(), 1)
        fpr = ((p >= t) & (y == 0)).sum() / max((y == 0).sum(), 1)
        if tpr - fpr > best:
            best, thr = tpr - fpr, t
    return float(thr)


THR = {m: youden_threshold(oof.y.values, oof['oof_' + m].values) for m in ['XGB', 'LR']}

#skupinové metriky
rows = []
for m in ['XGB', 'LR']:
    for w in [2, 3, 4]:
        d = P.query('model==@m and vlna==@w')
        for comp, (col, a, b) in COMPARISONS.items():
            for lvl in (a, b):
                g = d[d[col] == lvl]; y = g.y.values; p = g.p.values
                rng = np.random.default_rng(SEED + w)
                bc = [citl_fast(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(B))]
                yhat = p >= THR[m]
                rows.append(dict(model=m, vlna=w, porovnanie=comp, skupina=lvl, n=len(y), udalosti=int(y.sum()),
                                 pozorovane=y.mean(), predikovane=p.mean(), CITL=citl_fast(y, p),
                                 CITL_lo=np.quantile(bc, .025), CITL_hi=np.quantile(bc, .975),
                                 sklon=slope(y, p), AUROC=roc_auc_score(y, p),
                                 TPR=(yhat & (y == 1)).sum() / (y == 1).sum(),
                                 FPR=(yhat & (y == 0)).sum() / (y == 0).sum(), prah=THR[m]))
G = pd.DataFrame(rows)
G.to_csv(OUT_ + PFX + 'skupiny_metriky.csv', index=False)

#bootstrap replikácie rozdielov v rámci vlny (stratifikovane podľa skupiny)
def diff_reps(d, col, a, b, seed):
    ya, pa = d.loc[d[col] == a, 'y'].values, d.loc[d[col] == a, 'p'].values
    yb, pb = d.loc[d[col] == b, 'y'].values, d.loc[d[col] == b, 'p'].values
    rng = np.random.default_rng(seed); out = np.empty(B)
    for k in range(B):
        i = rng.integers(0, len(ya), len(ya)); j = rng.integers(0, len(yb), len(yb))
        out[k] = citl_fast(ya[i], pa[i]) - citl_fast(yb[j], pb[j])
    return citl_fast(ya, pa) - citl_fast(yb, pb), out


h2, h3 = [], []
for m in ['XGB', 'LR']:
    reps = {}
    for ci, (comp, (col, a, b)) in enumerate(COMPARISONS.items()):
        for w in [2, 3, 4]:
            d = P.query('model==@m and vlna==@w')
            est, r = diff_reps(d, col, a, b, SEED + 100 * w + ci)
            reps[(comp, w)] = (est, r)
            h2.append(dict(model=m, porovnanie=comp, vlna=w, rozdiel=est, lo=np.quantile(r, .025),
                           hi=np.quantile(r, .975), p=boot_p(r)))
        e2, r2 = reps[(comp, 2)]; e4, r4 = reps[(comp, 4)]
        dd = r4 - r2                                    # vlny sú nezávislé vzorky
        absch = np.abs(r4) - np.abs(r2)
        h3.append(dict(model=m, porovnanie=comp, rozdiel_v2=e2, rozdiel_v4=e4,
                       rozdiel_zmien=e4 - e2, lo=np.quantile(dd, .025), hi=np.quantile(dd, .975), p=boot_p(dd),
                       zmena_abs_rozdielu=abs(e4) - abs(e2), abs_lo=np.quantile(absch, .025),
                       abs_hi=np.quantile(absch, .975)))
H2 = pd.DataFrame(h2); H3 = pd.DataFrame(h3)
for T in (H2, H3):
    T['p_holm'] = np.nan
    for m in ['XGB', 'LR']:
        idx = T.model == m
        T.loc[idx, 'p_holm'] = holm(T.loc[idx, 'p'].values)
# zmeny CITL4 − CITL2 v jednotlivých skupinách
ch = []
for m in ['XGB', 'LR']:
    for comp, (col, a, b) in COMPARISONS.items():
        for lvl in (a, b):
            c2 = G.query('model==@m and vlna==2 and porovnanie==@comp and skupina==@lvl').CITL.iloc[0]
            c4 = G.query('model==@m and vlna==4 and porovnanie==@comp and skupina==@lvl').CITL.iloc[0]
            ch.append(dict(model=m, porovnanie=comp, skupina=lvl, CITL2=c2, CITL4=c4, zmena=c4 - c2))
CH = pd.DataFrame(ch)
H2.to_csv(OUT_ + PFX + 'H2.csv', index=False); H3.to_csv(OUT_ + PFX + 'H3.csv', index=False)
CH.to_csv(OUT_ + PFX + 'H3_zmeny_skupin.csv', index=False)
json.dump(THR, open(OUT_ + PFX + 'prah_vlna1.json', 'w'))
pd.set_option('display.width', 250)
print(THR)
print(G.round(3).to_string(index=False))
print(H2.round(3).to_string(index=False))
print(H3.round(3).to_string(index=False))
print(CH.round(3).to_string(index=False))
