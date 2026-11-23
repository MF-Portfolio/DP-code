"""Doplnkové tabuľky: kalibrácia podľa pätín predikovaného rizika (vlny 2–4) a opisné CITL malých skupín."""
import numpy as np, pandas as pd
from common import load, OUT_
from recal_utils import citl_fast
df = load('exclude'); P = pd.read_csv(OUT_ + 'predictions_waves2-4.csv')
rows = []
for w in [2, 3, 4]:
    d = P.query('model=="XGB" and vlna==@w').copy(); d['q'] = pd.qcut(d.p, 5, labels=False)
    for q, g in d.groupby('q'):
        rows.append(dict(vlna=w, patina=q + 1, n=len(g), udalosti=int(g.y.sum()), predikovane=g.p.mean(), pozorovane=g.y.mean()))
pd.DataFrame(rows).to_csv(OUT_ + 'patiny_vlny.csv', index=False)
groups = [('vek < 50', lambda d: d.Vek < 50), ('vek 50 – 65', lambda d: (d.Vek >= 50) & (d.Vek <= 65)),
          ('onkologické ochorenia', lambda d: d['Onkologické ochorenia'].astype(bool)),
          ('imunosupresia', lambda d: d['Imunosupresia'].astype(bool)),
          ('pečeňové ochorenia', lambda d: d['Pečeňové ochorenia'].astype(bool)),
          ('bez pečeňových ochorení', lambda d: ~d['Pečeňové ochorenia'].astype(bool))]
rows = []
for nm, f in groups:
    r = dict(skupina=nm)
    for w in [2, 3, 4]:
        dd = df[df.vlna == w]; pp = P.query('model=="XGB" and vlna==@w').p.values
        m = f(dd).values; y = dd.y.values[m]; p = pp[m]
        r[f'v{w}_n'] = int(m.sum()); r[f'v{w}_ud'] = int(y.sum()); r[f'v{w}_pred'] = p.mean(); r[f'v{w}_poz'] = y.mean()
        r[f'v{w}_citl'] = citl_fast(y, p) if 0 < y.sum() < len(y) else np.nan
    rows.append(r)
T = pd.DataFrame(rows); T.to_csv(OUT_ + 'opisne_skupiny.csv', index=False)
pd.set_option('display.width', 250); print(pd.read_csv(OUT_ + 'patiny_vlny.csv').round(3).to_string()); print(T.round(2).to_string())
