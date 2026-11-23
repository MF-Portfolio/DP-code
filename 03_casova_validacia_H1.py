"""Úloha 5: časová validácia zafixovaných modelov na vlnách 2–4 a test H1."""
import json, pickle, sys, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import CFG, load, X_of, metrics, bootstrap, SEED, OUT_, DATA

mode = sys.argv[1] if len(sys.argv) > 1 else 'exclude'
OUT = OUT_ + {'exclude': '', 'nonevent': 'sens_', 'event': 'sens2_', 'death': 'sens3_'}[mode]
B = CFG['bootstrap_B']
df = load(mode); pk = pickle.load(open(OUT + 'final_models.pkl', 'rb'))
models, feats = pk['models'], pk['feats']
rows, boots, preds = [], {}, []
for name, mdl in models.items():
    for w in [2, 3, 4]:
        d = df[df.vlna == w]
        p = mdl.predict_proba(X_of(d, feats))[:, 1]
        preds.append(pd.DataFrame({'model': name, 'vlna': w, 'y': d.y.values, 'p': p,
                                   'muz': d.muz.values, 'vek': d.Vek.values}))
        m = metrics(d.y.values, p)
        bs = bootstrap(d.y.values, p, B=B, seed=SEED + w)
        boots[(name, w)] = bs
        for c in ['AUROC', 'CITL', 'sklon', 'Brier']:
            m[c + '_lo'] = bs[c].quantile(.025); m[c + '_hi'] = bs[c].quantile(.975)
        m.update(model=name, vlna=w); rows.append(m)
tab = pd.DataFrame(rows)
cols = ['model', 'vlna', 'n', 'udalosti', 'pozorovane', 'predikovane'] + \
       [f'{c}{s}' for c in ['AUROC', 'CITL', 'sklon', 'Brier'] for s in ['', '_lo', '_hi']]
tab = tab[cols]; tab.to_csv(OUT + 'temporal_metrics.csv', index=False)
pd.concat(preds).to_csv(OUT + 'predictions_waves2-4.csv', index=False)
pd.set_option('display.width', 250)
print(tab.round(3).to_string(index=False))

# H1: |CITL4| - |CITL2| ; vlny sú nezávislé vzorky -> nezávislé bootstrap replikácie
h1 = {}
for name in models:
    d2 = df[df.vlna == 2]; d4 = df[df.vlna == 4]
    from common import citl
    c2 = tab.query('model==@name and vlna==2').CITL.iloc[0]
    c4 = tab.query('model==@name and vlna==4').CITL.iloc[0]
    diff = abs(c4) - abs(c2)
    b2 = boots[(name, 2)].CITL.values; b4 = boots[(name, 4)].CITL.values
    k = min(len(b2), len(b4)); bd = np.abs(b4[:k]) - np.abs(b2[:k])
    h1[name] = dict(CITL2=c2, CITL4=c4, diff=diff, lo=float(np.quantile(bd, .025)),
                    hi=float(np.quantile(bd, .975)), p_boot_le0=float((bd <= 0).mean()))
print(json.dumps(h1, indent=1, default=float))
json.dump(h1, open(OUT + 'H1.json', 'w'), indent=1, default=float)

# kalibračné grafy (decily + LOESS-like vyhladenie cez binning)
if mode == 'exclude':
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
    P = pd.concat(preds)
    for i, name in enumerate(['XGB', 'LR']):
        for j, w in enumerate([2, 3, 4]):
            ax = axes[i, j]; d = P.query('model==@name and vlna==@w')
            q = pd.qcut(d.p, 10, duplicates='drop')
            g = d.groupby(q, observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
            se = np.sqrt(g.y * (1 - g.y) / g.n)
            ax.plot([0, 1], [0, 1], ls='--', c='grey', lw=1)
            ax.errorbar(g.p, g.y, yerr=1.96 * se, fmt='o-', ms=4, capsize=2,
                        c='#1f5fa8' if name == 'XGB' else '#b5532a')
            r = tab.query('model==@name and vlna==@w').iloc[0]
            ax.set_title(f"{'XGBoost' if name == 'XGB' else 'Logistická regresia'} – vlna {w}\n"
                         f"CITL={r.CITL:.2f}, sklon={r.sklon:.2f}, AUROC={r.AUROC:.3f}", fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            if i == 1: ax.set_xlabel('Predikované riziko')
            if j == 0: ax.set_ylabel('Pozorovaný podiel udalostí')
    plt.tight_layout(); plt.savefig(OUT_ + 'kalibracne_grafy.png', dpi=200)
