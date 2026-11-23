"""Párové porovnanie XGBoost − logistická regresia na tých istých pacientoch (vlny 2–4): rozdiel Brierovho skóre,
AUROC a |CITL| s 95 % IS z párového bootstrapu. Výstup: out/porovnanie_modelov.csv
Použitie: python 06_porovnanie_modelov.py [exclude|nonevent|event|death]
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from common import CFG, SEED, OUT_
from recal_utils import citl_fast, boot_p

mode = sys.argv[1] if len(sys.argv) > 1 else 'exclude'
PFX = {'exclude': '', 'nonevent': 'sens_', 'event': 'sens2_', 'death': 'sens3_'}[mode]
B = CFG['bootstrap_B']
P = pd.read_csv(OUT_ + PFX + 'predictions_waves2-4.csv')
M = {'Brier': brier_score_loss, 'AUROC': roc_auc_score, 'absCITL': lambda y, p: abs(citl_fast(y, p))}
rows = []
for w in [2, 3, 4]:
    x = P.query('model=="XGB" and vlna==@w'); l = P.query('model=="LR" and vlna==@w')
    assert (x.y.values == l.y.values).all()
    y, px, pl = x.y.values, x.p.values, l.p.values
    rng = np.random.default_rng(SEED + 500 + w); reps = {k: [] for k in M}
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        for k, f in M.items():
            reps[k].append(f(y[i], px[i]) - f(y[i], pl[i]))
    for k, f in M.items():
        r = np.array(reps[k])
        rows.append(dict(vlna=w, metrika=k, XGB=f(y, px), LR=f(y, pl), rozdiel=f(y, px) - f(y, pl),
                         lo=np.quantile(r, .025), hi=np.quantile(r, .975), p=boot_p(r)))
R = pd.DataFrame(rows); R.to_csv(OUT_ + PFX + 'porovnanie_modelov.csv', index=False)
print(R.round(4).to_string(index=False))
