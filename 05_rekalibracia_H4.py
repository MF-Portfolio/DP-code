"""VO4/H4 – prvá časť: globálne rekalibračné varianty pevného modelu.
Prechody: rekalibrácia naučená na vlne 2 -> test na vlne 3; naučená na vlne 3 -> test na vlne 4.
Varianty: bez rekalibrácie (none), úprava interceptu (intercept), úprava interceptu a sklonu (intercept_slope).
Neistota: v každej replikácii sa nezávisle prevzorkuje aktualizačná aj testovacia vlna, rekalibrácia sa
naučí znova a všetky varianty sa hodnotia na tých istých testovacích pacientoch (párové porovnanie).
Použitie: python 05_rekalibracia_H4.py [exclude|nonevent|event]
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from common import CFG, load, SEED, OUT_, slope
from recal_utils import (citl_fast, fit_recal, apply_recal, attach_groups, COMPARISONS, boot_p)

mode = sys.argv[1] if len(sys.argv) > 1 else 'exclude'
PFX = {'exclude': '', 'nonevent': 'sens_', 'event': 'sens2_', 'death': 'sens3_'}[mode]
B = CFG['bootstrap_B_rekalibracia']
METHODS = ['none', 'intercept', 'intercept_slope']
df = load(mode)
P = attach_groups(pd.read_csv(OUT_ + PFX + 'predictions_waves2-4.csv'), df)


def evaluate(y, p, grp):
    out = dict(CITL=citl_fast(y, p), absCITL=abs(citl_fast(y, p)), Brier=brier_score_loss(y, p),
               AUROC=roc_auc_score(y, p), sklon=slope(y, p), predikovane=p.mean(), pozorovane=y.mean())
    gc = []
    for comp, (col, a, b) in COMPARISONS.items():
        ca = citl_fast(y[grp[col] == a], p[grp[col] == a]); cb = citl_fast(y[grp[col] == b], p[grp[col] == b])
        out[f'CITL[{comp}|{a}]'] = ca; out[f'CITL[{comp}|{b}]'] = cb
        out[f'gap[{comp}]'] = abs(ca - cb); gc += [ca, cb]
    out['rozpatie'] = max(gc) - min(gc)
    out['max_abs_skup'] = max(abs(c) for c in gc)
    return out


rows, params, diffs = [], [], []
for m in ['XGB', 'LR']:
    for wu, wt in [(2, 3), (3, 4)]:
        U = P.query('model==@m and vlna==@wu'); T = P.query('model==@m and vlna==@wt')
        yu, pu = U.y.values, U.p.values; yt, pt = T.y.values, T.p.values
        gt = {c: T[c].values for c in ['pohl', 'vek_kat', 'dm']}
        point = {}
        for meth in METHODS:
            ab = fit_recal(yu, pu, meth)
            params.append(dict(model=m, prechod=f'{wu}→{wt}', metoda=meth, a=ab[0], b=ab[1]))
            point[meth] = evaluate(yt, apply_recal(pt, ab), gt)
        rng = np.random.default_rng(SEED + 10 * wu + wt)
        reps = {meth: [] for meth in METHODS}
        for _ in range(B):
            iu = rng.integers(0, len(yu), len(yu)); it = rng.integers(0, len(yt), len(yt))
            gb = {c: v[it] for c, v in gt.items()}
            for meth in METHODS:
                ab = fit_recal(yu[iu], pu[iu], meth)
                reps[meth].append(evaluate(yt[it], apply_recal(pt[it], ab), gb))
        reps = {k: pd.DataFrame(v) for k, v in reps.items()}
        for meth in METHODS:
            r = dict(model=m, prechod=f'{wu}→{wt}', metoda=meth, **point[meth])
            for c in ['CITL', 'sklon', 'Brier', 'AUROC', 'rozpatie']:
                r[c + '_lo'] = reps[meth][c].quantile(.025); r[c + '_hi'] = reps[meth][c].quantile(.975)
            rows.append(r)
        for meth, ref in [('intercept', 'none'), ('intercept_slope', 'none'), ('intercept_slope', 'intercept')]:
            for c in ['absCITL', 'Brier', 'AUROC', 'rozpatie'] + [f'gap[{k}]' for k in COMPARISONS]:
                dr = reps[meth][c].values - reps[ref][c].values
                diffs.append(dict(model=m, prechod=f'{wu}→{wt}', porovnanie=f'{meth} − {ref}', metrika=c,
                                  rozdiel=point[meth][c] - point[ref][c], lo=np.quantile(dr, .025),
                                  hi=np.quantile(dr, .975), p=boot_p(dr) if np.std(dr) > 0 else np.nan))
R = pd.DataFrame(rows); D = pd.DataFrame(diffs); PR = pd.DataFrame(params)
R.to_csv(OUT_ + PFX + 'H4_globalne.csv', index=False); D.to_csv(OUT_ + PFX + 'H4_rozdiely.csv', index=False)
PR.to_csv(OUT_ + PFX + 'H4_parametre.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 60)
print(PR.round(3).to_string(index=False))
print(R[['model', 'prechod', 'metoda', 'CITL', 'CITL_lo', 'CITL_hi', 'sklon', 'Brier', 'AUROC', 'rozpatie',
         'rozpatie_lo', 'rozpatie_hi', 'predikovane', 'pozorovane'] + [c for c in R.columns if c.startswith('CITL[')]]
      .round(3).to_string(index=False))
print(D.round(3).to_string(index=False))
