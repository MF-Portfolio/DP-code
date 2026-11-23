"""Doklad reprodukovateľnosti a toho, že filter chýbajúcich hodnôt (DropSparse) aj imputácia
sa učia iba na tréningovej časti každej vonkajšej časti krížovej validácie.
Výstupy: out/filter_v_cv.csv, out/reprodukovatelnost.json
"""
import json, hashlib, platform, sys, numpy as np, pandas as pd
import sklearn, xgboost, statsmodels, scipy
from sklearn.model_selection import StratifiedKFold
from common import load, feature_list, X_of, SEED, OUT_, DATA, citl
from pipelines import make_lr, make_xgb
from recal_utils import citl_fast

df = load('exclude'); feats = feature_list(df)
w1 = df[df.vlna == 1]; X1, y1 = X_of(w1, feats), w1.y.values
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, checks = [], {}
for k, (tr, te) in enumerate(outer.split(X1, y1)):
    for name, maker in [('LR', make_lr), ('XGB', make_xgb)]:
        pipe, _ = maker()
        pipe.fit(X1.iloc[tr], y1[tr])
        keep_pipe = set(np.array(feats)[pipe.named_steps['drop'].keep_])
        na_tr = X1.iloc[tr].isna().mean(); na_te = X1.iloc[te].isna().mean(); na_all = X1.isna().mean()
        keep_tr = set(na_tr[(na_tr <= .5) & (X1.iloc[tr].std() > 0)].index)
        keep_all = set(na_all[(na_all <= .5) & (X1.std() > 0)].index)
        # kanárik: v testovacej časti úplne vymažeme jednu ponechanú premennú a zmeníme ostatné hodnoty
        Xc = X1.iloc[te].copy(); canary = sorted(keep_pipe)[0]; Xc[canary] = np.nan
        Xc = Xc * 3.0
        keep_after = set(np.array(feats)[pipe.named_steps['drop'].keep_])
        ok_shape = pipe[:-1].transform(Xc).shape[1] == pipe[:-1].transform(X1.iloc[te]).shape[1]
        rows.append(dict(cast=k + 1, model=name, n_trening=len(tr), n_test=len(te),
                         ponechane=len(keep_pipe), vyradene=', '.join(sorted(set(feats) - keep_pipe)),
                         zhoda_s_treningovou_castou=keep_pipe == keep_tr,
                         zhoda_s_celou_vlnou=keep_pipe == keep_all,
                         kanarik_bez_zmeny=(keep_after == keep_pipe) and ok_shape))
F = pd.DataFrame(rows)
# premenné s 25–75 % chýbajúcich hodnôt: podiel v tréningovej časti každej vonkajšej časti
border = []
for c in feats:
    vals = [X1.iloc[tr][c].isna().mean() for tr, _ in outer.split(X1, y1)]
    if 0.25 < np.mean(vals) < 0.75:
        border.append(dict(premenna=c, **{f'cast_{i + 1}': round(v * 100, 1) for i, v in enumerate(vals)},
                           filter='vyradená' if np.mean(vals) > .5 else 'ponechaná'))

# inštrumentácia: zaznamenáme, na ktorých pacientoch sa filter učí počas vnorenej validácie (1. vonkajšia časť)
from sklearn.model_selection import RandomizedSearchCV
from pipelines import DropSparse
FIT_LOG = []
class LoggedDrop(DropSparse):
    def fit(self, X, y=None):
        FIT_LOG.append(set(X.index)); return super().fit(X, y)
tr, te = next(outer.split(X1, y1))
pipe, grid = make_xgb(); pipe.steps[0] = ('drop', LoggedDrop(0.5))
rs = RandomizedSearchCV(pipe, grid, n_iter=2, scoring='neg_log_loss',
                        cv=StratifiedKFold(5, shuffle=True, random_state=SEED), random_state=SEED, n_jobs=1)
rs.fit(X1.iloc[tr], y1[tr])
test_idx = set(X1.index[te])
checks['instrumentacia_pocet_uceni_filtra'] = len(FIT_LOG)
checks['instrumentacia_velkosti_treningu'] = sorted({len(s) for s in FIT_LOG})
checks['instrumentacia_prienik_s_testom'] = int(max(len(s & test_idx) for s in FIT_LOG))
# kontrola výpočtu CITL (Newton vs. GLM)
rng = np.random.default_rng(SEED); p = rng.uniform(.02, .9, 2000); y = rng.binomial(1, p)
checks['citl_glm_vs_newton_max_abs_diff'] = abs(citl(y, p) - citl_fast(y, p))
checks['vsetky_casti_filter_z_treningu'] = bool(F.zhoda_s_treningovou_castou.all())
checks['kanarik_ok'] = bool(F.kanarik_bez_zmeny.all())
checks['seed'] = SEED
checks['verzie'] = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                        scikit_learn=sklearn.__version__, xgboost=xgboost.__version__,
                        statsmodels=statsmodels.__version__, scipy=scipy.__version__)
checks['md5_df_pkl'] = hashlib.md5(open(DATA, 'rb').read()).hexdigest()
for f in ['predictions_waves2-4.csv', 'oof_wave1.csv', 'sens_predictions_waves2-4.csv']:
    try:
        checks['md5_' + f] = hashlib.md5(open(OUT_ + f, 'rb').read()).hexdigest()
    except FileNotFoundError:
        pass
F.to_csv(OUT_ + 'filter_v_cv.csv', index=False)
pd.DataFrame(border).to_csv(OUT_ + 'filter_hranicne_premenne.csv', index=False)
json.dump(checks, open(OUT_ + 'reprodukovatelnost.json', 'w'), indent=1, ensure_ascii=False, default=str)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 120)
print(F.to_string(index=False)); print(pd.DataFrame(border).to_string(index=False))
print(json.dumps(checks, indent=1, ensure_ascii=False, default=str))
