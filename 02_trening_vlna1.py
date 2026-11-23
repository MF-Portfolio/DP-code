"""Úloha 4: vnorená krížová validácia na vlne 1 (interná validácia) a finálne zafixované modely."""
import json, pickle, sys, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from common import CFG, load, feature_list, X_of, metrics, bootstrap, SEED, OUT_, DATA
from pipelines import make_lr, make_xgb

mode = sys.argv[1] if len(sys.argv) > 1 else 'exclude'
OUT = OUT_ + {'exclude': '', 'nonevent': 'sens_', 'event': 'sens2_', 'death': 'sens3_'}[mode]
N_ITER = CFG['n_iter']
df = load(mode); feats = feature_list(df)
w1 = df[df.vlna == 1]
X1, y1 = X_of(w1, feats), w1.y.values
outer = StratifiedKFold(5, shuffle=True, random_state=SEED)

res, oof, final = {}, {}, {}
for name, maker in [('LR', make_lr), ('XGB', make_xgb)]:
    p_oof = np.zeros(len(y1)); fold_rows = []
    for k, (tr, te) in enumerate(outer.split(X1, y1)):
        pipe, grid = maker()
        inner = StratifiedKFold(5, shuffle=True, random_state=SEED + k)
        rs = RandomizedSearchCV(pipe, grid, n_iter=N_ITER, scoring='neg_log_loss',
                                cv=inner, random_state=SEED + k, n_jobs=4, refit=True)
        rs.fit(X1.iloc[tr], y1[tr])
        p_oof[te] = rs.predict_proba(X1.iloc[te])[:, 1]
        m = metrics(y1[te], p_oof[te]); m['fold'] = k + 1
        m['best'] = {kk.replace('clf__', ''): (round(float(v), 4)) for kk, v in rs.best_params_.items()}
        fold_rows.append(m)
        print(name, k + 1, {kk: round(v, 3) for kk, v in m.items() if isinstance(v, float)}, flush=True)
    oof[name] = p_oof
    folds = pd.DataFrame(fold_rows)
    bs = bootstrap(y1, p_oof, B=1000)
    pooled = metrics(y1, p_oof)
    res[name] = dict(folds=folds.to_dict('records'), pooled=pooled,
                     ci={c: [float(bs[c].quantile(.025)), float(bs[c].quantile(.975))]
                         for c in ['AUROC', 'CITL', 'sklon', 'Brier']})
    # finálny model: ladenie vnútornou CV na celej vlne 1, potom fit na celej vlne 1 a zafixovanie
    pipe, grid = maker()
    rs = RandomizedSearchCV(pipe, grid, n_iter=N_ITER, scoring='neg_log_loss',
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                            random_state=SEED, n_jobs=4, refit=True)
    rs.fit(X1, y1)
    final[name] = rs.best_estimator_
    res[name]['final_params'] = {kk.replace('clf__', ''): round(float(v), 4) for kk, v in rs.best_params_.items()}
    res[name]['n_features_used'] = int(len(rs.best_estimator_.named_steps['drop'].keep_))

pickle.dump(dict(models=final, feats=feats), open(OUT + 'final_models.pkl', 'wb'))
pd.DataFrame({'y': y1, 'oof_LR': oof['LR'], 'oof_XGB': oof['XGB']}).to_csv(OUT + 'oof_wave1.csv', index=False)
json.dump(res, open(OUT + 'wave1_internal.json', 'w'), indent=1, default=float, ensure_ascii=False)
print(json.dumps({k: (v['pooled'], v['ci'], v['final_params'], v['n_features_used']) for k, v in res.items()}, default=float, indent=1, ensure_ascii=False))
