"""Opis preložených pacientov a ich predikované riziko podľa zafixovaného hlavného modelu (podklad pre citlivostnú analýzu)
vrátane obrázka rozdelenia predikovaného rizika podľa stavu pri ukončení hospitalizácie."""
import pickle, numpy as np, pandas as pd
from common import load, X_of, OUT_
pk = pickle.load(open(OUT_ + 'final_models.pkl', 'rb'))
df = load('nonevent')   # všetci pacienti vrátane preložených
st = df['Stav prepustenia']
df['skupina_stav'] = np.select([st == 'preložený', df.y == 1], ['preložený', 'ťažký priebeh'], 'prepustený')
for m in ['XGB', 'LR']:
    df['p_' + m] = pk['models'][m].predict_proba(X_of(df, pk['feats']))[:, 1]
rows = []
for w in [1, 2, 3, 4]:
    for s in ['ťažký priebeh', 'prepustený', 'preložený']:
        d = df[(df.vlna == w) & (df.skupina_stav == s)]
        rows.append(dict(vlna=w, stav=s, n=len(d), podiel_vo_vlne=len(d) / (df.vlna == w).sum(),
                         vek_median=d.Vek.median(), muzi=d.muz.mean(), crp_median=d['S-CRP_prva_hodnota'].median(),
                         p_XGB_priemer=d.p_XGB.mean(), p_LR_priemer=d.p_LR.mean()))
T = pd.DataFrame(rows); T.to_csv(OUT_ + 'prelozeni_opis.csv', index=False)
pd.set_option('display.width', 200); print(T.round(3).to_string(index=False))

#obrázok: rozdelenie predikovaného rizika podľa stavu pri ukončení hospitalizácie (vlny 2 – 4)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                     'legend.frameon': False})
CF = FuncFormatter(lambda v, _: (f'{v:g}').replace('.', ',').replace('-', '−'))
t = df[df.vlna.isin([2, 3, 4])]; bins = np.linspace(0, 1, 26)
fig, ax = plt.subplots(figsize=(10, 3.2))
for s, lab, col in [('prepustený', 'prepustení', '#9aa5b1'), ('preložený', 'preložení na iné pracovisko', '#e0a050'),
                    ('ťažký priebeh', 's udalosťou', '#b5532a')]:
    ax.hist(t.loc[t.skupina_stav == s, 'p_XGB'], bins=bins, density=True, histtype='step', lw=2, color=col, label=lab)
ax.set_xlabel('Predikované riziko hlavného modelu (vlny 2 – 4)'); ax.set_ylabel('Hustota')
ax.xaxis.set_major_formatter(CF); ax.legend(loc='upper right')
plt.tight_layout(); plt.savefig(OUT_ + 'prelozeni_riziko.png', dpi=200); plt.close()
