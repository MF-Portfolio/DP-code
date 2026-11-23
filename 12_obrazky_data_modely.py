"""Doplnkové opisné výstupy pre rozšírenú praktickú časť: charakteristiky kohorty podľa vĺn, rozdelenia vybraných
vstupov, dôležitosť premenných finálnych modelov, kalibrácia a ROC na vlne 1, rozdelenie predikcií, skupinové
kalibračné grafy a rekalibračné krivky."""
import json, pickle, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve, roc_auc_score
from common import load, feature_list, X_of, OUT_
from recal_utils import attach_groups, fit_recal, apply_recal

O = OUT_
BLUE, ORANGE, GREEN, GREY, INK, MUTED, GRID = '#1f5fa8', '#b5532a', '#3b8b5a', '#888888', '#222222', '#666666', '#e5e5e5'
WC = {1: '#9bbbe0', 2: '#1f5fa8', 3: '#b5532a', 4: '#3b8b5a'}
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
from matplotlib.ticker import FuncFormatter
CF = FuncFormatter(lambda v, _: (f'{v:g}').replace('.', ',').replace('-', '−'))
df = load('exclude'); feats = feature_list(df); pk = pickle.load(open(O + 'final_models.pkl', 'rb'))

#  1) charakteristiky kohorty podľa vĺn
def med(s): s = s.dropna(); return f"{s.median():.1f} ({s.quantile(.25):.1f} – {s.quantile(.75):.1f})".replace('.', ',')
def pct(s): return f"{100 * s.astype(float).mean():.1f}".replace('.', ',')
rows = []
spec = [('Počet hospitalizácií', None, 'n'), ('Vek (roky), medián (IQR)', 'Vek', 'med'), ('Muži (%)', 'muz', 'pct'),
        ('Hypertenzia (%)', 'Hypertenzia', 'pct'), ('Diabetes mellitus (%)', 'Diabetes mellitus', 'pct'),
        ('Kardiovaskulárne ochorenia (%)', 'Kardiovaskulárne ochorenia', 'pct'),
        ('Chronické respiračné ochorenia (%)', 'Chronické respiračné ochorenia', 'pct'),
        ('Pečeňové ochorenia (%)', 'Pečeňové ochorenia', 'pct'), ('Onkologické ochorenia (%)', 'Onkologické ochorenia', 'pct'),
        ('Imunosupresia (%)', 'Imunosupresia', 'pct'), ('Očkovaní (%)', 'Vakcinácia', 'pct'), ('Kašeľ (%)', 'kasel', 'pct'),
        ('Saturácia bez O₂ (%), medián (IQR)', 'sat_bez_o2_prva', 'med'), ('Telesná teplota (°C), medián (IQR)', 'temp_first', 'med'),
        ('CRP (mg/l), medián (IQR)', 'S-CRP_prva_hodnota', 'med'), ('Urea (mmol/l), medián (IQR)', 'S-Urea_prva_hodnota', 'med'),
        ('Lymfocyty (10⁹/l), medián (IQR)', 'Ly abs_prva_hodnota', 'med'), ('D-dimér, medián (IQR)', 'D-dimér HS_prva_hodnota', 'med'),
        ('Úmrtia (%)', None, 'ex'), ('Preklady na ARO/OAIM (%)', None, 'aro'), ('Udalosť spolu (%)', 'y', 'pct')]
for lab, c, t in spec:
    r = {'charakteristika': lab}
    for w in [1, 2, 3, 4]:
        d = df[df.vlna == w]
        r[f'v{w}'] = (str(len(d)) if t == 'n' else med(d[c]) if t == 'med' else pct(d[c]) if t == 'pct' else
                      pct(d['Stav prepustenia'] == 'exitus letalis') if t == 'ex' else pct(d['Stav prepustenia'] == 'ARO/OAIM'))
    rows.append(r)
pd.DataFrame(rows).to_csv(O + 'charakteristiky_vlny.csv', index=False)

# 2) rozdelenia vybraných vstupov podľa vĺn
vars_ = [('Vek', 'vek (roky)', False), ('S-CRP_prva_hodnota', 'CRP (mg/l, log. os)', True), ('S-Urea_prva_hodnota', 'urea (mmol/l, log. os)', True),
         ('Ly abs_prva_hodnota', 'lymfocyty (10⁹/l, log. os)', True), ('sat_bez_o2_prva', 'saturácia bez O₂ (%)', False),
         ('D-dimér HS_prva_hodnota', 'D-dimér (log. os)', True)]
fig, axes = plt.subplots(2, 3, figsize=(10, 5.6))
for ax, (c, lab, lg) in zip(axes.ravel(), vars_):
    data = [df.loc[df.vlna == w, c].dropna() for w in [1, 2, 3, 4]]
    if lg:
        data = [x[x > 0] for x in data]
    bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False, medianprops=dict(color=INK))
    for b, w in zip(bp['boxes'], [1, 2, 3, 4]):
        b.set_facecolor(WC[w]); b.set_alpha(0.85)
    if lg:
        ax.set_yscale('log')
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(['vlna 1', 'vlna 2', 'vlna 3', 'vlna 4'], fontsize=8.5)
    ax.set_title(lab, fontsize=9.5); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    if not lg:
        ax.yaxis.set_major_formatter(CF)
plt.tight_layout(); plt.savefig(O + 'vstupy_vlny.png', dpi=200); plt.close()

#3) dôležitosť premenných
NM = pd.read_csv(O + 'atributy_pouzite.csv').set_index('premenna_v_subore').opis.to_dict()
def nice(c):
    k = 'Pohlavie' if c == 'muz' else c
    s = NM.get(k, c).replace('prvá zaznamenaná hodnota: ', '').replace('prvá zaznamenaná ', '')
    return s[:38]
xgb = pk['models']['XGB']; keep = np.array(feats)[xgb.named_steps['drop'].keep_]
gain = xgb.named_steps['clf'].get_booster().get_score(importance_type='gain')
imp = pd.Series({keep[int(k[1:])]: v for k, v in gain.items()}).sort_values(ascending=False)
imp = imp / imp.sum() * 100
lr = pk['models']['LR']; keep_l = np.array(feats)[lr.named_steps['drop'].keep_]
imp_ = lr.named_steps['imp']; names = list(keep_l) + [f'chýba: {keep_l[i]}' for i in imp_.indicator_.features_]
coef = pd.Series(lr.named_steps['clf'].coef_[0], index=names)
coef = coef[coef.abs() > 1e-8].sort_values(key=np.abs, ascending=False)
pd.DataFrame({'XGB_gain_pct': imp}).to_csv(O + 'dolezitost_xgb.csv'); coef.to_csv(O + 'koeficienty_lr.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
t = imp.head(15)[::-1]
axes[0].barh([nice(c) for c in t.index], t.values, color=BLUE)
axes[0].set_title('XGBoost: podiel na celkovom zisku (gain, %)', fontsize=9.5); axes[0].xaxis.set_major_formatter(CF)
t = coef.head(15)[::-1]
axes[1].barh([nice(c.replace('chýba: ', '')) + (' (chýba)' if c.startswith('chýba') else '') for c in t.index], t.values,
             color=[ORANGE if v > 0 else GREY for v in t.values])
axes[1].axvline(0, color='#555', lw=0.8)
axes[1].set_title('Logistická regresia: štandardizované koeficienty', fontsize=9.5); axes[1].xaxis.set_major_formatter(CF)
for ax in axes:
    ax.tick_params(axis='y', labelsize=8); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(O + 'dolezitost_premennych.png', dpi=200); plt.close()

#4) vlna 1: kalibrácia a ROC OOF predikcií
oof = pd.read_csv(O + 'oof_wave1.csv')
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for m, c, lab in [('XGB', BLUE, 'XGBoost'), ('LR', ORANGE, 'logistická regresia')]:
    p = oof['oof_' + m]; y = oof.y
    q = pd.qcut(p, 10, duplicates='drop'); g = pd.DataFrame({'p': p, 'y': y}).groupby(q, observed=True).mean()
    axes[0].plot(g.p, g.y, 'o-', color=c, ms=4, label=lab)
    fpr, tpr, _ = roc_curve(y, p); axes[1].plot(fpr, tpr, color=c, lw=1.8, label=f'{lab} (AUROC {roc_auc_score(y, p):.3f})'.replace('.', ','))
axes[0].plot([0, 1], [0, 1], ls='--', color=GREY, lw=1); axes[1].plot([0, 1], [0, 1], ls='--', color=GREY, lw=1)
axes[0].set_xlabel('Predikované riziko'); axes[0].set_ylabel('Pozorovaný podiel udalostí'); axes[0].set_title('Kalibračný graf (desatiny)', fontsize=10)
axes[1].set_xlabel('1 − špecificita'); axes[1].set_ylabel('Citlivosť'); axes[1].set_title('ROC krivka', fontsize=10)
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=8.5, loc='lower right' if ax is axes[1] else 'upper left')
    ax.xaxis.set_major_formatter(CF); ax.yaxis.set_major_formatter(CF); ax.grid(True, color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(O + 'vlna1_kalibracia_roc.png', dpi=200); plt.close()

#5) rozdelenie predikcií hlavného modelu podľa vĺn a výsledku
P = attach_groups(pd.read_csv(O + 'predictions_waves2-4.csv'), df)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
bins = np.linspace(0, 1, 26)
for ax, w in zip(axes, [2, 3, 4]):
    d = P.query('model=="XGB" and vlna==@w')
    ax.hist(d.p[d.y == 0], bins=bins, color=GREY, alpha=0.7, density=True, label='bez udalosti')
    ax.hist(d.p[d.y == 1], bins=bins, color=ORANGE, alpha=0.7, density=True, label='s udalosťou')
    ax.set_title(f'vlna {w}', fontsize=10); ax.set_xlabel('Predikované riziko (XGBoost)'); ax.xaxis.set_major_formatter(CF)
axes[0].set_ylabel('Hustota'); axes[0].legend(frameon=False, fontsize=8.5)
plt.tight_layout(); plt.savefig(O + 'rozdelenie_predikcii.png', dpi=200); plt.close()

#6) skupinové kalibračné grafy (kvintily)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
cfg = [(3, 'pohl', [('muži', BLUE), ('ženy', ORANGE)], 'vlna 3: pohlavie'),
       (4, 'vek_kat', [('>80', BLUE), ('66–80', ORANGE)], 'vlna 4: vek'),
       (4, 'dm', [('áno', BLUE), ('nie', ORANGE)], 'vlna 4: diabetes')]
LBL = {'muži': 'muži', 'ženy': 'ženy', '>80': 'vek > 80', '66–80': 'vek 66 – 80', 'áno': 'diabetes áno', 'nie': 'diabetes nie'}
for ax, (w, col, grp, tit) in zip(axes, cfg):
    d = P.query('model=="XGB" and vlna==@w')
    for lvl, c in grp:
        g = d[d[col] == lvl]; q = pd.qcut(g.p, 5, duplicates='drop')
        a = g.groupby(q, observed=True).agg(p=('p', 'mean'), y=('y', 'mean'))
        ax.plot(a.p, a.y, 'o-', color=c, ms=4, label=LBL[lvl])
    ax.plot([0, 1], [0, 1], ls='--', color=GREY, lw=1); ax.set_title(tit, fontsize=10); ax.legend(frameon=False, fontsize=8.5)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel('Predikované riziko'); ax.xaxis.set_major_formatter(CF); ax.yaxis.set_major_formatter(CF)
    ax.grid(True, color=GRID); ax.set_axisbelow(True)
axes[0].set_ylabel('Pozorovaný podiel udalostí')
plt.tight_layout(); plt.savefig(O + 'skupinova_kalibracia.png', dpi=200); plt.close()

#7) rekalibračné krivky (XGBoost)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
pp = np.linspace(0.005, 0.95, 200)
for ax, (wu, wt) in zip(axes, [(2, 3), (3, 4)]):
    U = P.query('model=="XGB" and vlna==@wu')
    for me, c, lab in [('none', GREY, 'bez rekalibrácie'), ('intercept', BLUE, 'intercept'), ('intercept_slope', ORANGE, 'intercept + sklon')]:
        ax.plot(pp, apply_recal(pp, fit_recal(U.y.values, U.p.values, me)), color=c, lw=1.8, label=lab)
    ax.set_title(f'naučené na vlne {wu}, použité na vlne {wt}', fontsize=10); ax.set_xlabel('Pôvodná predikcia p̂')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.xaxis.set_major_formatter(CF); ax.yaxis.set_major_formatter(CF); ax.grid(True, color=GRID)
axes[0].set_ylabel('Rekalibrovaná predikcia p̂*'); axes[0].legend(frameon=False, fontsize=8.5)
plt.tight_layout(); plt.savefig(O + 'rekalibracne_krivky.png', dpi=200); plt.close()

print(imp.head(10).round(1)); print(coef.head(10).round(3))
