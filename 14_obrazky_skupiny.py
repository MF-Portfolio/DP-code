"""Obrázky, ktoré nahrádzajú číselné tabuľky praktickej časti (v16): chýbajúce hodnoty podľa vĺn, vylúčení pacienti,
pätiny predikovaného rizika, skupinové CITL oboch modelov, diskriminácia a chybovosť podľa skupín, test H3.
Štýl zhodný s 13_obrazky_vysledky.py (desatinné čiarky, legendy mimo dát)."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from common import load, OUT_
from recal_utils import COMPARISONS

O = OUT_
BLUE, ORANGE, GREEN, GREY, INK, MUTED, GRID = '#1f5fa8', '#b5532a', '#3b8b5a', '#888888', '#222222', '#555555', '#e5e5e5'
WC = {2: BLUE, 3: ORANGE, 4: GREEN}
plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888', 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
                     'legend.frameon': False})
CF = FuncFormatter(lambda v, _: (f'{v:g}').replace('.', ',').replace('-', '−'))
num = lambda x, d=2: f'{x:.{d}f}'.replace('.', ',').replace('-', '−')
nt = lambda n: f'{int(n):,}'.replace(',', ' ')


FS = lambda w, h: (w * 0.82, h * 0.85)


def comma(ax, x=True, y=True):
    if x: ax.xaxis.set_major_formatter(CF)
    if y: ax.yaxis.set_major_formatter(CF)


ATR = pd.read_csv(O + 'atributy_pouzite.csv'); NMAP = ATR.set_index('premenna_v_subore').opis.to_dict()
G = pd.read_csv(O + 'skupiny_metriky.csv')
LBL = {'muži': 'muži', 'ženy': 'ženy', '>80': 'vek > 80', '66–80': 'vek 66 – 80', 'áno': 'diabetes áno', 'nie': 'diabetes nie'}
TIT = {'muži − ženy': 'Pohlavie', 'vek > 80 − vek 66–80': 'Vek', 'DM áno − DM nie': 'Diabetes mellitus'}
MOD = [('XGB', 'XGBoost (hlavný model)'), ('LR', 'Logistická regresia (referenčný model)')]

# 1) chýbajúce hodnoty podľa vĺn (teplotná mapa)
coh = load()
VAR = ['S-Chol_prva_hodnota', 'S-IgA_prva_hodnota', 'S-IgG_prva_hodnota', 'S-AMS_prva_hodnota', 'S-LD_prva_hodnota',
       'sat_s_o2_prva', 'NK_prva_hodnota', 'CD19+_prva_hodnota', 'sat_bez_o2_prva', 'S-KM_prva_hodnota',
       'S-IL6_prva_hodnota', 'S-PBNP_prva_hodnota', 'S-FER_prva_hodnota', 'D-dimér HS_prva_hodnota', 'S-CRP_prva_hodnota']
M = np.array([[100 * coh.loc[coh.vlna == w, c].isna().mean() for w in [1, 2, 3, 4]] for c in VAR])
names = [NMAP.get(c, c).replace('prvá zaznamenaná hodnota: ', '').replace('prvá zaznamenaná ', '') for c in VAR]
fig, ax = plt.subplots(figsize=FS(8.6, 6.2))
im = ax.imshow(M, cmap='Blues', vmin=0, vmax=100, aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, num(M[i, j], 1), ha='center', va='center', fontsize=8.6, color='white' if M[i, j] > 55 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels([f'vlna {w}' for w in [1, 2, 3, 4]]); ax.xaxis.tick_top()
ax.set_yticks(range(len(VAR))); ax.set_yticklabels(names, fontsize=8.8)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03); cb.set_label('Chýbajúce hodnoty (%)'); cb.ax.yaxis.set_major_formatter(CF)
cb.outline.set_visible(False)
plt.tight_layout(); plt.savefig(O + 'chybajuce_vlny.png', dpi=200); plt.close()

# 2) vylúčení (preložení) pacienti: podiel podľa vĺn a SMD
VYL = pd.read_csv(O + 'vylucenie_podla_vln.csv'); VCH = pd.read_csv(O + 'zaradeni_vs_vylucni.csv')
fig, (a1, a2) = plt.subplots(1, 2, figsize=FS(10.5, 4.3), gridspec_kw=dict(width_ratios=[1, 1.45]))
y = np.arange(4)[::-1]
a1.barh(y, VYL.zaradeni, color=BLUE, label='zaradené')
a1.barh(y, VYL.vylucni, left=VYL.zaradeni, color=ORANGE, label='vylúčené (preložené)')
for yy, r in zip(y, VYL.itertuples()):
    a1.text(r.zaradeni / 2, yy, nt(r.zaradeni), ha='center', va='center', color='white', fontsize=8.6)
    a1.text(r.vsetci + 15, yy, f'{nt(r.vylucni)} ({num(100 * r.podiel_vylucnych, 1)} %)', va='center', fontsize=8.6, color=INK)
a1.set_yticks(y); a1.set_yticklabels([f'vlna {w}' for w in VYL.vlna]); a1.set_xlim(0, 1500)
a1.set_xlabel('Počet hospitalizácií'); a1.set_title('a) Vylúčené hospitalizácie podľa vĺn', fontsize=9.5, color=INK, loc='left')
a1.xaxis.grid(True, color=GRID); a1.set_axisbelow(True); a1.xaxis.set_major_formatter(FuncFormatter(lambda v, _: nt(v)))
a1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=8.5)
d = VCH.sort_values('SMD'); yy = np.arange(len(d))
a2.axvspan(-0.1, 0.1, color='#f0f0f0', zorder=0)
a2.axvline(0, color='#555', lw=1)
for v in (-0.1, 0.1): a2.axvline(v, color=GREY, lw=1, ls='--')
a2.hlines(yy, 0, d.SMD, color='#bbb', lw=1.2)
a2.scatter(d.SMD, yy, s=34, zorder=3, color=[ORANGE if abs(v) > 0.1 else GREY for v in d.SMD])
a2.set_yticks(yy); a2.set_yticklabels(d.premenna, fontsize=8.5); a2.set_xlim(-0.25, 0.4)
a2.set_xlabel('SMD (vylúčené − zaradené)'); a2.xaxis.grid(True, color=GRID); a2.set_axisbelow(True); comma(a2, y=False)
a2.set_title('b) Štandardizovaný rozdiel charakteristík', fontsize=9.5, color=INK, loc='left')
plt.tight_layout(); plt.savefig(O + 'vylucni_smd.png', dpi=200); plt.close()

# 3) pätiny predikovaného rizika
PAT = pd.read_csv(O + 'patiny_vlny.csv')
fig, axes = plt.subplots(1, 3, figsize=FS(10.5, 3.7), sharey=True)
for ax, w in zip(axes, [2, 3, 4]):
    q = PAT[PAT.vlna == w]; x = np.arange(1, 6)
    ax.bar(x - 0.19, 100 * q.predikovane, 0.38, color='#9db7d9', label='predikované riziko')
    ax.bar(x + 0.19, 100 * q.pozorovane, 0.38, color=BLUE, label='pozorovaný podiel udalostí')
    for xx, r in zip(x, q.itertuples()):
        dd = 100 * (r.pozorovane - r.predikovane)
        if abs(dd) >= 2:
            ax.text(xx, max(r.pozorovane, r.predikovane) * 100 + 2, f'{dd:+.1f}'.replace('.', ',').replace('-', '−'),
                    ha='center', fontsize=7.8, color=ORANGE if dd < 0 else INK)
    ax.set_xticks(x); ax.set_xlabel('Pätina predikovaného rizika'); ax.set_title(f'vlna {w}', fontsize=9.5, color=INK)
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_ylim(0, 80)
axes[0].set_ylabel('Riziko (%)')
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=8.8)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(O + 'patiny_vlny.png', dpi=200); plt.close()

# 4) skupinové CITL oboch modelov
fig, axes = plt.subplots(2, 3, figsize=FS(10.5, 6.4), sharey=True)
for i, (m, mn) in enumerate(MOD):
    for j, (comp, (col, a, b)) in enumerate(COMPARISONS.items()):
        ax = axes[i, j]
        for lvl, c, off in [(a, BLUE, -0.07), (b, ORANGE, 0.07)]:
            d = G.query('model==@m and porovnanie==@comp and skupina==@lvl').sort_values('vlna')
            ax.errorbar(d.vlna + off, d.CITL, yerr=[d.CITL - d.CITL_lo, d.CITL_hi - d.CITL], fmt='o-', color=c, ms=5,
                        capsize=3, lw=1.5, label=LBL[lvl])
        ax.axhline(0, color='#555', lw=1, ls='--'); ax.set_xticks([2, 3, 4])
        ax.set_xticklabels(['vlna 2', 'vlna 3', 'vlna 4']); ax.set_xlim(1.7, 4.3)
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax, x=False)
        if i == 0:
            ax.set_title(TIT[comp], fontsize=9.5, color=INK)
        else:
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
        if j == 0: ax.set_ylabel(f'CITL – {"XGBoost" if m == "XGB" else "log. regresia"}')
fig.text(0.5, 0.995, 'Kladné CITL = model riziko podhodnocuje, záporné = nadhodnocuje', ha='center', va='top', fontsize=8.8, color=MUTED)
plt.tight_layout(rect=(0, 0, 1, 0.97)); plt.savefig(O + 'skupinove_citl.png', dpi=200); plt.close()

# 5) diskriminácia a chybovosť podľa skupín
ORD = [('muži − ženy', 'muži'), ('muži − ženy', 'ženy'), ('vek > 80 − vek 66–80', '>80'), ('vek > 80 − vek 66–80', '66–80'),
       ('DM áno − DM nie', 'áno'), ('DM áno − DM nie', 'nie')]
MET = [('AUROC', 'AUROC', (0.5, 1.0)), ('sklon', 'Kalibračný sklon', (0, 2.6)), ('TPR', 'Citlivosť (TPR)', (0.5, 1.03)),
       ('FPR', 'Falošné poplachy (FPR)', (0, 0.8))]
THR = json.load(open(O + 'prah_vlna1.json'))
fig, axes = plt.subplots(2, 4, figsize=FS(10.5, 6.2), sharey=True)
yb = np.arange(len(ORD))[::-1]
for i, (m, mn) in enumerate(MOD):
    for j, (met, mt, lim) in enumerate(MET):
        ax = axes[i, j]
        for k in range(0, len(ORD), 2): ax.axhspan(yb[k] - 1.5, yb[k] + 0.5, color='#f6f6f6' if k % 4 == 0 else 'white', zorder=0)
        for w, off in [(2, 0.22), (3, 0), (4, -0.22)]:
            v = [G.query('model==@m and vlna==@w and porovnanie==@c and skupina==@s')[met].iloc[0] for c, s in ORD]
            ax.scatter(v, yb + off, s=22, color=WC[w], label=f'vlna {w}', zorder=3)
        if met == 'sklon': ax.axvline(1, color='#555', lw=1, ls='--')
        ax.set_xlim(*lim); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax, y=False)
        ax.set_ylim(-0.6, len(ORD) - 0.4)
        if i == 0: ax.set_title(mt, fontsize=9.3, color=INK)
        ax.tick_params(axis='x', labelsize=8)
    axes[i, 0].set_yticks(yb); axes[i, 0].set_yticklabels([LBL[s] for _, s in ORD], fontsize=8.6)
    axes[i, 0].set_ylabel(f'{"XGBoost" if m == "XGB" else "Log. regresia"} (prah {num(THR[m], 3)})')
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=3, fontsize=8.8)
plt.tight_layout(rect=(0, 0.05, 1, 1)); plt.savefig(O + 'chybovost_skupiny.png', dpi=200); plt.close()

#  6) test H3: rozdiel zmien a zmena absolútneho rozdielu
H3 = pd.read_csv(O + 'H3.csv')
fig, axes = plt.subplots(1, 2, figsize=FS(10.5, 3.4), sharey=True)
yb = np.arange(3)[::-1]
for ax, (col, lo, hi, tt) in zip(axes, [('rozdiel_zmien', 'lo', 'hi', 'a) Rozdiel zmien (CITL₄ − CITL₂) v skupine A mínus B'),
                                        ('zmena_abs_rozdielu', 'abs_lo', 'abs_hi', 'b) Zmena absolútneho rozdielu |Δ₄| − |Δ₂|')]):
    for m, c, off in [('XGB', BLUE, 0.12), ('LR', ORANGE, -0.12)]:
        d = H3[H3.model == m].set_index('porovnanie').loc[list(COMPARISONS)]
        ax.errorbar(d[col], yb + off, xerr=[d[col] - d[lo], d[hi] - d[col]], fmt='o', color=c, ms=6, capsize=3, lw=1.6,
                    label='XGBoost' if m == 'XGB' else 'logistická regresia')
    ax.axvline(0, color='#555', lw=1, ls='--'); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax, y=False)
    ax.set_title(tt, fontsize=9.3, color=INK, loc='left')
axes[0].set_yticks(yb); axes[0].set_yticklabels(['muži − ženy', 'vek > 80 − vek 66 – 80', 'diabetes áno − nie'])
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=8.8)
plt.tight_layout(rect=(0, 0.1, 1, 1)); plt.savefig(O + 'h3_rozdiel_zmien.png', dpi=200); plt.close()
print('ok')
