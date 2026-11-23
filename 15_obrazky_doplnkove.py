"""Doplnkové obrázky praktickej časti: párové porovnanie modelov a skupinové CITL pred a po globálnej rekalibrácii."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from common import OUT_ as O

BLUE, ORANGE, GREEN, GREY, INK, MUTED, GRID = '#1f5fa8', '#b5532a', '#3b8b5a', '#888888', '#222222', '#555555', '#e5e5e5'
LIGHT = '#e8eef7'
plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888', 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
                     'legend.frameon': False})
CF = FuncFormatter(lambda v, _: (f'{v:g}').replace('.', ',').replace('-', '−'))
num = lambda x, d=2: f'{x:.{d}f}'.replace('.', ',').replace('-', '−')
expit = lambda z: 1 / (1 + np.exp(-z)); logit = lambda p: np.log(p / (1 - p))


# 1) párové porovnanie modelov
MC = pd.read_csv(O + 'porovnanie_modelov.csv')
fig, axes = plt.subplots(1, 2, figsize=(8.6, 2.6), sharey=True)
yb = np.array([2, 1, 0])
for ax, (met, tt, note) in zip(axes, [('Brier', 'a) Rozdiel Brierovho skóre', 'kladný rozdiel = lepšia log. regresia'),
                                      ('AUROC', 'b) Rozdiel AUROC', 'kladný rozdiel = lepší XGBoost')]):
    d = MC[MC.metrika == met].sort_values('vlna')
    ax.errorbar(d.rozdiel, yb, xerr=[d.rozdiel - d.lo, d.hi - d.rozdiel], fmt='o', color=BLUE, ms=6, capsize=3, lw=1.6)
    ax.axvline(0, color='#555', lw=1, ls='--'); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(CF); ax.set_title(tt, fontsize=9.5, color=INK, loc='left')
    ax.set_xlabel(f'XGBoost − logistická regresia ({note})', fontsize=8.6); ax.set_ylim(-0.6, 2.6)
    m = max(abs(d.lo.min()), abs(d.hi.max())) * 1.15; ax.set_xlim(-m, m)
axes[0].set_yticks(yb); axes[0].set_yticklabels(['vlna 2', 'vlna 3', 'vlna 4'])
plt.tight_layout(); plt.savefig(O + 'porovnanie_modelov.png', dpi=200); plt.close()

# 2) skupinové CITL pred a po globálnej rekalibrácii (XGBoost)
R4 = pd.read_csv(O + 'H4_globalne.csv'); R4 = R4[R4.model == 'XGB']
GR = [('muži − ženy', 'muži', 'muži', BLUE, '-'), ('muži − ženy', 'ženy', 'ženy', BLUE, '--'),
      ('vek > 80 − vek 66–80', '>80', 'vek > 80', ORANGE, '-'), ('vek > 80 − vek 66–80', '66–80', 'vek 66 – 80', ORANGE, '--'),
      ('DM áno − DM nie', 'áno', 'diabetes áno', GREEN, '-'), ('DM áno − DM nie', 'nie', 'diabetes nie', GREEN, '--')]
ME = ['none', 'intercept', 'intercept_slope']; ML = ['bez\nrekalibrácie', 'úprava\ninterceptu', 'úprava interceptu\na sklonu']
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9), sharey=True)
for ax, tr in zip(axes, ['2→3', '3→4']):
    q = R4[R4.prechod == tr].set_index('metoda').loc[ME]
    for c, g, lab, col, ls in GR:
        ax.plot(range(3), q[f'CITL[{c}|{g}]'], marker='o', ms=5, color=col, ls=ls, lw=1.5, label=lab)
    ax.axhline(0, color='#555', lw=1, ls=':')
    ax.set_xticks(range(3)); ax.set_xticklabels([f'{l}\n(rozpätie {num(r)})' for l, r in zip(ML, q.rozpatie)], fontsize=8.3)
    ax.set_xlim(-0.4, 2.4); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.yaxis.set_major_formatter(CF)
    ax.set_title(f'učenie na vlne {tr[0]} → test na vlne {tr[-1]}', fontsize=9.5, color=INK)
axes[0].set_ylabel('Skupinové CITL na testovacej vlne')
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=6, fontsize=8.2, columnspacing=1.2, handlelength=2.2)
plt.tight_layout(rect=(0, 0.07, 1, 1)); plt.savefig(O + 'h4_skupiny_posun.png', dpi=200); plt.close()
print('ok')
