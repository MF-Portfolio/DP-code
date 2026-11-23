"""Finálne verzie obrázkov praktickej časti: jednotný štýl, desatinné čiarky, legendy mimo dát, bez prekrývania textu.
Prepisuje: tok_pacientov, posun_vlny, metriky_v_case, roc_krivky, schema_validacie, kalibracne_grafy, h1_rozdiel,
h2_rozdiely, h4_rekalibracia, zlozenie_udalosti, udalosti_skupiny, postup_prehlad."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import roc_curve
from common import load, OUT_, DATA
from recal_utils import COMPARISONS

O = OUT_
BLUE, ORANGE, GREEN, GREY, INK, MUTED, GRID = '#1f5fa8', '#b5532a', '#3b8b5a', '#888888', '#222222', '#555555', '#e5e5e5'
plt.rcParams.update({'font.size': 9.5, 'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#888', 'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.labelcolor': INK,
                     'legend.frameon': False})
CF = FuncFormatter(lambda v, _: (f'{v:g}').replace('.', ',').replace('-', '−'))
num = lambda x, d=2: f'{x:.{d}f}'.replace('.', ',').replace('-', '−')
nt = lambda n: f'{int(n):,}'.replace(',', ' ')


def comma(ax, x=True, y=True):
    if x: ax.xaxis.set_major_formatter(CF)
    if y: ax.yaxis.set_major_formatter(CF)


raw = pd.read_pickle(DATA); df = load('exclude'); st = raw['Stav prepustenia'].value_counts()
w1 = json.load(open(O + 'wave1_internal.json')); tm = pd.read_csv(O + 'temporal_metrics.csv')


def box(ax, x, y, w, h, t, fc, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04', fc=fc, ec='#555', lw=1))
    ax.text(x + w / 2, y + h / 2, t, ha='center', va='center', fontsize=fs, color=INK)


def arr(ax, x1, y1, x2, y2):
    ax.annotate('', (x2, y2), (x1, y1), arrowprops=dict(arrowstyle='->', lw=1.2, color='#444'))


#prehľad postupu
fig, ax = plt.subplots(figsize=(10, 2.3)); ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(0, 2.3)
for x, t, c in [(0.05, f'Dáta\n{nt(len(raw))} hospitalizácií\n579 premenných', '#eef3fa'),
                (2.05, 'Definícia úlohy\nudalosť, 65 vstupov,\nvylúčenie preložených', '#eef3fa'),
                (4.05, 'Vlna 1\nvnorená 5 × 5 CV,\nfinálny model', '#e8f3ec'),
                (6.05, 'Vlny 2, 3, 4\nzafixovaný model\n(bez zmien)', '#fbefe8'),
                (8.05, 'Hodnotenie\nH1 · H2 · H3 · H4,\ncitlivostná analýza', '#f3f3f3')]:
    box(ax, x, 0.75, 1.75, 1.35, t, c, 8.8)
for x in [1.8, 3.8, 5.8, 7.8]:
    arr(ax, x + 0.02, 1.42, x + 0.23, 1.42)
ax.text(5, 0.25, 'Rekalibrácia (H4): parametre sa učia na predchádzajúcej vlne a hodnotia na nasledujúcej (2 → 3, 3 → 4)',
        ha='center', fontsize=8.8, color=MUTED)
plt.tight_layout(pad=0.3); plt.savefig(O + 'postup_prehlad.png', dpi=200); plt.close()

#tok pacientov
fig, ax = plt.subplots(figsize=(10, 4.8)); ax.axis('off'); ax.set_xlim(0, 11); ax.set_ylim(0, 5.2)
box(ax, 3.8, 4.2, 3.4, 0.8, f'Hospitalizácie s COVID-19\nn = {nt(len(raw))}', '#f2f2f2', 9.5)
box(ax, 7.75, 2.95, 3.15, 1.0, f'Vylúčené: preklad na iné pracovisko\n(neznámy výsledok), n = {st["preložený"]}\n→ citlivostná analýza', '#fbeee6')
box(ax, 3.8, 2.9, 3.4, 0.9, f'Analyzovaná kohorta, n = {nt(len(df))}\nudalosť n = {df.y.sum()}\n(úmrtie {st["exitus letalis"]}, ARO/OAIM {st["ARO/OAIM"]})', '#f2f2f2', 9.5)
arr(ax, 5.5, 4.2, 5.5, 3.8); arr(ax, 5.5, 4.0, 7.75, 3.45)
lab = {1: 'tréning a interná validácia', 2: 'časový test', 3: 'časový test', 4: 'časový test'}
for i, w in enumerate([1, 2, 3, 4]):
    d = df[df.vlna == w]; x = 0.3 + i * 2.7
    box(ax, x, 0.55, 2.3, 1.35, f'Vlna {w}\nn = {nt(len(d))}, udalosti {d.y.sum()}\n({num(100 * d.y.mean(), 1)} %)\n{lab[w]}',
        '#dbe7f5' if w == 1 else '#f8e3d6')
    arr(ax, 5.5, 2.9, x + 1.15, 1.9)
plt.tight_layout(pad=0.3); plt.savefig(O + 'tok_pacientov.png', dpi=200); plt.close()

#posun medzi vlnami
g = df.groupby('vlna'); ev = g.y.mean(); n = g.size(); se = np.sqrt(ev * (1 - ev) / n)
fig, axes = plt.subplots(1, 4, figsize=(11, 3.2))
items = [('Podiel udalostí (%)', 100 * ev, 100 * 1.96 * se), ('Medián veku (roky)', g.Vek.median(), None),
         ('Podiel očkovaných (%)', 100 * g['Vakcinácia'].mean(), None),
         ('Medián CRP (mg/l)', g['S-CRP_prva_hodnota'].median(), None)]
for ax, (t, v, e) in zip(axes, items):
    ax.bar(v.index, v.values, color=BLUE, width=0.6, yerr=e, capsize=3, error_kw=dict(ecolor=INK, lw=1))
    top = v.values + (e.values if e is not None else 0)
    for x, y, tp in zip(v.index, v.values, top):
        ax.text(x, tp + v.max() * 0.04, num(y, 1), ha='center', va='bottom', fontsize=8.5, color=INK)
    ax.set_title(t, fontsize=9.5, color=INK); ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(['vlna 1', 'vlna 2', 'vlna 3', 'vlna 4'], fontsize=8)
    ax.set_ylim(0, top.max() * 1.22); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax, x=False)
plt.tight_layout(); plt.savefig(O + 'posun_vlny.png', dpi=200); plt.close()

# metriky v čase
fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
for ax, met, ttl, ref in zip(axes, ['AUROC', 'CITL', 'sklon'], ['AUROC', 'CITL (0 = ideálna hodnota)', 'Kalibračný sklon (1 = ideálna hodnota)'], [None, 0, 1]):
    for k, nm, col, off in [('XGB', 'XGBoost', BLUE, -0.08), ('LR', 'logistická regresia', ORANGE, 0.08)]:
        ys = [w1[k]['pooled'][met]] + list(tm[tm.model == k][met])
        lo = [w1[k]['ci'][met][0]] + list(tm[tm.model == k][met + '_lo'])
        hi = [w1[k]['ci'][met][1]] + list(tm[tm.model == k][met + '_hi'])
        ax.errorbar(np.array([1, 2, 3, 4]) + off, ys, yerr=[np.array(ys) - lo, np.array(hi) - ys], fmt='o-', ms=5, lw=1.8,
                    capsize=3, color=col, label=nm)
    if ref is not None: ax.axhline(ref, ls='--', color=GREY, lw=1)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(['vlna 1\n(tréning)', 'vlna 2', 'vlna 3', 'vlna 4'], fontsize=8.5)
    ax.set_title(ttl, fontsize=9.5, color=INK); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax, x=False)
h, l = axes[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=2, fontsize=9)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig(O + 'metriky_v_case.png', dpi=200); plt.close()

#ROC krivky
P = pd.read_csv(O + 'predictions_waves2-4.csv'); oof = pd.read_csv(O + 'oof_wave1.csv')
shades = {1: '#9fc0e6', 2: '#5f8fcc', 3: '#2d62a8', 4: '#0b2f66'}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, (k, nm) in zip(axes, [('XGB', 'XGBoost'), ('LR', 'Logistická regresia')]):
    curves = [(1, oof.y, oof[f'oof_{k}'])] + [(w, d.y, d.p) for w, d in P[P.model == k].groupby('vlna')]
    for w, y, p in curves:
        fpr, tpr, _ = roc_curve(y, p)
        a = w1[k]['pooled']['AUROC'] if w == 1 else tm.query('model==@k and vlna==@w').AUROC.iloc[0]
        ax.plot(fpr, tpr, lw=1.8, color=shades[w], label=f"vlna {w}{' (krížová validácia)' if w == 1 else ''}: AUROC {num(a, 3)}")
    ax.plot([0, 1], [0, 1], ls='--', color=GREY, lw=1)
    ax.set_title(nm, fontsize=10, color=INK); ax.set_xlabel('1 − špecificita'); ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax)
axes[0].set_ylabel('Citlivosť')
plt.tight_layout(); plt.savefig(O + 'roc_krivky.png', dpi=200); plt.close()

# schéma validácie
fig, ax = plt.subplots(figsize=(11, 5.0)); ax.axis('off'); ax.set_xlim(0, 12); ax.set_ylim(0, 5.6)
B, G, Or = '#dbe7f5', '#e3f1df', '#f8e3d6'
ax.text(0.1, 5.35, 'A. Interná validácia na vlne 1 (vnorená krížová validácia)', fontsize=10.5, weight='bold')
box(ax, 0.2, 3.6, 2.0, 1.2, 'Vlna 1\nn = 890, 174 udalostí', B)
box(ax, 2.8, 3.6, 2.7, 1.2, 'Vonkajšia 5-násobná\nkrížová validácia\n(4/5 tréning, 1/5 test)', B)
box(ax, 6.1, 3.6, 3.0, 1.2, 'Vnútorná 5-násobná\nkrížová validácia:\npredspracovanie a 40 kombinácií\nhyperparametrov (log-loss)', B)
box(ax, 9.7, 3.6, 2.1, 1.2, 'Spojené predikcie\ntestovacích častí:\nAUROC, CITL, sklon,\nBrierovo skóre', B)
arr(ax, 2.2, 4.2, 2.8, 4.2); arr(ax, 5.5, 4.2, 6.1, 4.2); arr(ax, 9.1, 4.2, 9.7, 4.2)
ax.text(0.1, 2.85, 'B. Finálny model a časové testovanie', fontsize=10.5, weight='bold')
box(ax, 0.2, 1.0, 2.0, 1.4, 'Celá vlna 1\nladenie 5-násobnou\nkrížovou validáciou', G)
box(ax, 2.8, 1.0, 2.7, 1.4, 'Tréning na celej vlne 1;\nmodel a predspracovanie\nsa zafixujú', G)
for i, (w, t) in enumerate([(2, 'n = 647, 127 udalostí'), (3, 'n = 522, 71 udalostí'), (4, f'n = {nt(1003)}, 132 udalostí')]):
    box(ax, 6.1 + i * 1.95, 1.0, 1.75, 1.4, f'Vlna {w}\n{t}\nlen predikcia', Or, 8.5)
arr(ax, 2.2, 1.7, 2.8, 1.7); arr(ax, 5.5, 1.7, 6.1, 1.7)
ax.text(0.2, 0.3, 'Predspracovanie (vyradenie premenných s viac ako 50 % chýbajúcich hodnôt, imputácia, transformácia, škálovanie) '
        'sa učí vždy len na tréningovej časti.', fontsize=8.3, style='italic', color=MUTED)
plt.tight_layout(pad=0.3); plt.savefig(O + 'schema_validacie.png', dpi=200); plt.close()

# kalibračné grafy
fig, axes = plt.subplots(2, 3, figsize=(10.5, 7.2), sharex=True, sharey=True)
for i, (k, nm, col) in enumerate([('XGB', 'XGBoost', BLUE), ('LR', 'Logistická regresia', ORANGE)]):
    for j, w in enumerate([2, 3, 4]):
        ax = axes[i, j]; d = P.query('model==@k and vlna==@w')
        q = pd.qcut(d.p, 10, duplicates='drop')
        gg = d.groupby(q, observed=True).agg(p=('p', 'mean'), y=('y', 'mean'), n=('y', 'size'))
        se_ = np.sqrt(gg.y * (1 - gg.y) / gg.n)
        ax.plot([0, 1], [0, 1], ls='--', c=GREY, lw=1)
        ax.errorbar(gg.p, gg.y, yerr=1.96 * se_, fmt='o-', ms=4, capsize=2, c=col, lw=1.5)
        r = tm.query('model==@k and vlna==@w').iloc[0]
        ax.set_title(f'{nm} – vlna {w}', fontsize=9.5, color=INK)
        ax.text(0.04, 0.95, f'CITL {num(r.CITL)}\nsklon {num(r.sklon)}\nAUROC {num(r.AUROC, 3)}', transform=ax.transAxes,
                va='top', fontsize=8.3, color=INK, bbox=dict(fc='white', ec='#ccc', lw=0.6, pad=3))
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax)
        if i == 1: ax.set_xlabel('Predikované riziko')
        if j == 0: ax.set_ylabel('Pozorovaný podiel udalostí')
plt.tight_layout(); plt.savefig(O + 'kalibracne_grafy.png', dpi=200); plt.close()

#H1
h = json.load(open(O + 'H1.json')); s = json.load(open(O + 'sens_H1.json')); s2 = json.load(open(O + 'sens2_H1.json'))
rows = [('XGBoost – hlavná analýza', h['XGB'], BLUE, 'o', True), ('Logistická regresia – hlavná analýza', h['LR'], ORANGE, 'o', True),
        ('XGBoost – preložený = bez udalosti', s['XGB'], BLUE, 's', False), ('Logistická regresia – preložený = bez udalosti', s['LR'], ORANGE, 's', False),
        ('XGBoost – preložený = udalosť', s2['XGB'], BLUE, '^', True), ('Logistická regresia – preložený = udalosť', s2['LR'], ORANGE, '^', True)]
fig, ax = plt.subplots(figsize=(10, 3.9))
for i, (nm, r, c, mk, fill) in enumerate(rows):
    y = len(rows) - 1 - i
    ax.errorbar(r['diff'], y, xerr=[[r['diff'] - r['lo']], [r['hi'] - r['diff']]], fmt=mk, color=c, ms=7, capsize=4, lw=1.8,
                mfc=c if fill else 'white', mew=1.8)
    ax.text(1.03, y, f"{num(r['diff'])} [{num(r['lo'])}; {num(r['hi'])}]", va='center', fontsize=8.8, color=INK,
            transform=ax.get_yaxis_transform())
ax.axvline(0, color='#555', lw=1, ls='--')
ax.set_yticks(range(len(rows))[::-1]); ax.set_yticklabels([r[0] for r in rows], fontsize=8.8)
ax.set_xlabel('|CITL₄| − |CITL₂| (kladná hodnota = väčšia odchýlka vo vlne 4)'); ax.set_xlim(-0.4, 1.0)
ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax, y=False)
plt.tight_layout(rect=(0, 0, 0.84, 1)); plt.savefig(O + 'h1_rozdiel.png', dpi=200); plt.close()

# H2
TIT = {'muži − ženy': 'muži − ženy', 'vek > 80 − vek 66–80': 'vek > 80 − vek 66 – 80', 'DM áno − DM nie': 'diabetes áno − nie'}
H2 = pd.read_csv(O + 'H2.csv')
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), sharex=True)
for ax, comp in zip(axes, COMPARISONS):
    for m, c, off, lb in [('XGB', BLUE, 0.13, 'XGBoost'), ('LR', ORANGE, -0.13, 'logistická regresia')]:
        d = H2.query('model==@m and porovnanie==@comp'); y = 4 - d.vlna.values + off
        ax.errorbar(d.rozdiel, y, xerr=[d.rozdiel - d.lo, d.hi - d.rozdiel], fmt='o', color=c, ms=5, capsize=3, lw=1.6, label=lb)
        for yy, r in zip(y, d.itertuples()):
            if r.p_holm < 0.05:
                ax.text(r.hi + 0.06, yy, '*', color=c, fontsize=13, va='center')
    ax.axvline(0, color='#555', lw=1, ls='--'); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(['vlna 2', 'vlna 3', 'vlna 4'])
    ax.set_title(TIT[comp], fontsize=9.5, color=INK); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_xlim(-1.9, 1.5); ax.set_ylim(-0.5, 2.5); comma(ax, y=False)
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=9)
fig.text(0.5, 0.11, 'Rozdiel CITL medzi skupinami (A − B) s 95 % IS; * = významné po Holmovej korekcii', ha='center', fontsize=9, color=INK)
plt.tight_layout(rect=(0, 0.15, 1, 1)); plt.savefig(O + 'h2_rozdiely.png', dpi=200); plt.close()

#H4
R = pd.read_csv(O + 'H4_globalne.csv')
MN = {'none': 'bez rekalibrácie', 'intercept': 'intercept', 'intercept_slope': 'intercept + sklon'}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharex=True)
for ax, tr in zip(axes, ['2→3', '3→4']):
    for m, c, off, lb in [('XGB', BLUE, 0.13, 'XGBoost'), ('LR', ORANGE, -0.13, 'logistická regresia')]:
        d = R.query('model==@m and prechod==@tr').set_index('metoda').loc[list(MN)]
        y = np.array([2, 1, 0]) + off
        ax.errorbar(d.CITL, y, xerr=[d.CITL - d.CITL_lo, d.CITL_hi - d.CITL], fmt='o', color=c, ms=5, capsize=3, lw=1.6, label=lb)
    ax.axvline(0, color='#555', lw=1, ls='--'); ax.set_yticks([2, 1, 0]); ax.set_yticklabels(list(MN.values()))
    ax.set_ylim(-0.5, 2.5); ax.set_title(f'učenie na vlne {tr[0]} → test na vlne {tr[-1]}', fontsize=9.5, color=INK)
    ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True); comma(ax, y=False)
h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, loc='lower center', ncol=2, fontsize=9)
fig.text(0.5, 0.11, 'CITL na testovacej vlne s 95 % intervalom spoľahlivosti', ha='center', fontsize=9, color=INK)
plt.tight_layout(rect=(0, 0.15, 1, 1)); plt.savefig(O + 'h4_rekalibracia.png', dpi=200); plt.close()

#zloženie udalosti
Z = pd.read_csv(O + 'zlozenie_udalosti.csv')
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
ct = pd.crosstab(raw.vlna, raw['Stav prepustenia'], normalize='index') * 100
bot = np.zeros(4)
for k, c, l in [('prepustenie amb. star.', '#c9d6e8', 'prepustenie'), ('preložený', '#e6c9a8', 'preklad na iné pracovisko'),
                ('exitus letalis', ORANGE, 'úmrtie'), ('ARO/OAIM', BLUE, 'preklad na ARO/OAIM')]:
    axes[0].bar([1, 2, 3, 4], ct[k].values, bottom=bot, color=c, label=l, width=0.6); bot += ct[k].values
axes[0].set_xticks([1, 2, 3, 4]); axes[0].set_xticklabels(['vlna 1', 'vlna 2', 'vlna 3', 'vlna 4'])
axes[0].set_ylabel('% hospitalizácií'); axes[0].set_title('Stav pri ukončení hospitalizácie', fontsize=9.5, color=INK)
axes[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
axes[1].bar(Z.vlna - 0.18, Z.umrtia, 0.36, color=ORANGE, label='úmrtia'); axes[1].bar(Z.vlna + 0.18, Z.aro, 0.36, color=BLUE, label='preklady na ARO/OAIM')
for _, r in Z.iterrows():
    axes[1].text(r.vlna - 0.18, r.umrtia + 2, int(r.umrtia), ha='center', va='bottom', fontsize=8)
    axes[1].text(r.vlna + 0.18, r.aro + 2, int(r.aro), ha='center', va='bottom', fontsize=8)
axes[1].set_ylim(0, Z.umrtia.max() * 1.18)
axes[1].set_xticks([1, 2, 3, 4]); axes[1].set_xticklabels(['vlna 1', 'vlna 2', 'vlna 3', 'vlna 4'])
axes[1].set_title('Zložky cieľovej udalosti (počty)', fontsize=9.5, color=INK)
axes[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=8)
plt.tight_layout(); plt.savefig(O + 'zlozenie_udalosti.png', dpi=200); plt.close()

# ---------- počty udalostí v skupinách
c = pd.read_csv(O + 'pocty_skupiny.csv')
sel = [('Pohlavie', 'Muž', 'muži'), ('Pohlavie', 'Žena', 'ženy'), ('Vek', '<50', 'vek < 50'), ('Vek', '50–65', 'vek 50 – 65'),
       ('Vek', '66–80', 'vek 66 – 80'), ('Vek', '>80', 'vek > 80'), ('Diabetes mellitus', 'áno', 'diabetes áno'),
       ('Diabetes mellitus', 'nie', 'diabetes nie'), ('Onkologické ochorenia', 'áno', 'onkológia'), ('Imunosupresia', 'áno', 'imunosupresia')]
fig, ax = plt.subplots(figsize=(10.5, 3.6)); x = np.arange(len(sel)); WC = {2: BLUE, 3: ORANGE, 4: GREEN}
for k, w in enumerate([2, 3, 4]):
    v = [int(c[(c['atribút'] == a) & (c.skupina == s_)][f'v{w}_ud'].iloc[0]) for a, s_, _ in sel]
    ax.bar(x + (k - 1) * 0.27, v, 0.27, color=WC[w], label=f'vlna {w}')
ax.axhline(20, ls='--', color='#444', lw=1, label='hranica 20 udalostí'); ax.axhline(10, ls=':', color='#777', lw=1, label='hranica 10 udalostí')
ax.set_xticks(x); ax.set_xticklabels([l for *_, l in sel], fontsize=8.5); ax.set_ylabel('Počet udalostí')
ax.legend(loc='upper right', fontsize=8.3, ncol=1); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
plt.tight_layout(); plt.savefig(O + 'udalosti_skupiny.png', dpi=200); plt.close()
print('ok')
