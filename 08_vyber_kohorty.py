"""Výberové skreslenie a zloženie cieľovej udalosti:
(1) počty a podiely vylúčených (preložených) podľa vĺn, (2) zaradení vs. vylúčení – základné charakteristiky so štandardizovaným
rozdielom (SMD), (3) zloženie udalosti (úmrtia vs. preklady na ARO/OAIM) podľa vĺn.
Výstupy: out/vylucenie_podla_vln.csv, out/zaradeni_vs_vylucni.csv, out/zlozenie_udalosti.csv
"""
import numpy as np, pandas as pd
from scipy.stats import chi2_contingency
from common import DATA, OUT_

raw = pd.read_pickle(DATA); st = raw['Stav prepustenia']
raw['vylucny'] = (st == 'preložený').astype(int); raw['muz'] = (raw.Pohlavie == 'Muž').astype(int)

# (1) vylúčenie podľa vĺn
R1 = []
for w in [1, 2, 3, 4]:
    d = raw[raw.vlna == w]
    R1.append(dict(vlna=w, vsetci=len(d), vylucni=int(d.vylucny.sum()), podiel_vylucnych=d.vylucny.mean(),
                   zaradeni=int((1 - d.vylucny).sum())))
R1 = pd.DataFrame(R1)
chi = chi2_contingency(pd.crosstab(raw.vlna, raw.vylucny))[1]
R1.to_csv(OUT_ + 'vylucenie_podla_vln.csv', index=False)

# (2) zaradení vs. vylúčení
def smd_bin(a, b):
    p1, p2 = a.mean(), b.mean(); s = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / s if s > 0 else 0.0
def smd_num(a, b):
    a, b = a.dropna(), b.dropna(); s = np.sqrt((a.var() + b.var()) / 2)
    return (a.mean() - b.mean()) / s if s > 0 else 0.0
VARS = [('Vek', 'vek (roky), medián', 'num'), ('muz', 'muži (%)', 'bin'), ('Hypertenzia', 'hypertenzia (%)', 'bin'),
        ('Diabetes mellitus', 'diabetes mellitus (%)', 'bin'), ('Kardiovaskulárne ochorenia', 'kardiovaskulárne ochorenia (%)', 'bin'),
        ('Chronické respiračné ochorenia', 'chronické respiračné ochorenia (%)', 'bin'),
        ('Onkologické ochorenia', 'onkologické ochorenia (%)', 'bin'), ('Imunosupresia', 'imunosupresia (%)', 'bin'),
        ('Vakcinácia', 'očkovaní (%)', 'bin'), ('S-CRP_prva_hodnota', 'CRP (mg/l), medián', 'num'),
        ('sat_bez_o2_prva', 'saturácia bez O₂ (%), medián', 'num'), ('S-Urea_prva_hodnota', 'urea (mmol/l), medián', 'num'),
        ('Ly abs_prva_hodnota', 'lymfocyty (10⁹/l), medián', 'num')]
inc, exc = raw[raw.vylucny == 0], raw[raw.vylucny == 1]
R2 = []
for c, lab, t in VARS:
    a, b = inc[c].astype(float), exc[c].astype(float)
    if t == 'bin':
        R2.append(dict(premenna=lab, zaradeni=100 * a.mean(), vylucni=100 * b.mean(), SMD=smd_bin(b, a)))
    else:
        R2.append(dict(premenna=lab, zaradeni=a.median(), vylucni=b.median(), SMD=smd_num(b, a)))
R2 = pd.DataFrame(R2); R2.to_csv(OUT_ + 'zaradeni_vs_vylucni.csv', index=False)

# (3) zloženie udalosti
R3 = []
for w in [1, 2, 3, 4]:
    d = raw[(raw.vlna == w) & (st != 'preložený')]; s = d['Stav prepustenia']
    ex, aro = int((s == 'exitus letalis').sum()), int((s == 'ARO/OAIM').sum())
    R3.append(dict(vlna=w, zaradeni=len(d), umrtia=ex, aro=aro, udalosti=ex + aro, podiel_aro_z_udalosti=aro / (ex + aro),
                   podiel_umrti=ex / len(d), podiel_aro=aro / len(d)))
R3 = pd.DataFrame(R3); R3.to_csv(OUT_ + 'zlozenie_udalosti.csv', index=False)
pd.set_option('display.width', 200)
print(R1.round(3).to_string(index=False), 'chi2 p =', chi)
print(R2.round(2).to_string(index=False)); print(R3.round(3).to_string(index=False))
