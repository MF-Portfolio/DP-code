"""Zoznam všetkých atribútov súboru so zaradením (použitý / nepoužitý a dôvod) a podrobný zoznam 65 použitých vstupov.
Výstupy: out/atributy_vsetky.csv, out/atributy_pouzite.csv, out/Zoznam_atributov.xlsx
"""
import re, numpy as np, pandas as pd
from common import load, feature_list, OUT_, DATA

raw = pd.read_pickle(DATA)
df = load('exclude'); feats = feature_list(df)
w1 = df[df.vlna == 1]
LAB = {
    'S-Bil-T': 'bilirubín celkový', 'S-AST': 'aspartátaminotransferáza (AST)', 'S-ALT': 'alanínaminotransferáza (ALT)',
    'S-GMT': 'gama-glutamyltransferáza (GMT)', 'S-ALP': 'alkalická fosfatáza (ALP)', 'S-Chol': 'cholesterol',
    'S-CB': 'celková bielkovina', 'S-Na': 'sodík', 'S-K': 'draslík', 'S-CL': 'chloridy', 'S-IgG': 'imunoglobulín G',
    'S-IgA': 'imunoglobulín A', 'S-Ig M': 'imunoglobulín M', 'S-CRP': 'C-reaktívny proteín (CRP)', 'S-Alb': 'albumín',
    'S-Bil-D': 'bilirubín priamy', 'S-Gluk': 'glukóza', 'S-Urea': 'urea', 'S-Kreat': 'kreatinín', 'S-KM': 'kyselina močová',
    'S-AMS': 'amyláza', 'S-CK': 'kreatínkináza (CK)', 'S-CK-MB': 'kreatínkináza MB', 'S-LD': 'laktátdehydrogenáza (LD)',
    'P-Laktát': 'laktát', 'S-FER': 'feritín', 'HGB': 'hemoglobín', 'WBC': 'leukocyty', 'PLT': 'trombocyty',
    'Neu abs': 'neutrofily (absolútny počet)', 'Eo abs': 'eozinofily (absolútny počet)', 'Ly abs': 'lymfocyty (absolútny počet)',
    'PT (INR)': 'protrombínový čas (INR)', 'APTT-R': 'aktivovaný parciálny tromboplastínový čas (pomer)', 'Fib': 'fibrinogén',
    'CD3+': 'CD3+ T-lymfocyty', 'CD4+': 'CD4+ T-lymfocyty', 'CD8+': 'CD8+ T-lymfocyty', 'CD4+/CD8+': 'pomer CD4+/CD8+',
    'PDW': 'distribučná šírka trombocytov (PDW)', 'S-PBNP': 'NT-proBNP', 'S-IL6': 'interleukín 6', 'CD19+': 'CD19+ B-lymfocyty',
    'NK': 'NK bunky', 'S-VITD': 'vitamín D', 'NE/LY(NLR)': 'pomer neutrofilov a lymfocytov (NLR)', 'D-dimér HS': 'D-dimér',
}
OTHER = {'Vek': ('vek (roky)', 'demografia', 'číselná'), 'muz': ('pohlavie (1 = muž), z premennej Pohlavie', 'demografia', 'binárna'),
         'Hypertenzia': ('hypertenzia', 'komorbidita', 'binárna'), 'Diabetes mellitus': ('diabetes mellitus', 'komorbidita', 'binárna'),
         'Kardiovaskulárne ochorenia': ('kardiovaskulárne ochorenia', 'komorbidita', 'binárna'),
         'Chronické respiračné ochorenia': ('chronické respiračné ochorenia', 'komorbidita', 'binárna'),
         'Pečeňové ochorenia': ('pečeňové ochorenia', 'komorbidita', 'binárna'),
         'Onkologické ochorenia': ('onkologické ochorenia', 'komorbidita', 'binárna'),
         'Imunosupresia': ('imunosupresia', 'komorbidita', 'binárna'), 'Vakcinácia': ('očkovanie proti COVID-19', 'očkovanie', 'binárna'),
         'kasel': ('kašeľ', 'príznak', 'binárna'), 'hnacka': ('hnačka', 'príznak', 'binárna'),
         'triaska_zimnica': ('triaška / zimnica', 'príznak', 'binárna'), 'zvracanie': ('zvracanie', 'príznak', 'binárna'),
         'nechutenstvo': ('nechutenstvo', 'príznak', 'binárna'),
         'sat_bez_o2_prva': ('prvá zaznamenaná saturácia O₂ bez kyslíka (%)', 'vitálna funkcia', 'číselná'),
         'sat_s_o2_prva': ('prvá zaznamenaná saturácia O₂ s kyslíkom (%)', 'vitálna funkcia', 'číselná'),
         'temp_first': ('prvá zaznamenaná telesná teplota (°C)', 'vitálna funkcia', 'číselná')}
na1 = w1[feats].isna().mean() * 100
dropped = {c for c in feats if na1[c] > 50}
rows = []
for i, c in enumerate(feats, 1):
    if c in OTHER:
        opis, sk, typ = OTHER[c]; orig = 'Pohlavie' if c == 'muz' else c
    else:
        k = c.replace('_prva_hodnota', ''); opis = f'prvá zaznamenaná hodnota: {LAB[k]}'; orig = c
        sk = 'laboratórny parameter'; typ = 'číselná'
    rows.append(dict(c=i, premenna_v_subore=orig, opis=opis, skupina=sk, typ=typ,
                     chyba_vlna1_pct=round(na1[c], 1), chyba_vsetci_pct=round(df[c].isna().mean() * 100, 1),
                     v_modeli='vyradená filtrom (> 50 % chýbajúcich v tréningovej časti)' if c in dropped else 'áno'))
U = pd.DataFrame(rows); U.to_csv(OUT_ + 'atributy_pouzite.csv', index=False)

used_orig = set(U.premenna_v_subore)
def classify(c):
    if c in used_orig: return 'použitý vstup', 'údaj z prijatia / prvá zaznamenaná hodnota'
    if c == 'Stav prepustenia': return 'cieľová premenná', 'zdroj cieľovej premennej'
    if c == 'vlna': return 'časové rozdelenie', 'určuje tréningovú a testovacie vlny'
    if c == 'Typ vakcíny': return 'nepoužitý', 'podrobnosť k očkovaniu, väčšinou prázdna; použitá binárna Vakcinácia'
    if '_prva_ref_' in c: return 'nepoužitý', 'príznak prekročenia referenčnej hranice odvodený z prvej hodnoty – redundantný'
    if '_posledna' in c or c.endswith('_median') or c.endswith('_med') or re.search(r'_(min|max)$', c) \
            or c in ('febrilita', 'hyperpyrexia'):
        return 'nepoužitý – únik informácií', 'vzniká počas hospitalizácie (posledná hodnota, medián, minimum/maximum, febrilita)'
    if c.endswith('_prva'): return 'nepoužitý', 'odvodený index vypočítaný z prvých hodnôt – redundantný'
    return 'nepoužitý', 'iné'
A = []
for c in raw.columns:
    st, why = classify(c)
    A.append(dict(premenna=c, zaradenie=st, dovod=why, chyba_vsetci_pct=round(raw[c].isna().mean() * 100, 1)))
A = pd.DataFrame(A); A.to_csv(OUT_ + 'atributy_vsetky.csv', index=False)
S = A.groupby(['zaradenie', 'dovod']).size().reset_index(name='pocet')
with pd.ExcelWriter(OUT_ + 'Zoznam_atributov.xlsx') as xw:
    S.to_excel(xw, 'Prehľad', index=False); U.to_excel(xw, 'Použité vstupy (65)', index=False)
    A.to_excel(xw, 'Všetky atribúty (579)', index=False)
print(S.to_string(index=False)); print(len(U), len(A), len(dropped))
