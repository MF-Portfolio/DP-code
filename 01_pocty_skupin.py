"""problémy s dátami, počty podľa vĺn a skupín."""
import pandas as pd
from common import load, COMORB, OUT_, DATA
pd.set_option('display.width', 250)
raw = pd.read_pickle(DATA)
print(pd.crosstab(raw.vlna, raw['Stav prepustenia'], margins=True).to_string())
df = load('exclude')
rows = []
groups = {'Spolu': pd.Series('všetci', index=df.index),
          'Pohlavie': df['Pohlavie'], 'Vek': df['vek_kat'].astype(str)}
for c in ['Hypertenzia', 'Diabetes mellitus', 'Kardiovaskulárne ochorenia',
          'Chronické respiračné ochorenia', 'Onkologické ochorenia',
          'Imunosupresia', 'Pečeňové ochorenia']:
    groups[c] = df[c].map({True: 'áno', False: 'nie'})
for gname, g in groups.items():
    for lvl in sorted(g.unique()):
        m = g == lvl
        r = {'atribút': gname, 'skupina': lvl}
        for w in [1, 2, 3, 4]:
            s = df.loc[m & (df.vlna == w), 'y']
            r[f'v{w}_n'] = len(s); r[f'v{w}_ud'] = int(s.sum())
        mn = min(r[f'v{w}_ud'] for w in [2, 3, 4])
        mn_non = min(r[f'v{w}_n'] - r[f'v{w}_ud'] for w in [2, 3, 4])
        r['min_ud_v2-4'] = mn
        r['posúdenie'] = ('hodnotenie' if min(mn, mn_non) >= 20 else
                          'opatrne (10–19 udalostí)' if mn >= 10 else 'iba opisne')
        rows.append(r)
t = pd.DataFrame(rows)
print(t.to_string(index=False))
t.to_csv(OUT_ + 'pocty_skupiny.csv', index=False)
