"""Podiel chýbajúcich hodnôt vo vlne 1 a hranica filtra 50 %."""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from common import load, feature_list, OUT_, DATA
O = OUT_
BLUE, ORANGE, INK, MUTED, GRID = '#1f5fa8', '#b5532a', '#222222', '#666666', '#e5e5e5'
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.color': MUTED, 'ytick.color': INK})
# chýbajúce hodnoty (vlna 1)
df = load(); w = df[df.vlna == 1]; f = feature_list(df)
m = (w[f].isna().mean() * 100).sort_values(ascending=False).head(20)
names = {'S-Chol': 'cholesterol', 'S-IgA': 'imunoglobulín A', 'S-Ig M': 'imunoglobulín M', 'S-IgG': 'imunoglobulín G',
         'S-AMS': 'amyláza', 'S-LD': 'laktátdehydrogenáza', 'sat_s_o2_prva': 'saturácia s kyslíkom',
         'NK': 'NK bunky', 'CD19+': 'CD19+ lymfocyty', 'sat_bez_o2_prva': 'saturácia bez kyslíka',
         'S-KM': 'kyselina močová', 'S-Bil-D': 'priamy bilirubín', 'S-Alb': 'albumín', 'S-CB': 'celková bielkovina',
         'temp_first': 'telesná teplota', 'NE/LY(NLR)': 'pomer neutrofilov a lymfocytov', 'Fib': 'fibrinogén',
         'CD4+/CD8+': 'pomer CD4+/CD8+', 'CD8+': 'CD8+ lymfocyty', 'CD4+': 'CD4+ lymfocyty', 'S-VITD': 'vitamín D',
         'CD3+': 'CD3+ lymfocyty'}
lab = [names.get(k.replace('_prva_hodnota', ''), names.get(k, k)) for k in m.index]
fig, ax = plt.subplots(figsize=(9, 5))
cols = [ORANGE if v > 50 else BLUE for v in m.values]
ax.barh(range(len(m))[::-1], m.values, color=cols, height=0.7)
for i, v in enumerate(m.values):
    ax.text(v + 1, len(m) - 1 - i, f'{v:.1f} %'.replace('.', ','), va='center', fontsize=8.5, color=INK)
ax.set_yticks(range(len(m))[::-1]); ax.set_yticklabels(lab, fontsize=9)
ax.axvline(50, ls='--', color='#555', lw=1); ax.text(51, len(m) - 0.4, 'hranica 50 %', fontsize=8.5, color=MUTED)
ax.set_xlim(0, 110); ax.set_xlabel('Podiel chýbajúcich hodnôt vo vlne 1 (%)'); ax.xaxis.grid(True, color=GRID); ax.set_axisbelow(True)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label='vyradené (viac ako 50 %)'), Patch(color=BLUE, label='ponechané')],
          frameon=False, loc='lower right')
plt.tight_layout(); plt.savefig(O + 'chybajuce_hodnoty.png', dpi=200); plt.close()
