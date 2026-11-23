#!/usr/bin/env bash
# Spustí celú praktickú časť od dát po výsledky.
# Štruktúra: ../data/ (vstupné dáta), ../code/ (skripty), ../vysledky/ (tabuľky a obrázky).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../data ../vysledky
if [ ! -f ../data/df.pkl ]; then
  python -c "import pandas as pd; pd.read_excel('../data/DF_clean 1.xlsx').to_pickle('../data/df.pkl')"
fi

python 01_pocty_skupin.py                       # počty pacientov a udalostí podľa vĺn a skupín

# hlavná analýza (exclude) a citlivostné analýzy A (nonevent) a B (event) s preloženými pacientmi
for mode in exclude nonevent event; do
  python 02_trening_vlna1.py $mode              # vnorená 5 × 5 CV na vlne 1 + finálny model (cca 10 – 20 min)
  python 03_casova_validacia_H1.py $mode        # časová validácia na vlnách 2 – 4, test H1
  python 04_skupiny_H2_H3.py $mode              # skupinové CITL, H2 a H3 s Holmovou korekciou
  python 05_rekalibracia_H4.py $mode            # H4 – globálna rekalibrácia 2 → 3 a 3 → 4
  python 06_porovnanie_modelov.py $mode         # párové porovnanie XGBoost − logistická regresia
done
# citlivostná analýza C: udalosť = iba úmrtie
python 02_trening_vlna1.py death
python 03_casova_validacia_H1.py death

python 07_atributy.py                           # zoznam atribútov (Zoznam_atributov.xlsx)
python 08_vyber_kohorty.py                      # vylúčení pacienti, SMD, zloženie udalosti
python 09_prelozeni.py                          # opis preložených pacientov a ich predikované riziko
python 10_reprodukovatelnost.py                 # filter chýbajúcich hodnôt vnútri validácie, verzie, MD5
python 11_opisne_tabulky.py                     # opisné tabuľky podľa vĺn a skupín

python 12_obrazky_data_modely.py                # obrázky: vstupy, dôležitosť premenných, vlna 1, rekalibračné krivky
python 13_obrazky_vysledky.py                   # obrázky: postup, kohorta, posun, metriky, H1, H2, H4
python 14_obrazky_skupiny.py                    # obrázky: chýbajúce hodnoty vo vlnách, vylúčení, skupiny, H3
python 15_obrazky_doplnkove.py                  # obrázky: porovnanie modelov, skupinové CITL po rekalibrácii
python 16_obrazok_chybajuce.py                  # obrázok: chýbajúce hodnoty vo vlne 1
echo "Hotovo. Výsledky sú v ../vysledky"
