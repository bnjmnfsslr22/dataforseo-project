# Kausaler Einfluss von LLM-Content-Faktoren auf Google-Rankings

Code zu meiner Bachelorarbeit (B.Sc. Informatik, FH Aachen, 2025/2026):
„Kausaler Einfluss von LLM-Content-Faktoren auf Google-Rankings: Kohortenanalyse mit FE und DML“.

**English summary:** A Kedro pipeline that collects Google search results for about 100 German keywords via the DataForSEO API, downloads the ranking articles, has an LLM (gpt-5-mini) rate each article on six content features, and estimates how these features relate to ranking position using keyword fixed-effects OLS and a Double-ML-style partialling-out estimator.

## Was die Pipeline macht

1. **Keywords erzeugen:** rund 100 Keywords aus vier Themenfeldern (`nodes_keyword_generation.py`)
2. **Suchergebnisse abrufen:** Google-SERP je Keyword über die DataForSEO-API, Deutschland, Desktop (`nodes_cohort.py`)
3. **Artikel laden:** Text der gefundenen Seiten mit `requests` und BeautifulSoup (`nodes_content_cohort.py`)
4. **LLM-Bewertung:** `gpt-5-mini` bewertet jeden Artikel nach sechs Merkmalen auf einer Skala von 0 bis 1. Der Prompt enthält eine Rubrik mit Skalenankern und verlangt striktes JSON (`nodes_llm_cohort.py`)
5. **Strukturmerkmale ohne LLM:** Keyword-Überdeckung, Keyword-Dichte, Lesbarkeit, Wortzahl, URL-Tiefe (`nodes_struct_cohort.py`)
6. **Panel bauen:** eine Zeile je Keyword und URL, Zielgröße `rank_score` = negativer Rang (`nodes_panel.py`)
7. **Effekte schätzen:** Fixed-Effects-OLS und Double Machine Learning (`nodes_causal_panel.py`, `nodes_dml.py`)
8. **Berichte und Grafiken** (`nodes_scientific_reporting.py`, `nodes_visualizations.py`)

Die sechs LLM-Merkmale: `headline_quality`, `topic_coherence`, `clickbait_score`, `information_density`, `entity_richness`, `keyword_alignment`.

Der LLM-Schritt speichert Zwischenstände, damit ein abgebrochener Lauf fortgesetzt werden kann, und zählt Tokens und Kosten mit. In `src/dataforseo_project/utils/` liegen außerdem die Keyword-Erzeugung für die Seed-Pipeline und weitere Hilfsmodule (Caching, Kostenrechner, Datenqualität), die noch nicht in die Pipeline eingebunden sind.

## Datenstand

| Schritt | Anzahl |
| --- | --- |
| Keywords | 101, davon 91 mit Suchergebnissen |
| Suchergebnisse | 1.387 |
| Artikel erfolgreich geladen | 1.290 |
| Artikel mit LLM-Bewertung (Stichprobe der Analyse) | 1.285 |

Quelle: `docs/BACHELORARBEIT_DATEN_SUMMARY.md` (Lauf vom 04.11.2025). Die Daten selbst liegen nicht im Repository.

## Methoden

- **Fixed-Effects-OLS:** `rank_score ~ Merkmale + C(keyword)`, Merkmale z-standardisiert, robuste Standardfehler (HC1). Die Keyword-Fixed-Effects sorgen dafür, dass nur Artikel zum selben Keyword miteinander verglichen werden.
- **Double Machine Learning:** Für jedes Merkmal werden Zielgröße und Merkmal mit `LassoCV` aus den übrigen Merkmalen vorhergesagt. Der Effekt ist die Regression der beiden Residuen aufeinander (HC1).

## Stand und Grenzen

Was dieses Repository nicht leistet, steht hier bewusst offen:

- **Beobachtungsdaten:** Die Schätzungen zeigen Zusammenhänge unter Kontrolle der übrigen Merkmale. Gesicherte Kausalität lässt sich daraus nicht ableiten. Das Fixed-Effects-Modell erklärt nur einen kleinen Teil der Ranking-Varianz (R² rund 0,05).
- **DML ohne Cross-Fitting:** Die Residuen werden auf denselben Daten gebildet, auf denen die Lasso-Modelle trainiert sind. Keyword-Dummies sind im DML-Schritt nicht enthalten.
- **LLM-Validierung ist simuliert:** `nodes_validation.py` berechnet Test-Retest- und Inter-Prompt-Reliabilität mit künstlichem Zufallsrauschen. Das ist ein Platzhalter und keine echte Messung. Eine echte Prüfung bewertet dieselben Artikel mehrfach und vergleicht mit menschlichen Bewertungen.
- **Berichtstexte:** Der automatisch erzeugte Bericht enthält Textbausteine zu Bootstrap, Placebo-Tests und Cross-Fitting. Diese Verfahren sind nicht implementiert.

Nächste Schritte: echte Wiederholungsmessung der LLM-Bewertungen, Cross-Fitting im DML-Schritt, Vergleich mit menschlichen Bewertungen.

## Starten

```bash
pip install -r requirements.txt
cp env.example .env        # Zugangsdaten für DataForSEO und OpenAI eintragen
kedro run                  # ganze Pipeline
kedro run --pipeline seed  # mit vorhandenen Keywords statt neu erzeugten
```

Die Pipeline ruft kostenpflichtige APIs auf. `utils/cost_calculator.py` schätzt die Kosten vorab. Parameter wie Modell, Anzahl der Keywords und Textlängen stehen in `conf/base/parameters.yml`.

## Aufbau

```
conf/base/                    Parameter und Datenkatalog
docs/                         Datenfluss und Datenzusammenfassung
src/dataforseo_project/
  pipelines/                  Pipeline-Schritte (siehe oben)
  utils/                      Keyword-Erzeugung, weitere Hilfsmodule (teils noch nicht eingebunden)
```

Technik: Python, Kedro, pandas, NumPy, scikit-learn, statsmodels, OpenAI-API, DataForSEO-API.
