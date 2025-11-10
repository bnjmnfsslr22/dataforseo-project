# 📚 Automatische Dokumentation - Übersicht

## Was wird automatisch dokumentiert?

### ✅ Nach LLM-Verarbeitung (`run_llm_all_articles.py`):

1. **LLM-Verarbeitungs-Details** (`data/04_model/llm_processing_complete.json`):
   - Timestamp
   - Modell (gpt-5-mini)
   - Vollständiger Prompt (System + User)
   - Anzahl verarbeiteter Artikel
   - Kosten (USD)
   - Feature-Namen

2. **Pipeline-Dokumentation** (`docs/PIPELINE_DATENFLUSS.md`):
   - Automatisches Update mit neuen Zahlen
   - Abdeckungsraten
   - Kosten-Informationen

### ✅ Nach Analyse-Neuberechnung (`rerun_all_analyses_after_llm.py`):

1. **Panel-Datensatz** (`data/03_model_input/cohort_panel.parquet`):
   - Neu erstellt mit allen LLM-Features

2. **Statistische Ergebnisse**:
   - `data/04_model/cohort_effects.json` - OLS Ergebnisse
   - `data/04_model/cohort_effects_dml.json` - DML Ergebnisse
   - `data/04_model/heterogeneity_analysis.json` - Heterogenitätsanalysen
   - `data/04_model/diagnostic_tests.json` - Diagnostische Tests
   - `data/04_model/robustness_check_results.json` - Robustheitsprüfungen

3. **Empfehlungen** (`data/05_reporting/cohort_recommendations.md`):
   - SEO-Empfehlungen basierend auf neuen Ergebnissen

4. **Zusammenfassung** (`docs/BACHELORARBEIT_DATEN_SUMMARY.md`):
   - Vollständige Übersicht aller Zahlen
   - Fertig für Copy-Paste

## Workflow

### Option 1: Schritt für Schritt
```bash
# 1. LLM-Verarbeitung
python run_llm_all_articles.py

# 2. Analysen neu berechnen
python rerun_all_analyses_after_llm.py

# 3. Zusammenfassung generieren
python generate_bachelorarbeit_summary.py
```

### Option 2: Alles automatisch
```bash
python complete_pipeline_rerun.py
```

## Dokumentierte Informationen


- ✅ Prompt (vollständig, wissenschaftlich)
- ✅ Modell (gpt-5-mini)
- ✅ Stichprobengröße (n=???)
- ✅ Scraping-Erfolgsrate (93%)
- ✅ LLM-Feature-Abdeckung (100% nach Verarbeitung)
- ✅ Kosten (USD)
- ✅ R²-Werte
- ✅ Signifikante Effekte
- ✅ Timestamps aller Schritte

Alles ist **automatisch dokumentiert** 🎓

