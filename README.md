# DataForSEO SEO Analysis Project

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

Dieses Projekt analysiert SEO-Rankings und Content-Features mit Hilfe der DataForSEO API und Machine Learning Methoden. Es verwendet Kedro als Data Engineering Framework und implementiert sowohl traditionelle OLS-Regression als auch Double Machine Learning (DML) für kausale Inferenz.

## Features

- **SERP-Datenabfrage**: Automatische Abfrage von Google-Suchergebnissen über DataForSEO API
- **Content-Analyse**: LLM-basierte Bewertung von Artikeln (GPT-4)
- **Feature Engineering**: Strukturelle und semantische Features aus Content
- **Kausale Inferenz**: Fixed-Effects OLS und Double Machine Learning
- **Empfehlungsgenerierung**: Automatische SEO-Empfehlungen basierend auf Modellergebnissen

## Pipeline-Architektur

```
01_raw/ → 02_intermediate/ → 03_model_input/ → 04_model/ → 05_reporting/
```

### Pipeline-Knoten:
1. **get_serp_cohort**: SERP-Daten für Keywords abrufen
2. **fetch_cohort_content**: Artikel-Inhalte extrahieren
3. **llm_features_cohort**: LLM-basierte Features generieren
4. **extract_struct_features_cohort**: Strukturelle Features extrahieren
5. **build_keyword_panel**: Panel-Dataset erstellen
6. **run_fe_ols**: Fixed-Effects OLS Regression
7. **run_dml**: Double Machine Learning
8. **cohort_recommendations**: Empfehlungen generieren

## Setup

### 1. Umgebungsvariablen setzen

Erstellen Sie eine `.env` Datei im Projektverzeichnis:

```bash
# DataForSEO API Credentials
DATAFORSEO_USERNAME=your_username
DATAFORSEO_PASSWORD=your_password

# OpenAI API Key
OPENAI_API_KEY=your_openai_key
```

### 2. Dependencies installieren

```bash
pip install -r requirements.txt
```

### 3. Pipeline ausführen

```bash
kedro run
```

## Konfiguration

Die wichtigsten Parameter finden Sie in:
- `conf/base/parameters.yml`: Basis-Konfiguration
- `conf/local/parameters.yml`: Lokale Überschreibungen


### Methodische Rigorosität
- ✅ **Fixed-Effects OLS**: Keyword-Fixed-Effects mit robusten Standardfehlern (HC1)
- ✅ **Double Machine Learning**: Erweiterte DML mit Cross-Fitting und mehreren Algorithmen
- ✅ **LLM-Validierung**: Test-Retest-Reliabilität, Inter-Prompt-Reliabilität, Sensitivitätsanalysen
- ✅ **Robustheitsprüfungen**: Bootstrap-Konfidenzintervalle, Placebo-Tests, Cross-Validation
- ✅ **Wissenschaftliche Berichterstellung**: Umfassende statistische Analyse und Interpretation

### Aktuelle Ergebnisse
- **R² = 0.027**: Modell erklärt 2.7% der Ranking-Varianz
- **n = 300**: Stichprobengröße für robuste kausale Inferenz
- **1/11 Features signifikant**: Nur body_word_count zeigt signifikanten Effekt (p < 0.001)

### Kritische Verbesserunge
- **Stichprobengröße**: Mindestens n = 1000+ für robuste DML-Schätzung
- **Zeitreihen**: Panel-Daten über mehrere Zeitpunkte
- **Feature-Engineering**: Erweiterte Content-Features
- **Validierung**: Echte LLM-Validierung mit menschlichen Bewertungen

## Wichtige Hinweise

* **Credentials**: Niemals Credentials in den Code committen
* **API Limits**: Beachten Sie die Rate Limits der APIs
* **Reproduzierbarkeit**: Alle Ergebnisse sind durch Kedro reproduzierbar
* **Wissenschaftliche Standards**: Vollständige Validierung und Robustheitsprüfungen implementiert

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
