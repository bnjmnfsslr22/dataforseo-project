import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

CORE_T = [
    "headline_quality","topic_coherence","clickbait_score",
    "information_density","entity_richness","keyword_alignment",
    "keyword_alignment_struct","keyword_density_struct","readability_struct",
    "body_word_count","url_depth"
]

def run_fe_ols(panel_df: pd.DataFrame) -> dict:
    """
    Keyword‑Fixed‑Effects OLS, numerische Features z‑standardisiert.
    
    Uses only complete cases (rank_score + LLM features) for fair comparison with DML.
    """
    # Filter to complete cases: rank_score + at least one LLM feature
    df = panel_df.dropna(subset=["rank_score"]).copy()
    # Only use rows with LLM features (dropna on first LLM feature for consistency with DML)
    llm_cols = [c for c in CORE_T if c in df.columns and any(x in c for x in ["headline_quality", "topic_coherence", "clickbait_score", "information_density", "entity_richness", "keyword_alignment"])]
    if llm_cols:
        df = df.dropna(subset=[llm_cols[0]]).copy()
    
    cols = [c for c in CORE_T if c in df.columns]
    if not cols or df.empty:
        return {}
    # Standardisieren
    df[cols] = StandardScaler().fit_transform(df[cols])
    formula = "rank_score ~ " + " + ".join(cols) + " + C(keyword)"
    model = smf.ols(formula, data=df).fit(cov_type="HC1")

    eff = {c: float(model.params[c]) for c in cols}
    eff_se = {c: float(model.bse[c]) for c in cols}
    eff_p  = {c: float(model.pvalues[c]) for c in cols}
    # Meta
    eff["_n"]  = int(len(df))
    eff["_r2"] = float(model.rsquared)
    eff["_se"] = eff_se        # nested dict
    eff["_p"]  = eff_p
    return eff
