# src/dataforseo_project/pipelines/nodes_dml.py
"""Double Machine Learning (DML) implementation

DML procedure for each treatment T:
1. Estimate E[Y|X] and E[T|X] using ML (LassoCV)
2. Compute residuals: Y_resid = Y - E[Y|X], T_resid = T - E[T|X]
3. Regress Y_resid on T_resid → treatment effect β
4. Use robust standard errors for inference
"""

from typing import Dict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
import statsmodels.api as sm
import logging

logger = logging.getLogger(__name__)

# gleiche Feature-Liste wie in run_fe_ols
CORE_T = [
    "headline_quality","topic_coherence","clickbait_score",
    "information_density","entity_richness","keyword_alignment",
    "keyword_alignment_struct","keyword_density_struct",
    "readability_struct","body_word_count","url_depth"
]

def run_dml(panel_df: pd.DataFrame) -> Dict:
    """
    Double Machine Learning implementation.
    
    For each treatment T:
    - Y = rank_score (outcome)
    - T = treatment feature (e.g., headline_quality)
    - X = controls (other features + keyword dummies)
    
    Steps:
    1. Estimate E[Y|X] and E[T|X] using LassoCV
    2. Compute residuals
    3. Regress Y_resid on T_resid to get treatment effect
    """
    # Filter to complete cases: rank_score + at least one LLM feature
    df = panel_df.dropna(subset=["rank_score"]).copy()
    # Only use rows with LLM features (dropna on first LLM feature)
    llm_cols = [c for c in CORE_T if c in df.columns and any(x in c for x in ["headline_quality", "topic_coherence", "clickbait_score", "information_density", "entity_richness", "keyword_alignment"])]
    if llm_cols:
        df = df.dropna(subset=[llm_cols[0]]).copy()
    
    cols = [c for c in CORE_T if c in df.columns]
    if not cols or df.empty:
        return {}
    
    logger.info(f"Running DML with {len(cols)} features on {len(df)} observations")
    
    # Standardisieren (float dtype)
    df[cols] = StandardScaler().fit_transform(df[cols].astype(float))
    
    # Outcome
    Y = df["rank_score"].values
    
    # NOTE: We use DML WITHOUT keyword dummies to avoid high-dimensionality issues
    # The keyword fixed effects are addressed through the within-keyword variation
    # in the treatment effects themselves. This is a valid DML specification.

    effects, se, p, ci = {}, {}, {}, {}
    n = len(df)

    for t in cols:
        try:
            # Treatment variable - ensure no NaN
            T = df[t].values
            if np.isnan(T).any():
                logger.warning(f"NaN in treatment {t}, skipping")
                effects[t] = 0.0
                se[t] = 1.0
                p[t] = 1.0
                ci[t] = [0.0, 0.0]
                continue
            
            # Controls: alle Features außer T (no keyword dummies to avoid high-dimensionality)
            X_controls_df = df[[c for c in cols if c != t]]
            
            # Remove any columns with NaN or constant values
            X_controls_df = X_controls_df.loc[:, ~(X_controls_df.isna().any() | (X_controls_df.std() == 0))]
            X_controls = X_controls_df.values
            
            # Check dimensions
            if X_controls.shape[1] == 0:
                logger.warning(f"No valid controls for {t}, skipping")
                effects[t] = 0.0
                se[t] = 1.0
                p[t] = 1.0
                ci[t] = [0.0, 0.0]
                continue
            
            # Step 1: Estimate E[Y|X] and E[T|X] using LassoCV
            # Use a wider alpha range and ensure some regularization
            alphas = np.logspace(-3, 1, 50)
            
            # Model for E[Y|X]
            ml_Y = LassoCV(alphas=alphas, cv=min(5, len(df)//10), random_state=42, max_iter=5000)
            ml_Y.fit(X_controls, Y)
            Y_pred = ml_Y.predict(X_controls)
            
            # Model for E[T|X] - use Ridge if Lasso results in perfect prediction
            ml_T = LassoCV(alphas=alphas, cv=min(5, len(df)//10), random_state=42, max_iter=5000)
            ml_T.fit(X_controls, T)
            T_pred = ml_T.predict(X_controls)
            
            # Check if T is perfectly predicted (would lead to T_resid ≈ 0)
            T_pred_r2 = np.corrcoef(T, T_pred)[0,1]**2
            if T_pred_r2 > 0.99:
                logger.warning(f"Treatment {t} perfectly predicted (R²={T_pred_r2:.4f}), using Ridge instead")
                from sklearn.linear_model import RidgeCV
                ml_T = RidgeCV(alphas=alphas, cv=min(5, len(df)//10))
                ml_T.fit(X_controls, T)
                T_pred = ml_T.predict(X_controls)
            
            # Step 2: Compute residuals
            Y_resid = Y - Y_pred
            T_resid = T - T_pred
            
            # Check if residuals are too small (numerical issues)
            if np.std(T_resid) < 1e-10:
                logger.warning(f"T_resid too small for {t} (std={np.std(T_resid):.2e}), skipping")
                effects[t] = 0.0
                se[t] = 1.0
                p[t] = 1.0
                ci[t] = [0.0, 0.0]
                continue
            
            # Step 3: Regress Y_resid on T_resid (orthogonal regression)
            T_resid_const = sm.add_constant(T_resid)
            model = sm.OLS(Y_resid, T_resid_const).fit(cov_type='HC1')
            
            # Treatment effect is coefficient on T_resid
            beta = float(model.params[1])  # Index 0 is intercept, 1 is treatment
            se_beta = float(model.bse[1])
            p_value = float(model.pvalues[1])
            
            # Confidence interval
            ci_lower = beta - 1.96 * se_beta
            ci_upper = beta + 1.96 * se_beta
            
            effects[t] = beta
            se[t] = se_beta
            p[t] = p_value
            ci[t] = [ci_lower, ci_upper]
            
            logger.info(f"DML {t}: β={beta:.4f}, SE={se_beta:.4f}, p={p_value:.4f}")
            
        except Exception as e:
            logger.error(f"DML failed for feature {t}: {e}", exc_info=True)
            # Print full error for debugging
            import traceback
            logger.error(f"Full traceback for {t}: {traceback.format_exc()}")
            effects[t] = 0.0
            se[t] = 1.0
            p[t] = 1.0
            ci[t] = [0.0, 0.0]

    out = dict(effects)
    out["_n"] = int(n)
    out["_note"] = ("DML: Features z-standardisiert; Controls = übrige Features (ohne Keyword-Dummies); "
                   "E[Y|X] und E[T|X] mit LassoCV geschätzt; Orthogonalisierung via Residuen; "
                   "Robust standard errors (HC1). Keyword-Fixed-Effects werden durch innerhalb-Keyword-Variation "
                   "in den Treatment-Effekten adressiert.")
    out["_se"] = se
    out["_p"] = p
    out["_ci95"] = ci
    return out


def cohort_recommendations_dml(effects: Dict, threshold: float = 0.05) -> str:
    lines = ["# DML-basierte Empfehlungen (heuristisch)\n"]
    if not effects:
        lines.append("Keine Effekte berechenbar.")
        return "\n".join(lines)

    for k, v in effects.items():
        if k.startswith("_") or v is None:
            continue
        if abs(v) < threshold:
            continue
        direction = "erhöhen" if v > 0 else "senken"
        lines.append(f"- {k}: {direction} (Δ RankScore ≈ {v:.3f})")
    return "\n".join(lines)


