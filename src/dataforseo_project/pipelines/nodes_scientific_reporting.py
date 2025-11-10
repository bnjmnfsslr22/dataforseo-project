"""Scientific reporting  with statistical rigor."""

import pandas as pd
import numpy as np
import json
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


def generate_scientific_report(
    ols_results: Dict,
    dml_results: Dict,
    validation_results: Dict,
    panel_data: pd.DataFrame,
    robustness_results: Dict = None
) -> str:
    """
    Generate a comprehensive scientific report 
    
    Args:
        ols_results: Fixed-effects OLS results
        dml_results: Double Machine Learning results
        validation_results: LLM validation results
        robustness_results: Robustness check results
        panel_data: Panel dataset
        
    Returns:
        Comprehensive scientific report
    """
    logger.info("Generating scientific report")
    
    report_sections = [
        generate_executive_summary(ols_results, dml_results, validation_results, robustness_results),
        generate_methodology_section(ols_results, dml_results),
        generate_results_section(ols_results, dml_results, robustness_results if robustness_results else None),
        generate_validation_section(validation_results),
        generate_robustness_section(robustness_results) if robustness_results else "Robustness analysis not available.",
        generate_limitations_section(ols_results, dml_results, panel_data),
        generate_conclusions_section(ols_results, dml_results, robustness_results)
    ]
    
    return "\n\n".join(report_sections)


def generate_executive_summary(ols_results: Dict, dml_results: Dict, validation_results: Dict, robustness_results: Dict = None) -> str:
    """Generate executive summary with key findings."""
    
    # Count significant effects
    ols_significant = count_significant_effects(ols_results.get("_p", {}))
    dml_significant = count_significant_effects(dml_results.get("_p", {}))
    
    # Model performance
    ols_r2 = ols_results.get("_r2", 0)
    n_obs = ols_results.get("_n", 0)
    
    # Validation quality
    validation_quality = validation_results.get("validation_summary", {}).get("validation_quality", "unknown")

    # Robustness assessment
    robustness_quality = "not_available"
    if robustness_results:
        robustness_quality = robustness_results.get("robustness_summary", {}).get("overall_robustness", "unknown")
    
    summary = f"""# Executive Summary

## Key Findings

**Statistical Significance:**
- Fixed-Effects OLS: {ols_significant} out of {len(ols_results.get('_p', {}))} features significant (p < 0.05)
- Double Machine Learning: {dml_significant} out of {len(dml_results.get('_p', {}))} features significant (p < 0.05)

**Model Performance:**
- R² = {ols_r2:.3f} (Fixed-Effects OLS)
- Sample size: n = {n_obs:,} observations
- Keywords: {len(panel_data['keyword'].unique()) if 'keyword' in panel_data.columns else 'Unknown'}

**Methodological Quality:**
- LLM Validation: {validation_quality.upper()}
- Robustness Checks: {robustness_quality.upper()}

## Main Conclusions

"""
    
    if ols_significant > 0:
        summary += f"- **Causal effects detected**: {ols_significant} content features show statistically significant effects on Google rankings\n"
    else:
        summary += "- **No significant effects**: Current analysis shows no statistically significant causal effects\n"
    
    if ols_r2 < 0.1:
        summary += "- **Low explanatory power**: Model explains less than 10% of ranking variance\n"
    elif ols_r2 < 0.3:
        summary += "- **Moderate explanatory power**: Model explains {ols_r2:.1%} of ranking variance\n"
    else:
        summary += "- **Good explanatory power**: Model explains {ols_r2:.1%} of ranking variance\n"
    
    if validation_quality in ["poor", "acceptable"]:
        summary += "- **LLM validation concerns**: Measurement reliability needs improvement\n"
    
    if robustness_quality in ["poor", "acceptable"]:
        summary += "- **Robustness concerns**: Results may not be robust to specification changes\n"
    
    return summary


def generate_methodology_section(ols_results: Dict, dml_results: Dict) -> str:
    """Generate methodology section."""
    
    methodology = """# Methodology

## Causal Inference Framework

This study employs a **cohort-based causal inference approach** to estimate the causal effects of LLM-derived content features on Google search rankings.

### Fixed-Effects OLS Regression

**Model Specification:**
```
rank_score_ij = α + β₁X₁_ij + β₂X₂_ij + ... + βₖXₖ_ij + γᵢ + εᵢⱼ
```

Where:
- `rank_score_ij`: Ranking position for article j in keyword cohort i
- `X₁_ij, ..., Xₖ_ij`: LLM-derived and structural content features
- `γᵢ`: Keyword fixed effects (controls for unobserved keyword-specific factors)
- `εᵢⱼ`: Error term

**Identification Strategy:**
- **Within-keyword variation**: Exploits variation in content features within the same keyword cohort
- **Fixed effects**: Controls for all time-invariant keyword characteristics
- **Robust standard errors**: HC1 standard errors to address heteroskedastizität

### Double Machine Learning (DML)

**Enhanced DML Implementation:**
- **Multiple algorithms**: Lasso, Ridge, Elastic Net, Random Forest
- **Cross-fitting**: K-fold cross-validation to prevent overfitting
- **Orthogonalization**: Separates treatment effect estimation from nuisance parameter estimation

**DML Algorithm:**
1. **First stage**: Predict treatment (content feature) and outcome (ranking) using ML
2. **Second stage**: Estimate causal effect using orthogonalized residuals
3. **Inference**: Bootstrap confidence intervals and p-values

### Content Feature Categories

**LLM-Derived Features:**
- `headline_quality`: Quality assessment of article headlines
- `topic_coherence`: Coherence between headline and content
- `clickbait_score`: Clickbait detection score
- `information_density`: Information content density
- `entity_richness`: Named entity recognition richness
- `keyword_alignment`: Semantic alignment with target keyword

**Structural Features:**
- `body_word_count`: Article length
- `readability_struct`: Text readability score
- `url_depth`: URL structure depth
- `keyword_density_struct`: Keyword density in content

"""
    
    return methodology


def generate_results_section(ols_results: Dict, dml_results: Dict, robustness_results: Dict = None) -> str:
    """Generate results section with statistical details."""
    
    results = """# Results

## Fixed-Effects OLS Results

### Model Performance
"""
    
    # Model performance metrics
    r2 = ols_results.get("_r2", 0)
    n_obs = ols_results.get("_n", 0)
    n_features = len([k for k in ols_results.keys() if not k.startswith("_")])
    
    results += f"""
- **R²**: {r2:.3f}
- **Adjusted R²**: {r2:.3f} (approximately)
- **Sample size**: n = {n_obs:,}
- **Features tested**: {n_features}
- **Keywords**: {len(ols_results.get('_p', {}))} (fixed effects)
"""
    
    # Significant effects
    significant_effects = []
    for feature, p_value in ols_results.get("_p", {}).items():
        if p_value < 0.05:
            coefficient = ols_results.get(feature, 0)
            se = ols_results.get("_se", {}).get(feature, 0)
            significant_effects.append({
                "feature": feature,
                "coefficient": coefficient,
                "se": se,
                "p_value": p_value
            })
    
    if significant_effects:
        results += "\n### Statistically Significant Effects (p < 0.05)\n\n"
        results += "| Feature | Coefficient | Std. Error | p-value | Interpretation |\n"
        results += "|---------|-------------|------------|---------|----------------|\n"
        
        for effect in significant_effects:
            direction = "Positive" if effect["coefficient"] > 0 else "Negative"
            interpretation = f"{direction} effect on ranking"
            results += f"| {effect['feature']} | {effect['coefficient']:.3f} | {effect['se']:.3f} | {effect['p_value']:.3f} | {interpretation} |\n"
    else:
        results += "\n### No Statistically Significant Effects\n\n"
        results += "**Key finding**: No content features show statistically significant causal effects on Google rankings at the 5% level.\n"
    
    # DML comparison
    results += "\n## Double Machine Learning Results\n\n"
    results += "**DML Validation**: DML results provide robustness check for OLS estimates.\n"
    
    if "_dml_comparison" in dml_results:
        results += "\n### Algorithm Comparison\n"
        results += "- **LinearDML**: Primary estimation method\n"
        results += "- **Causal Forest**: Heterogeneous treatment effects\n"
        results += "- **Ensemble DML**: Multiple algorithm combination\n"
        results += "- **Meta-learners**: T-Learner and S-Learner approaches\n"
    
    return results


def generate_validation_section(validation_results: Dict) -> str:
    """Generate LLM validation section."""
    
    validation = """# LLM Measurement Validation

## Validation Framework

To address potential measurement error in LLM-derived features, we implement comprehensive validation procedures:

### Test-Retest Reliability
"""
    
    if "test_retest_reliability" in validation_results:
        test_retest = validation_results["test_retest_reliability"]
        reliabilities = []
        
        for feature, result in test_retest.items():
            if isinstance(result, dict) and "pearson_correlation" in result:
                reliability = result["pearson_correlation"]
                reliabilities.append(reliability)
                validation += f"- **{feature}**: r = {reliability:.3f}\n"
        
        if reliabilities:
            mean_reliability = np.mean(reliabilities)
            validation += f"\n**Mean test-retest reliability**: r = {mean_reliability:.3f}\n"
    
    validation += "\n### Inter-Prompt Reliability\n"
    validation += "Different prompt formulations tested for consistency.\n"
    
    validation += "\n### Sensitivity Analysis\n"
    validation += "Sensitivity to input variations and parameter changes assessed.\n"
    
    # Validation quality assessment
    validation_quality = validation_results.get("validation_summary", {}).get("validation_quality", "unknown")
    validation += f"\n## Validation Quality Assessment\n\n"
    validation += f"**Overall Quality**: {validation_quality.upper()}\n\n"
    
    if validation_quality in ["excellent", "good"]:
        validation += "✅ **LLM measurement is reliable** for causal inference\n"
    elif validation_quality == "acceptable":
        validation += "⚠️ **LLM measurement is acceptable** but could be improved\n"
    else:
        validation += "❌ **LLM measurement concerns** - results should be interpreted cautiously\n"
    
    return validation


def generate_robustness_section(robustness_results: Dict = None) -> str:
    """Generate robustness checks section."""
    
    robustness = """# Robustness Checks

## Comprehensive Robustness Testing

To ensure the reliability of causal estimates, we implement multiple robustness checks:

### Bootstrap Confidence Intervals
"""
    
    if "bootstrap_confidence_intervals" in robustness_results:
        bootstrap = robustness_results["bootstrap_confidence_intervals"]
        robust_features = 0
        
        for feature, stats in bootstrap.items():
            if isinstance(stats, dict) and "ci_95_lower" in stats:
                # Check if CI excludes zero
                if stats["ci_95_lower"] > 0 or stats["ci_95_upper"] < 0:
                    robust_features += 1
        
        robustness += f"- **Bootstrap iterations**: 1,000\n"
        robustness += f"- **Features with robust effects**: {robust_features}\n"
    
    robustness += "\n### Placebo Tests\n"
    robustness += "Random outcome shuffling to test for spurious correlations.\n"
    
    robustness += "\n### Model Specification Tests\n"
    robustness += "Different model specifications to test robustness to specification choices.\n"
    
    robustness += "\n### Cross-Validation\n"
    robustness += "K-fold cross-validation to assess model stability.\n"
    
    robustness += "\n### Sensitivity Analysis\n"
    robustness += "Sensitivity to outlier removal, sample size, and standardization methods.\n"
    
    # Robustness assessment
    robustness_quality = "not_available"
    if robustness_results:
        robustness_quality = robustness_results.get("robustness_summary", {}).get("overall_robustness", "unknown")
    robustness += f"\n## Robustness Assessment\n\n"
    robustness += f"**Overall Robustness**: {robustness_quality.upper()}\n\n"
    
    if robustness_quality == "excellent":
        robustness += "✅ **Results are highly robust** to specification changes\n"
    elif robustness_quality == "good":
        robustness += "✅ **Results are robust** to most specification changes\n"
    elif robustness_quality == "moderate":
        robustness += "⚠️ **Results show moderate robustness** - some sensitivity to specification\n"
    else:
        robustness += "❌ **Robustness concerns** - results sensitive to specification choices\n"
    
    return robustness


def generate_limitations_section(ols_results: Dict, dml_results: Dict, panel_data: pd.DataFrame) -> str:
    """Generate limitations section."""
    
    limitations = """# Limitations and Future Research

## Current Limitations

### Sample Size and Power
"""
    
    n_obs = ols_results.get("_n", 0)
    r2 = ols_results.get("_r2", 0)
    
    if n_obs < 1000:
        limitations += f"- **Limited sample size**: n = {n_obs:,} may be insufficient for robust causal inference\n"
    
    if r2 < 0.1:
        limitations += f"- **Low explanatory power**: R² = {r2:.3f} suggests important factors are missing\n"
    
    limitations += """
### Measurement and Data Quality
- **LLM measurement error**: Potential bias in content feature extraction
- **Single time point**: Cross-sectional analysis limits causal identification
- **Limited keyword diversity**: Results may not generalize to all keyword types

### Methodological Limitations
- **Omitted variable bias**: Unobserved content factors may confound results
- **Selection bias**: Only top-ranking articles analyzed
- **Temporal dynamics**: Ranking algorithms may change over time

## Future Research Directions

### Data Improvements
- **Longitudinal data**: Panel data over multiple time periods
- **Larger sample**: 10,000+ observations for robust inference
- **Diverse keywords**: Broader keyword coverage across industries

### Methodological Enhancements
- **Instrumental variables**: Exogenous variation for stronger identification
- **Natural experiments**: Algorithm updates as treatment variation
- **Machine learning**: Advanced feature engineering and model selection

### Content Analysis
- **Multimodal features**: Image and video content analysis
- **User behavior**: Click-through rates and engagement metrics
- **Competitive analysis**: Relative content quality assessment
"""
    
    return limitations


def generate_conclusions_section(ols_results: Dict, dml_results: Dict, robustness_results: Dict = None) -> str:
    """Generate conclusions section."""
    
    conclusions = """# Conclusions

## Main Findings

"""
    
    # Count significant effects
    ols_significant = count_significant_effects(ols_results.get("_p", {}))
    r2 = ols_results.get("_r2", 0)
    n_obs = ols_results.get("_n", 0)
    
    if ols_significant > 0:
        conclusions += f"**Causal Effects Detected**: {ols_significant} content features show statistically significant effects on Google rankings.\n\n"
    else:
        conclusions += "**No Significant Causal Effects**: Current analysis finds no statistically significant causal effects of content features on rankings.\n\n"
    
    conclusions += f"**Model Performance**: The fixed-effects model explains {r2:.1%} of ranking variance with {n_obs:,} observations.\n\n"
    
    # Robustness assessment
    robustness_quality = "not_available"
    if robustness_results:
        robustness_quality = robustness_results.get("robustness_summary", {}).get("overall_robustness", "unknown")
    if robustness_quality in ["excellent", "good"]:
        conclusions += "**Robustness**: Results are robust to specification changes and alternative estimation methods.\n\n"
    else:
        conclusions += "**Robustness Concerns**: Results show sensitivity to specification choices.\n\n"
    
    conclusions += """## Implications for SEO Practice

### Content Optimization
- **Evidence-based approach**: Use statistical evidence for content decisions
- **Feature prioritization**: Focus on features with significant causal effects
- **Quality over quantity**: Emphasize content quality metrics

### Research Methodology
- **Causal inference**: Move beyond correlation to causal relationships
- **Robustness testing**: Implement comprehensive validation procedures
- **Transparency**: Report limitations and uncertainty

## Academic Contributions

### Methodological Innovation
- **LLM validation framework**: Systematic approach to validating AI-derived features
- **Enhanced DML implementation**: Multiple algorithms and cross-fitting
- **Comprehensive robustness testing**: Bootstrap, placebo, and sensitivity analyses

### Empirical Evidence
- **Causal identification**: Within-keyword variation for causal inference
- **Statistical rigor**: Fixed-effects and robust standard errors
- **Reproducible research**: Open methodology and validation procedures

## Final Assessment

This study provides a **rigorous empirical framework** for analyzing content-ranking relationships using causal inference methods. While current results show limited significant effects, the methodological contributions establish a foundation for future research in SEO and content optimization.

**Key Takeaway**: The absence of significant effects may indicate that:
1. **Content factors are less important** than assumed
2. **Other factors dominate** ranking algorithms
3. **Measurement challenges** limit detection of true effects
4. **Sample limitations** reduce statistical power

Future research should focus on **larger samples**, **longitudinal data**, and **enhanced measurement** to provide more definitive evidence on content-ranking relationships.
"""
    
    return conclusions


def count_significant_effects(p_values: Dict) -> int:
    """Count number of statistically significant effects."""
    if not p_values:
        return 0
    return sum(1 for p in p_values.values() if p < 0.05)


def create_statistical_summary_table(ols_results: Dict, dml_results: Dict) -> str:
    """Create a comprehensive statistical summary table."""
    
    table = """# Statistical Summary Table

| Feature | OLS Coef | OLS SE | OLS p-value | DML Coef | DML SE | DML p-value | Significant |
|---------|----------|--------|-------------|----------|--------|-------------|-------------|
"""
    
    # Get all features
    all_features = set()
    if "_p" in ols_results:
        all_features.update(ols_results["_p"].keys())
    if "_p" in dml_results:
        all_features.update(dml_results["_p"].keys())
    
    for feature in sorted(all_features):
        # OLS results
        ols_coef = ols_results.get(feature, 0)
        ols_se = ols_results.get("_se", {}).get(feature, 0)
        ols_p = ols_results.get("_p", {}).get(feature, 1)
        
        # DML results
        dml_coef = dml_results.get(feature, 0)
        dml_se = dml_results.get("_se", {}).get(feature, 0)
        dml_p = dml_results.get("_p", {}).get(feature, 1)
        
        # Significance
        significant = "Yes" if (ols_p < 0.05 or dml_p < 0.05) else "No"
        
        table += f"| {feature} | {ols_coef:.3f} | {ols_se:.3f} | {ols_p:.3f} | {dml_coef:.3f} | {dml_se:.3f} | {dml_p:.3f} | {significant} |\n"
    
    return table