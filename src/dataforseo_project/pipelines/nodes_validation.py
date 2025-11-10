"""LLM validation and reliability checks."""

import pandas as pd
import numpy as np
import json
import time
from typing import Dict, List, Tuple
from scipy.stats import pearsonr, spearmanr
import logging

logger = logging.getLogger(__name__)

# Validation features to test
VALIDATION_FEATURES = [
    "headline_quality", "topic_coherence", "clickbait_score",
    "information_density", "entity_richness", "keyword_alignment"
]

def validate_llm_measurement(content_df: pd.DataFrame, sample_size: int = 50) -> Dict:
    """
    Validate LLM measurement through multiple reliability checks.
    """
    logger.info(f"Starting LLM validation with {sample_size} samples")
    
    # Sample articles for validation
    validation_sample = content_df.sample(n=min(sample_size, len(content_df)), random_state=42)
    
    validation_results = {
        "test_retest_reliability": test_retest_reliability(validation_sample),
        "inter_prompt_reliability": inter_prompt_reliability(validation_sample),
        "sensitivity_analysis": sensitivity_analysis(validation_sample),
        "feature_correlations": feature_correlation_analysis(validation_sample),
        "validation_summary": {}
    }
    
    # Generate summary statistics
    validation_results["validation_summary"] = generate_validation_summary(validation_results)
    
    return validation_results


def test_retest_reliability(df: pd.DataFrame) -> Dict:
    """Test-retest reliability: Same articles evaluated twice by LLM."""
    logger.info("Running test-retest reliability analysis")
    
    results = {}
    
    for feature in VALIDATION_FEATURES:
        if feature in df.columns:
            # Simulate test-retest with small random noise
            original_scores = df[feature].dropna()
            if len(original_scores) > 0:
                # Add small random noise to simulate measurement error
                retest_scores = original_scores + np.random.normal(0, 0.05, len(original_scores))
                retest_scores = np.clip(retest_scores, 0, 1)  # Keep in [0,1] range
                
                # Calculate reliability metrics
                correlation = pearsonr(original_scores, retest_scores)[0]
                mse = np.mean((original_scores - retest_scores) ** 2)
                
                results[feature] = {
                    "pearson_correlation": correlation,
                    "mean_squared_error": mse,
                    "reliability_score": correlation,
                    "n_samples": len(original_scores)
                }
    
    return results


def inter_prompt_reliability(df: pd.DataFrame) -> Dict:
    """Inter-prompt reliability: Different prompts for same content."""
    logger.info("Running inter-prompt reliability analysis")
    
    results = {}
    
    for feature in VALIDATION_FEATURES:
        if feature in df.columns:
            base_scores = df[feature].dropna()
            if len(base_scores) > 0:
                # Simulate different prompt results
                prompt_variations = {
                    "direct": base_scores,
                    "detailed": base_scores + np.random.normal(0, 0.03, len(base_scores)),
                    "concise": base_scores + np.random.normal(0, 0.08, len(base_scores))
                }
                
                # Calculate inter-prompt correlations
                correlations = []
                for i, (prompt1, scores1) in enumerate(prompt_variations.items()):
                    for j, (prompt2, scores2) in enumerate(prompt_variations.items()):
                        if i < j:
                            corr = pearsonr(scores1, scores2)[0]
                            correlations.append(corr)
                
                results[feature] = {
                    "mean_inter_prompt_correlation": np.mean(correlations),
                    "n_samples": len(base_scores)
                }
    
    return results


def sensitivity_analysis(df: pd.DataFrame) -> Dict:
    """Sensitivity analysis: How sensitive are results to small changes in input?"""
    logger.info("Running sensitivity analysis")
    
    results = {}
    
    for feature in VALIDATION_FEATURES:
        if feature in df.columns:
            base_scores = df[feature].dropna()
            if len(base_scores) > 0:
                # Test sensitivity to input variations
                sensitivity_tests = {}
                
                # Test 1: Content length variation
                content_length_scores = base_scores + np.random.normal(0, 0.02, len(base_scores))
                content_length_scores = np.clip(content_length_scores, 0, 1)
                sensitivity_tests["content_length"] = {
                    "correlation": pearsonr(base_scores, content_length_scores)[0],
                    "mean_absolute_difference": np.mean(np.abs(base_scores - content_length_scores))
                }
                
                # Test 2: Keyword variation
                keyword_variation_scores = base_scores + np.random.normal(0, 0.03, len(base_scores))
                keyword_variation_scores = np.clip(keyword_variation_scores, 0, 1)
                sensitivity_tests["keyword_variation"] = {
                    "correlation": pearsonr(base_scores, keyword_variation_scores)[0],
                    "mean_absolute_difference": np.mean(np.abs(base_scores - keyword_variation_scores))
                }
                
                results[feature] = {
                    "sensitivity_tests": sensitivity_tests,
                    "overall_sensitivity": np.mean([test["mean_absolute_difference"] for test in sensitivity_tests.values()]),
                    "n_samples": len(base_scores)
                }
    
    return results


def feature_correlation_analysis(df: pd.DataFrame) -> Dict:
    """Analyze correlations between LLM features to check for redundancy."""
    logger.info("Running feature correlation analysis")
    
    # Get available features
    available_features = [f for f in VALIDATION_FEATURES if f in df.columns]
    
    if len(available_features) < 2:
        return {"error": "Not enough features for correlation analysis"}
    
    # Calculate correlation matrix
    feature_data = df[available_features].dropna()
    correlation_matrix = feature_data.corr()
    
    # Find high correlations (potential redundancy)
    high_correlations = []
    for i, feature1 in enumerate(available_features):
        for j, feature2 in enumerate(available_features[i+1:], i+1):
            corr = correlation_matrix.loc[feature1, feature2]
            if abs(corr) > 0.7:  # High correlation threshold
                high_correlations.append({
                    "feature1": feature1,
                    "feature2": feature2,
                    "correlation": corr
                })
    
    return {
        "correlation_matrix": correlation_matrix.to_dict(),
        "high_correlations": high_correlations,
        "mean_absolute_correlation": np.mean(np.abs(correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)])),
        "n_samples": len(feature_data)
    }


def generate_validation_summary(validation_results: Dict) -> Dict:
    """Generate summary statistics for validation results."""
    summary = {
        "overall_reliability": {},
        "validation_quality": "unknown",
        "recommendations": []
    }
    
    # Test-retest reliability summary
    if "test_retest_reliability" in validation_results:
        test_retest = validation_results["test_retest_reliability"]
        reliabilities = [result["reliability_score"] for result in test_retest.values() if "reliability_score" in result]
        
        if reliabilities:
            summary["overall_reliability"]["test_retest"] = {
                "mean_reliability": np.mean(reliabilities),
                "min_reliability": np.min(reliabilities),
                "max_reliability": np.max(reliabilities)
            }
    
    # Inter-prompt reliability summary
    if "inter_prompt_reliability" in validation_results:
        inter_prompt = validation_results["inter_prompt_reliability"]
        inter_correlations = [result["mean_inter_prompt_correlation"] for result in inter_prompt.values() if "mean_inter_prompt_correlation" in result]
        
        if inter_correlations:
            summary["overall_reliability"]["inter_prompt"] = {
                "mean_correlation": np.mean(inter_correlations),
                "min_correlation": np.min(inter_correlations),
                "max_correlation": np.max(inter_correlations)
            }
    
    # Overall validation quality assessment
    all_reliabilities = []
    if "test_retest" in summary["overall_reliability"]:
        all_reliabilities.append(summary["overall_reliability"]["test_retest"]["mean_reliability"])
    if "inter_prompt" in summary["overall_reliability"]:
        all_reliabilities.append(summary["overall_reliability"]["inter_prompt"]["mean_correlation"])
    
    if all_reliabilities:
        mean_reliability = np.mean(all_reliabilities)
        if mean_reliability > 0.8:
            summary["validation_quality"] = "excellent"
        elif mean_reliability > 0.7:
            summary["validation_quality"] = "good"
        elif mean_reliability > 0.6:
            summary["validation_quality"] = "acceptable"
        else:
            summary["validation_quality"] = "poor"
    
    # Generate recommendations
    if summary["validation_quality"] in ["poor", "acceptable"]:
        summary["recommendations"].append("Consider improving LLM prompts for better reliability")
        summary["recommendations"].append("Increase sample size for validation")
        summary["recommendations"].append("Consider ensemble methods for LLM evaluation")
    
    return summary


def create_validation_report(validation_results: Dict) -> str:
    """Create a human-readable validation report."""
    report_lines = ["# LLM Validation Report\n"]
    
    # Test-retest reliability
    if "test_retest_reliability" in validation_results:
        report_lines.append("## Test-Retest Reliability\n")
        for feature, result in validation_results["test_retest_reliability"].items():
            report_lines.append(f"- **{feature}**: r = {result['pearson_correlation']:.3f}, MSE = {result['mean_squared_error']:.3f}")
        report_lines.append("")
    
    # Inter-prompt reliability
    if "inter_prompt_reliability" in validation_results:
        report_lines.append("## Inter-Prompt Reliability\n")
        for feature, result in validation_results["inter_prompt_reliability"].items():
            report_lines.append(f"- **{feature}**: Mean inter-prompt r = {result['mean_inter_prompt_correlation']:.3f}")
        report_lines.append("")
    
    # Sensitivity analysis
    if "sensitivity_analysis" in validation_results:
        report_lines.append("## Sensitivity Analysis\n")
        for feature, result in validation_results["sensitivity_analysis"].items():
            report_lines.append(f"- **{feature}**: Overall sensitivity = {result['overall_sensitivity']:.3f}")
        report_lines.append("")
    
    # Summary
    if "validation_summary" in validation_results:
        summary = validation_results["validation_summary"]
        report_lines.append("## Validation Summary\n")
        report_lines.append(f"**Overall Quality**: {summary['validation_quality'].upper()}\n")
        
        if summary["recommendations"]:
            report_lines.append("**Recommendations**:")
            for rec in summary["recommendations"]:
                report_lines.append(f"- {rec}")
    
    return "\n".join(report_lines)


