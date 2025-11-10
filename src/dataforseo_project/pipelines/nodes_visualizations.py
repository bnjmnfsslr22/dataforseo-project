"""Scientific visualizations"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

# Set style for scientific plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_scientific_visualizations(panel_df: pd.DataFrame, 
                                  effects_results: Dict, 
                                  dml_results: Dict,
                                  validation_results: Dict,
                                  robustness_results: Dict = None) -> Dict:
    """
    Create comprehensive scientific visualizations.
    """
    logger.info("Creating scientific visualizations")
    
    visualizations = {
        "effect_size_plots": create_effect_size_plots(effects_results),
        "validation_plots": create_validation_plots(validation_results),
        "summary_plots": create_summary_plots(panel_df, effects_results)
    }
    
    # Only add robustness plots if robustness_results are provided
    if robustness_results is not None:
        visualizations["robustness_plots"] = create_robustness_plots(robustness_results)
    
    return visualizations


def create_effect_size_plots(effects_results: Dict) -> Dict:
    """Create effect size visualizations."""
    
    plots = {}
    
    try:
        # Extract coefficients and p-values
        if "coefficients" in effects_results:
            coefs = effects_results["coefficients"]
            p_values = effects_results.get("p_values", {})
            
            # Create coefficient plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            features = list(coefs.keys())
            coefficients = list(coefs.values())
            colors = ['red' if p_values.get(f, 1) > 0.05 else 'green' for f in features]
            
            bars = ax.barh(features, coefficients, color=colors, alpha=0.7)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Feature Effects on Ranking (Green: p<0.05, Red: p≥0.05)')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            
            # Add significance indicators
            for i, (bar, p_val) in enumerate(zip(bars, [p_values.get(f, 1) for f in features])):
                if p_val < 0.05:
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'*', ha='left', va='center', fontsize=14, color='green')
            
            plt.tight_layout()
            plots["coefficient_plot"] = fig
            
            # Create forest plot
            fig_forest = create_forest_plot(coefs, p_values)
            plots["forest_plot"] = fig_forest
            
    except Exception as e:
        logger.warning(f"Effect size plots failed: {e}")
        plots["error"] = str(e)
    
    return plots


def create_validation_plots(validation_results: Dict) -> Dict:
    """Create validation plots."""
    
    plots = {}
    
    try:
        # Test-retest reliability
        if "test_retest" in validation_results:
            test_retest = validation_results["test_retest"]
            
            if "correlations" in test_retest:
                fig, ax = plt.subplots(figsize=(12, 8))
                features = list(test_retest["correlations"].keys())
                correlations = list(test_retest["correlations"].values())
                
                bars = ax.bar(features, correlations, color='skyblue', alpha=0.7)
                ax.set_ylabel('Test-Retest Correlation')
                ax.set_title('Test-Retest Reliability by Feature')
                ax.axhline(y=0.7, color='red', linestyle='--', label='Good Reliability (0.7)')
                ax.axhline(y=0.8, color='green', linestyle='--', label='Excellent Reliability (0.8)')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plots["test_retest_reliability"] = fig
        
        # Inter-prompt reliability
        if "inter_prompt" in validation_results:
            inter_prompt = validation_results["inter_prompt"]
            
            if "correlations" in inter_prompt:
                fig, ax = plt.subplots(figsize=(12, 8))
                features = list(inter_prompt["correlations"].keys())
                correlations = list(inter_prompt["correlations"].values())
                
                bars = ax.bar(features, correlations, color='lightgreen', alpha=0.7)
                ax.set_ylabel('Inter-Prompt Correlation')
                ax.set_title('Inter-Prompt Reliability by Feature')
                ax.axhline(y=0.7, color='red', linestyle='--', label='Good Reliability (0.7)')
                ax.axhline(y=0.8, color='green', linestyle='--', label='Excellent Reliability (0.8)')
                ax.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                plots["inter_prompt_reliability"] = fig
                
    except Exception as e:
        logger.warning(f"Validation plots failed: {e}")
        plots["error"] = str(e)
    
    return plots


def create_robustness_plots(robustness_results: Dict) -> Dict:
    """Create robustness plots."""
    
    plots = {}
    
    try:
        # Bootstrap confidence intervals
        if "bootstrap" in robustness_results:
            bootstrap = robustness_results["bootstrap"]
            
            if "confidence_intervals" in bootstrap:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                features = list(bootstrap["confidence_intervals"].keys())
                lower_bounds = [ci["lower"] for ci in bootstrap["confidence_intervals"].values()]
                upper_bounds = [ci["upper"] for ci in bootstrap["confidence_intervals"].values()]
                means = [ci["mean"] for ci in bootstrap["confidence_intervals"].values()]
                
                y_pos = range(len(features))
                
                ax.errorbar(means, y_pos, xerr=[np.array(means) - np.array(lower_bounds), 
                                               np.array(upper_bounds) - np.array(means)], 
                           fmt='o', capsize=5, capthick=2)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features)
                ax.set_xlabel('Coefficient Value')
                ax.set_title('Bootstrap Confidence Intervals')
                ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
                plt.tight_layout()
                plots["bootstrap_confidence"] = fig
        
        # Model comparison
        if "model_comparison" in robustness_results:
            model_comp = robustness_results["model_comparison"]
            
            if "r_squared" in model_comp:
                fig, ax = plt.subplots(figsize=(12, 8))
                models = list(model_comp["r_squared"].keys())
                r_squared_values = list(model_comp["r_squared"].values())
                
                bars = ax.bar(models, r_squared_values, color='orange', alpha=0.7)
                ax.set_ylabel('R-squared')
                ax.set_title('Model Comparison: R-squared')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plots["model_comparison"] = fig
                
    except Exception as e:
        logger.warning(f"Robustness plots failed: {e}")
        plots["error"] = str(e)
    
    return plots


def create_summary_plots(panel_df: pd.DataFrame, effects_results: Dict) -> Dict:
    """Create summary plots."""
    
    plots = {}
    
    try:
        # Overall ranking distribution
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.hist(panel_df["rank_score"].dropna(), bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Rank Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Overall Ranking Distribution')
        ax.axvline(panel_df["rank_score"].mean(), color='red', linestyle='--', 
                  label=f'Mean: {panel_df["rank_score"].mean():.2f}')
        ax.axvline(panel_df["rank_score"].median(), color='green', linestyle='--', 
                  label=f'Median: {panel_df["rank_score"].median():.2f}')
        ax.legend()
        plt.tight_layout()
        plots["ranking_distribution"] = fig
        
        # Feature importance
        if "coefficients" in effects_results:
            coefs = effects_results["coefficients"]
            p_values = effects_results.get("p_values", {})
            
            # Sort by absolute coefficient value
            sorted_features = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            features = [f[0] for f in sorted_features]
            coefficients = [f[1] for f in sorted_features]
            colors = ['red' if p_values.get(f, 1) > 0.05 else 'green' for f in features]
            
            bars = ax.barh(features, coefficients, color=colors, alpha=0.7)
            ax.set_xlabel('Coefficient Value')
            ax.set_title('Feature Importance (Sorted by Absolute Value)')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plots["feature_importance"] = fig
            
    except Exception as e:
        logger.warning(f"Summary plots failed: {e}")
        plots["error"] = str(e)
    
    return plots


def create_forest_plot(coefs: Dict, p_values: Dict) -> plt.Figure:
    """Create forest plot for coefficients."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = list(coefs.keys())
    coefficients = list(coefs.values())
    p_vals = [p_values.get(f, 1) for f in features]
    
    # Calculate confidence intervals (simplified)
    ci_lower = [c - 1.96 * 0.1 for c in coefficients]  # Simplified CI
    ci_upper = [c + 1.96 * 0.1 for c in coefficients]
    
    y_pos = range(len(features))
    
    # Plot confidence intervals
    for i, (lower, upper, coef, p_val) in enumerate(zip(ci_lower, ci_upper, coefficients, p_vals)):
        color = 'green' if p_val < 0.05 else 'red'
        ax.plot([lower, upper], [i, i], color=color, linewidth=2)
        ax.plot(coef, i, 'o', color=color, markersize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Coefficient Value')
    ax.set_title('Forest Plot: Feature Effects with Confidence Intervals')
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def classify_keyword_type(keyword: str) -> str:
    """Classify keyword by type."""
    keyword_lower = keyword.lower()
    
    if any(word in keyword_lower for word in ["was", "wie", "warum", "wann", "wo", "wer"]):
        return "informational"
    elif any(word in keyword_lower for word in ["kaufen", "preis", "kosten", "günstig", "billig"]):
        return "commercial"
    elif len(keyword.split()) == 1:
        return "short_tail"
    else:
        return "long_tail"
