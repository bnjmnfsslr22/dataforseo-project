from kedro.pipeline import Pipeline, node
from .nodes_keyword_generation import generate_thesis_keywords
from .nodes_cohort import get_serp_cohort
from .nodes_content_cohort import fetch_cohort_content
from .nodes_struct_cohort import extract_struct_features_cohort
from .nodes_llm_cohort import llm_features_cohort
from .nodes_panel import build_keyword_panel
from .nodes_causal_panel import run_fe_ols
from .nodes_recos import cohort_recommendations
from .nodes_dml import run_dml, cohort_recommendations_dml
from .nodes_validation import validate_llm_measurement, create_validation_report
from .nodes_scientific_reporting import generate_scientific_report, create_statistical_summary_table
from .nodes_extended_features import create_extended_features
from .nodes_visualizations import create_scientific_visualizations


def create_pipeline(**_):
    return Pipeline([
        node(generate_thesis_keywords,
             inputs=["params:keyword_generation"],
             outputs="generated_keywords",
             name="generate_thesis_keywords"),
        node(get_serp_cohort,
             inputs=["generated_keywords","params:dataforseo","params:cohort"],
             outputs="serp_cohort",
             name="get_serp_cohort"),
        node(fetch_cohort_content,
             inputs=["serp_cohort"],
             outputs="cohort_articles_with_content",
             name="cohort_fetch_article_content"),
        node(llm_features_cohort,
             inputs=["cohort_articles_with_content"],
             outputs="cohort_llm_features",
             name="cohort_llm_features"),
        node(create_extended_features,
             inputs=["cohort_articles_with_content"],
             outputs="cohort_extended_features",
             name="create_cohort_extended_features"),
        node(extract_struct_features_cohort,
             inputs=["cohort_articles_with_content"],
             outputs="cohort_struct_features",
             name="cohort_extract_struct_features"),
        node(build_keyword_panel,
             inputs=["serp_cohort","cohort_articles_with_content","cohort_llm_features","cohort_struct_features","cohort_extended_features"],
             outputs="cohort_panel",
             name="build_keyword_panel"),
        node(run_fe_ols,
             inputs=["cohort_panel"],
             outputs="cohort_effects",
             name="run_fe_ols"),
        node(run_dml,
             inputs=["cohort_panel"],
             outputs="cohort_effects_dml",
             name="run_dml"),
        node(cohort_recommendations,
             inputs=["cohort_effects_dml"],
             outputs="cohort_recommendations_dml",
             name="cohort_recommendations"),
        node(validate_llm_measurement,
             inputs=["cohort_articles_with_content"],
             outputs="llm_validation_results",
             name="validate_llm_measurement"),
        node(create_validation_report,
             inputs=["llm_validation_results"],
             outputs="llm_validation_report",
             name="create_validation_report"),
        node(generate_scientific_report,
             inputs=["cohort_effects", "cohort_effects_dml", "llm_validation_results", "cohort_panel"],
             outputs="scientific_report",
             name="generate_scientific_report"),
        node(create_statistical_summary_table,
             inputs=["cohort_effects", "cohort_effects_dml"],
             outputs="statistical_summary_table",
             name="create_statistical_summary_table"),
        node(create_scientific_visualizations,
             inputs=["cohort_panel", "cohort_effects", "cohort_effects_dml", "llm_validation_results"],
             outputs="scientific_visualizations",
             name="create_scientific_visualizations"),
     ])
