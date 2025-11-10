"""Keyword generation node  pipeline."""

import pandas as pd
from typing import Dict
import logging
from ..utils.keyword_generator import (
    generate_keywords_for_thesis, 
    create_keyword_csv, 
    analyze_keyword_distribution,
    validate_keywords_for_thesis
)

logger = logging.getLogger(__name__)

def generate_keywords(params: Dict) -> Dict:
    """
    Generate keywords optimized 
    
    Args:
        params: Dictionary with keyword generation parameters
    
    Returns:
        Dictionary with keyword generation results and metadata
    """
    n_keywords = params.get("n_keywords", 100)
    logger.info(f"Generating {n_keywords} keywords")
    
    try:
        # Generate keywords with optimal distribution for causal inference
        keywords = generate_keywords_for_thesis(
            n_keywords=n_keywords,
            include_high_impact=params.get("include_high_impact", True),
            min_keywords_per_category=params.get("min_keywords_per_category", 8),
            category_distribution=params.get("category_distribution")
        )
        
        # Create CSV file
        df = create_keyword_csv(keywords, "data/01_raw/seed_keywords.csv")
        
        # Analyze distribution
        analysis = analyze_keyword_distribution(keywords)
        
        # Validate for thesis requirements
        validation = validate_keywords_for_thesis(keywords)
        
        # Generate summary
        summary = {
            "generation_successful": True,
            "total_keywords": len(keywords),
            "categories": len(analysis["category_distribution"]),
            "high_impact_keywords": analysis["category_distribution"].get("high_impact", 0),
            "validation_passed": validation["sufficient_for_power"] and validation["category_balance"],
            "recommendations": validation["recommendations"]
        }
        
        logger.info(f"Successfully generated {len(keywords)} keywords across {len(analysis['category_distribution'])} categories")
        
        return {
            "keywords": keywords,
            "analysis": analysis,
            "validation": validation,
            "summary": summary,
            "csv_created": True
        }
        
    except Exception as e:
        logger.error(f"Keyword generation failed: {e}")
        return {
            "generation_successful": False,
            "error": str(e),
            "keywords": [],
            "analysis": {},
            "validation": {},
            "summary": {"generation_successful": False, "error": str(e)},
            "csv_created": False
        }

def validate_keyword_quality(keywords: list) -> Dict:
    """
    Validate keyword quality.
    
    Args:
        keywords: List of keyword dictionaries
    
    Returns:
        Validation results
    """
    logger.info("Validating keyword quality")
    
    if not keywords:
        return {"valid": False, "error": "No keywords provided"}
    
    # Check for duplicates
    keyword_texts = [k["keyword"] for k in keywords]
    duplicates = len(keyword_texts) - len(set(keyword_texts))
    
    # Check category diversity
    categories = set(k["category"] for k in keywords)
    
    # Check for high-impact keywords
    high_impact_count = sum(1 for k in keywords if k["category"] == "high_impact")
    
    validation = {
        "valid": True,
        "total_keywords": len(keywords),
        "duplicates": duplicates,
        "category_diversity": len(categories),
        "high_impact_count": high_impact_count,
        "quality_score": 0.0
    }
    
    # Calculate quality score
    quality_score = 0.0
    
    # Points for sufficient keywords
    if len(keywords) >= 50:
        quality_score += 0.3
    if len(keywords) >= 100:
        quality_score += 0.2
    
    # Points for category diversity
    if len(categories) >= 3:
        quality_score += 0.2
    if len(categories) >= 4:
        quality_score += 0.1
    
    # Points for high-impact keywords
    if high_impact_count >= 10:
        quality_score += 0.2
    
    # Penalty for duplicates
    if duplicates > 0:
        quality_score -= min(0.1, duplicates * 0.01)
    
    validation["quality_score"] = quality_score
    validation["valid"] = quality_score >= 0.6
    
    return validation
