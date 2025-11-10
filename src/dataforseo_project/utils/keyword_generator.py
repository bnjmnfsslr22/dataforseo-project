"""
Intelligent keyword generation.
Generates thematically relevant keywords optimized for causal inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

# Thematic keyword categories for comprehensive analysis
KEYWORD_CATEGORIES = {
    "technology": {
        "ai_ml": [
            "künstliche intelligenz", "machine learning", "deep learning", "neuronale netze",
            "chatgpt", "openai", "google bard", "claude", "gpt-4", "gpt-3.5",
            "algorithmus", "automatisierung", "roboter", "datenanalyse", "big data",
            "maschinelles lernen", "künstliche neuronale netze", "tensorflow", "pytorch",
            "computer vision", "natural language processing", "nlp", "sprachmodell"
        ],
        "digital_transformation": [
            "digitalisierung", "industrie 4.0", "iot", "internet der dinge", "cloud computing",
            "edge computing", "5g", "blockchain", "kryptowährung", "bitcoin", "ethereum",
            "cybersecurity", "datenschutz", "dsgvo", "privacy", "verschlüsselung"
        ],
        "software_development": [
            "programmierung", "softwareentwicklung", "coding", "python", "javascript",
            "java", "c++", "react", "vue.js", "angular", "node.js", "docker", "kubernetes",
            "agile", "scrum", "devops", "ci/cd", "microservices", "api", "rest api"
        ]
    },
    
    "business": {
        "startups": [
            "startup", "gründung", "unternehmer", "venture capital", "finanzierung",
            "pitch deck", "business plan", "marketing", "vertrieb", "kundenakquise",
            "scaling", "wachstum", "exit", "ipo", "acquisition", "merger"
        ],
        "ecommerce": [
            "online shop", "e-commerce", "amazon", "shopify", "woocommerce",
            "dropshipping", "affiliate marketing", "seo", "google ads", "facebook ads",
            "conversion rate", "checkout", "payment", "logistik", "versand"
        ],
        "finance": [
            "fintech", "banking", "investment", "trading", "aktien", "etf",
            "kryptowährung", "defi", "nft", "metaverse", "web3", "dapp",
            "robo advisor", "neobank", "mobile payment", "paypal", "stripe"
        ]
    },
    
    "science_research": {
        "climate_energy": [
            "klimawandel", "erneuerbare energien", "solar", "windkraft", "wasserstoff",
            "co2", "nachhaltigkeit", "umweltschutz", "energiewende", "e-mobilität",
            "tesla", "batterie", "speicher", "smart grid", "energieeffizienz"
        ],
        "health_medicine": [
            "medizin", "gesundheit", "pharma", "biotech", "genetik", "genom",
            "impfung", "covid", "pandemie", "telemedizin", "digital health",
            "wearables", "fitness tracker", "gesundheitsapp", "diagnose", "therapie"
        ],
        "space_physics": [
            "raumfahrt", "spacex", "nasa", "mars", "mond", "satellit",
            "quantenphysik", "quantencomputer", "quanten", "relativitätstheorie",
            "teilchenphysik", "cern", "lhc", "forschung", "wissenschaft"
        ]
    },
    
    "society_culture": {
        "politics": [
            "politik", "wahl", "bundestag", "regierung", "opposition", "partei",
            "demokratie", "wahlen", "abstimmung", "parlament", "minister",
            "kanzler", "präsident", "europa", "eu", "brexit", "nato"
        ],
        "social_media": [
            "social media", "facebook", "instagram", "tiktok", "twitter", "linkedin",
            "youtube", "influencer", "content creator", "viral", "trending",
            "hashtag", "story", "reel", "livestream", "podcast", "streaming"
        ],
        "education": [
            "bildung", "schule", "universität", "studium", "ausbildung", "weiterbildung",
            "online lernen", "e-learning", "mooc", "edtech", "digitales lernen",
            "lehrer", "professor", "student", "abschluss", "zertifikat"
        ]
    }
}

# High-impact keywords for causal inference (more variation in content quality)
HIGH_IMPACT_KEYWORDS = [
    # Technology topics with high content variation
    "künstliche intelligenz", "chatgpt", "machine learning", "deep learning",
    "blockchain", "kryptowährung", "bitcoin", "nft", "metaverse",
    
    # Business topics with diverse content quality
    "startup", "e-commerce", "marketing", "seo", "fintech",
    "venture capital", "investment", "trading", "cryptocurrency",
    
    # Science topics with varying information density
    "klimawandel", "erneuerbare energien", "covid", "impfung", "gesundheit",
    "raumfahrt", "spacex", "quantencomputer", "forschung",
    
    # Social topics with different clickbait levels
    "politik", "wahl", "social media", "influencer", "youtube",
    "bildung", "universität", "studium", "online lernen"
]

def generate_keywords_for_thesis(
    n_keywords: int = 100,
    category_distribution: Dict[str, float] = None,
    include_high_impact: bool = True,
    min_keywords_per_category: int = 5
) -> List[Dict]:
    """
    
    Args:
        n_keywords: Total number of keywords to generate
        category_distribution: Distribution across categories (default: balanced)
        include_high_impact: Include high-impact keywords for causal inference
        min_keywords_per_category: Minimum keywords per category
    
    Returns:
        List of keyword dictionaries with metadata
    """
    logger.info(f"Generating {n_keywords} keywords")
    
    if category_distribution is None:
        # Balanced distribution across categories
        category_distribution = {
            "technology": 0.35,
            "business": 0.25,
            "science_research": 0.25,
            "society_culture": 0.15
        }
    
    keywords = []
    
    # Add high-impact keywords first (guaranteed inclusion)
    if include_high_impact:
        high_impact_count = min(20, n_keywords // 5)
        for i, keyword in enumerate(HIGH_IMPACT_KEYWORDS[:high_impact_count]):
            keywords.append({
                "keyword": keyword,
                "category": "high_impact",
                "subcategory": "causal_inference_optimized",
                "priority": "high",
                "expected_variation": "high",
                "content_quality_range": "wide"
            })
    
    # Generate keywords from categories
    remaining_keywords = n_keywords - len(keywords)
    
    for main_category, subcategories in KEYWORD_CATEGORIES.items():
        if main_category not in category_distribution:
            continue
            
        category_count = int(remaining_keywords * category_distribution[main_category])
        category_count = max(category_count, min_keywords_per_category)
        
        # Distribute across subcategories
        subcategory_names = list(subcategories.keys())
        keywords_per_subcategory = category_count // len(subcategory_names)
        
        for subcategory_name, subcategory_keywords in subcategories.items():
            # Select keywords from this subcategory
            selected_keywords = np.random.choice(
                subcategory_keywords, 
                size=min(keywords_per_subcategory, len(subcategory_keywords)),
                replace=False
            )
            
            for keyword in selected_keywords:
                if len(keywords) >= n_keywords:
                    break
                    
                keywords.append({
                    "keyword": keyword,
                    "category": main_category,
                    "subcategory": subcategory_name,
                    "priority": "medium",
                    "expected_variation": "medium",
                    "content_quality_range": "moderate"
                })
    
    # Fill remaining slots with random keywords from all categories
    all_keywords = []
    for subcategories in KEYWORD_CATEGORIES.values():
        for keyword_list in subcategories.values():
            all_keywords.extend(keyword_list)
    
    while len(keywords) < n_keywords:
        remaining_keywords = [kw for kw in all_keywords 
                            if kw not in [k["keyword"] for k in keywords]]
        if not remaining_keywords:
            break
            
        selected_keyword = np.random.choice(remaining_keywords)
        keywords.append({
            "keyword": selected_keyword,
            "category": "mixed",
            "subcategory": "random",
            "priority": "low",
            "expected_variation": "unknown",
            "content_quality_range": "unknown"
        })
    
    logger.info(f"Generated {len(keywords)} keywords across {len(set(k['category'] for k in keywords))} categories")
    return keywords[:n_keywords]

def create_keyword_csv(keywords: List[Dict], output_path: str = "data/01_raw/seed_keywords.csv"):
    """Create CSV file with generated keywords."""
    df = pd.DataFrame(keywords)
    df[["keyword"]].to_csv(output_path, index=False)
    logger.info(f"Saved {len(keywords)} keywords to {output_path}")
    return df

def analyze_keyword_distribution(keywords: List[Dict]) -> Dict:
    """Analyze keyword distribution for optimization."""
    df = pd.DataFrame(keywords)
    
    analysis = {
        "total_keywords": len(keywords),
        "category_distribution": df["category"].value_counts().to_dict(),
        "subcategory_distribution": df["subcategory"].value_counts().to_dict(),
        "priority_distribution": df["priority"].value_counts().to_dict(),
        "expected_variation_distribution": df["expected_variation"].value_counts().to_dict(),
        "content_quality_distribution": df["content_quality_range"].value_counts().to_dict()
    }
    
    return analysis

def validate_keywords(keywords: List[Dict]) -> Dict:
    """Validate keywords"""
    df = pd.DataFrame(keywords)
    
    validation = {
        "total_count": len(keywords),
        "min_required": 50,
        "sufficient_for_power": len(keywords) >= 50,
        "category_balance": len(df["category"].unique()) >= 3,
        "high_impact_included": "high_impact" in df["category"].values,
        "variation_coverage": len(df["expected_variation"].unique()) >= 2,
        "recommendations": []
    }
    
    # Generate recommendations
    if not validation["sufficient_for_power"]:
        validation["recommendations"].append("Increase keyword count to at least 50 for statistical power")
    
    if not validation["category_balance"]:
        validation["recommendations"].append("Include keywords from at least 3 different categories")
    
    if not validation["high_impact_included"]:
        validation["recommendations"].append("Include high-impact keywords for causal inference")
    
    if len(keywords) < 100:
        validation["recommendations"].append("Consider 100+ keywords")
    
    return validation

# Example usage and testing
if __name__ == "__main__":
    # Generate keywords
    keywords = generate_keywords(n_keywords=100)
    
    # Create CSV file
    df = create_keyword_csv(keywords)
    
    # Analyze distribution
    analysis = analyze_keyword_distribution(keywords)
    print("Keyword Distribution Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Validate for thesis
    validation = validate_keywords_for_thesis(keywords)
    print("\nValidation Results:")
    for key, value in validation.items():
        print(f"  {key}: {value}")
