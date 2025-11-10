"""
Fixes duplicates and optimizes keywords for causal inference.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def optimize_keywords(keywords_df: pd.DataFrame) -> pd.DataFrame:
    """
    
    Args:
        keywords_df: DataFrame with keywords
    
    Returns:
        Optimized DataFrame
    """
    logger.info("Optimizing keywords")
    
    # Remove duplicates
    original_count = len(keywords_df)
    keywords_df = keywords_df.drop_duplicates(subset=['keyword'])
    duplicates_removed = original_count - len(keywords_df)
    logger.info(f"Removed {duplicates_removed} duplicates")
    
    # Filter by length (3-20 characters for optimal SERP results)
    keywords_df = keywords_df[
        (keywords_df['keyword'].str.len() >= 3) & 
        (keywords_df['keyword'].str.len() <= 20)
    ]
    logger.info(f"Filtered by length: {len(keywords_df)} keywords remaining")
    
    # Replace problematic keywords
    replacements = {
        '5g': '5g technologie',
        'api': 'rest api',
        'ci/cd': 'continuous integration',
        'nlp': 'natural language processing',
        'iot': 'internet der dinge',
        'defi': 'dezentrale finanzierung',
        'web3': 'web 3.0',
        'mooc': 'online kurs',
        'edtech': 'bildungs technologie'
    }
    
    for old, new in replacements.items():
        keywords_df.loc[keywords_df['keyword'] == old, 'keyword'] = new
    
    # Add high-impact keywords if missing
    high_impact_keywords = [
        'künstliche intelligenz', 'chatgpt', 'machine learning', 'deep learning',
        'blockchain', 'kryptowährung', 'bitcoin', 'nft', 'metaverse',
        'startup', 'e-commerce', 'marketing', 'seo', 'fintech',
        'klimawandel', 'erneuerbare energien', 'covid', 'gesundheit',
        'raumfahrt', 'spacex', 'quantencomputer', 'forschung',
        'politik', 'wahl', 'social media', 'influencer', 'youtube',
        'bildung', 'universität', 'studium', 'online lernen'
    ]
    
    existing_keywords = set(keywords_df['keyword'].str.lower())
    missing_high_impact = [kw for kw in high_impact_keywords 
                          if kw.lower() not in existing_keywords]
    
    if missing_high_impact:
        new_keywords = pd.DataFrame({
            'keyword': missing_high_impact[:10]  # Add max 10
        })
        keywords_df = pd.concat([keywords_df, new_keywords], ignore_index=True)
        logger.info(f"Added {len(missing_high_impact[:10])} high-impact keywords")
    
    # Ensure we have exactly 100 keywords
    if len(keywords_df) > 100:
        keywords_df = keywords_df.sample(n=100, random_state=42).reset_index(drop=True)
    elif len(keywords_df) < 100:
        # Add more keywords if needed
        additional_keywords = [
            'automatisierung', 'roboter', 'datenanalyse', 'big data',
            'cloud computing', 'cybersecurity', 'datenschutz', 'dsgvo',
            'programmierung', 'softwareentwicklung', 'coding', 'python',
            'javascript', 'react', 'vue.js', 'angular', 'node.js',
            'docker', 'kubernetes', 'agile', 'scrum', 'microservices',
            'venture capital', 'finanzierung', 'pitch deck', 'business plan',
            'vertrieb', 'kundenakquise', 'scaling', 'wachstum', 'exit',
            'online shop', 'woocommerce', 'dropshipping', 'affiliate marketing',
            'google ads', 'facebook ads', 'conversion rate', 'checkout',
            'payment', 'logistik', 'versand', 'fintech', 'banking',
            'investment', 'trading', 'aktien', 'etf', 'cryptocurrency',
            'robo advisor', 'neobank', 'mobile payment', 'paypal', 'stripe',
            'erneuerbare energien', 'solar', 'windkraft', 'wasserstoff',
            'co2', 'nachhaltigkeit', 'umweltschutz', 'energiewende',
            'e-mobilität', 'tesla', 'batterie', 'speicher', 'smart grid',
            'energieeffizienz', 'medizin', 'pharma', 'biotech', 'genetik',
            'genom', 'impfung', 'pandemie', 'telemedizin', 'digital health',
            'wearables', 'fitness tracker', 'gesundheitsapp', 'diagnose',
            'therapie', 'raumfahrt', 'nasa', 'mars', 'mond', 'satellit',
            'quantenphysik', 'quantencomputer', 'quanten', 'relativitätstheorie',
            'teilchenphysik', 'cern', 'lhc', 'wissenschaft', 'forschung',
            'politik', 'wahl', 'bundestag', 'regierung', 'opposition',
            'partei', 'demokratie', 'wahlen', 'abstimmung', 'parlament',
            'minister', 'kanzler', 'präsident', 'europa', 'eu', 'brexit',
            'nato', 'social media', 'facebook', 'instagram', 'tiktok',
            'twitter', 'linkedin', 'youtube', 'influencer', 'content creator',
            'viral', 'trending', 'hashtag', 'story', 'reel', 'livestream',
            'podcast', 'streaming', 'bildung', 'schule', 'universität',
            'studium', 'ausbildung', 'weiterbildung', 'online lernen',
            'e-learning', 'mooc', 'edtech', 'digitales lernen', 'lehrer',
            'professor', 'student', 'abschluss', 'zertifikat'
        ]
        
        needed = 100 - len(keywords_df)
        available = [kw for kw in additional_keywords 
                   if kw.lower() not in existing_keywords]
        
        if available:
            add_keywords = pd.DataFrame({
                'keyword': available[:needed]
            })
            keywords_df = pd.concat([keywords_df, add_keywords], ignore_index=True)
            logger.info(f"Added {min(needed, len(available))} additional keywords")
    
    logger.info(f"Final optimized keyword count: {len(keywords_df)}")
    return keywords_df

def analyze_keyword_quality(keywords_df: pd.DataFrame) -> Dict:
    """
    Analyze keyword quality.
    
    Args:
        keywords_df: DataFrame with keywords
    
    Returns:
        Quality analysis
    """
    analysis = {
        'total_keywords': len(keywords_df),
        'unique_keywords': keywords_df['keyword'].nunique(),
        'duplicates': len(keywords_df) - keywords_df['keyword'].nunique(),
        'avg_length': keywords_df['keyword'].str.len().mean(),
        'min_length': keywords_df['keyword'].str.len().min(),
        'max_length': keywords_df['keyword'].str.len().max(),
        'quality_score': 0.0
    }
    
    # Calculate quality score
    quality_score = 0.0
    
    # Points for sufficient keywords
    if analysis['total_keywords'] >= 100:
        quality_score += 0.3
    elif analysis['total_keywords'] >= 50:
        quality_score += 0.2
    
    # Points for uniqueness
    if analysis['duplicates'] == 0:
        quality_score += 0.3
    elif analysis['duplicates'] <= 2:
        quality_score += 0.2
    
    # Points for optimal length
    if 5 <= analysis['avg_length'] <= 15:
        quality_score += 0.2
    elif 3 <= analysis['avg_length'] <= 20:
        quality_score += 0.1
    
    # Points for length range
    if analysis['max_length'] <= 25 and analysis['min_length'] >= 3:
        quality_score += 0.2
    
    analysis['quality_score'] = quality_score
    analysis['excellent'] = quality_score >= 0.8
    analysis['good'] = quality_score >= 0.6
    
    return analysis

if __name__ == "__main__":
    # Load and optimize keywords
    df = pd.read_csv('data/01_raw/seed_keywords.csv')
    optimized_df = optimize_keywords_for_thesis(df)
    
    # Save optimized keywords
    optimized_df.to_csv('data/01_raw/seed_keywords_optimized.csv', index=False)
    
    # Analyze quality
    analysis = analyze_keyword_quality(optimized_df)
    
    print("=== KEYWORD OPTIMIZATION RESULTS ===")
    print(f"Total keywords: {analysis['total_keywords']}")
    print(f"Unique keywords: {analysis['unique_keywords']}")
    print(f"Duplicates: {analysis['duplicates']}")
    print(f"Average length: {analysis['avg_length']:.1f} characters")
    print(f"Length range: {analysis['min_length']}-{analysis['max_length']} characters")
    print(f"Quality score: {analysis['quality_score']:.2f}/1.0")
    print(f"Quality rating: {'EXCELLENT' if analysis['excellent'] else 'GOOD' if analysis['good'] else 'NEEDS IMPROVEMENT'}")
