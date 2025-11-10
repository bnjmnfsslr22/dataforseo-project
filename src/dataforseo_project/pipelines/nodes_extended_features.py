"""Extended features for better model quality"""

import pandas as pd
import numpy as np
from typing import Dict, List
import re
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)


def create_extended_features(articles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create extended features for better model quality.
    """
    logger.info("Creating extended features")
    
    df = articles_df.copy()
    
    # Calculate body_word_count if not present (needed for many features)
    if "body_word_count" not in df.columns and "body" in df.columns:
        df["body_word_count"] = df["body"].apply(lambda x: len(str(x).split()) if pd.notna(x) and str(x).strip() else 0)
    
    # Text-based features
    df = add_text_features(df)
    
    # URL-based features
    df = add_url_features(df)
    
    # Content structure features
    df = add_structure_features(df)
    
    # Semantic features
    df = add_semantic_features(df)
    
    # Interaction features
    df = add_interaction_features(df)
    
    return df


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced text-based features."""
    
    # Use "body" column (from cohort_articles_with_content) instead of "body_text"
    if "body" in df.columns:
        # Rename for compatibility with existing code
        if "body_text" not in df.columns:
            df["body_text"] = df["body"]
    
    if "body_text" in df.columns:
        # Text length features
        df["body_char_count"] = df["body_text"].str.len()
        df["body_sentence_count"] = df["body_text"].str.count(r'[.!?]+')
        df["body_paragraph_count"] = df["body_text"].str.count(r'\n\n')
        
        # Readability features
        df["avg_sentence_length"] = df["body_word_count"] / (df["body_sentence_count"] + 1)
        df["avg_paragraph_length"] = df["body_word_count"] / (df["body_paragraph_count"] + 1)
        
        # Text complexity features
        df["text_complexity"] = df["body_char_count"] / (df["body_word_count"] + 1)
        df["sentence_variety"] = df["body_sentence_count"] / (df["body_word_count"] + 1)
        
        # Keyword density features
        if "keyword" in df.columns:
            df["keyword_density"] = df.apply(lambda row: calculate_keyword_density(row["body_text"], row["keyword"]), axis=1)
            df["keyword_frequency"] = df.apply(lambda row: calculate_keyword_frequency(row["body_text"], row["keyword"]), axis=1)
        
        # Text quality features
        df["text_quality_score"] = df.apply(calculate_text_quality, axis=1)
        df["readability_score"] = df.apply(calculate_readability, axis=1)
    
    return df


def add_url_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add URL-based features."""
    
    if "url" in df.columns:
        # URL structure features
        df["url_depth"] = df["url"].apply(calculate_url_depth)
        df["url_length"] = df["url"].str.len()
        df["url_has_https"] = df["url"].str.startswith("https://").astype(int)
        df["url_has_www"] = df["url"].str.contains("www.").astype(int)
        
        # URL parameters
        df["url_has_params"] = df["url"].str.contains("\\?").astype(int)
        df["url_param_count"] = df["url"].str.count("&")
        
        # Domain features
        df["domain"] = df["url"].apply(extract_domain)
        df["domain_length"] = df["domain"].str.len()
        df["domain_has_subdomain"] = df["domain"].str.count("\\.") > 1
        
        # URL keywords
        if "keyword" in df.columns:
            df["url_keyword_match"] = df.apply(lambda row: keyword_in_url(row["url"], row["keyword"]), axis=1)
    
    return df


def add_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add content structure features."""
    
    # Use "body" column (from cohort_articles_with_content) instead of "body_text"
    if "body" in df.columns:
        if "body_text" not in df.columns:
            df["body_text"] = df["body"]
    
    if "body_text" in df.columns:
        # Heading structure
        df["heading_count"] = df["body_text"].str.count(r'<h[1-6]')
        df["h1_count"] = df["body_text"].str.count(r'<h1')
        df["h2_count"] = df["body_text"].str.count(r'<h2')
        df["h3_count"] = df["body_text"].str.count(r'<h3')
        
        # List structure
        df["list_count"] = df["body_text"].str.count(r'<ul>|<ol>')
        df["list_item_count"] = df["body_text"].str.count(r'<li>')
        
        # Link structure
        df["internal_link_count"] = df["body_text"].str.count(r'href="[^"]*"')
        df["external_link_count"] = df["body_text"].str.count(r'href="http')
        
        # Image structure
        df["image_count"] = df["body_text"].str.count(r'<img')
        df["image_alt_count"] = df["body_text"].str.count(r'alt="[^"]*"')
        
        # Table structure
        df["table_count"] = df["body_text"].str.count(r'<table')
        df["table_row_count"] = df["body_text"].str.count(r'<tr')
        
        # Content organization
        df["content_organization"] = df["heading_count"] / (df["body_word_count"] + 1)
        df["list_density"] = df["list_item_count"] / (df["body_word_count"] + 1)
        df["image_density"] = df["image_count"] / (df["body_word_count"] + 1)
    
    return df


def add_semantic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add semantic features."""
    
    # Use "body" column (from cohort_articles_with_content) instead of "body_text"
    if "body" in df.columns:
        if "body_text" not in df.columns:
            df["body_text"] = df["body"]
    
    if "body_text" in df.columns:
        # Question features
        df["question_count"] = df["body_text"].str.count(r'\\?')
        df["question_density"] = df["question_count"] / (df["body_word_count"] + 1)
        
        # Number features
        df["number_count"] = df["body_text"].str.count(r'\\d+')
        df["number_density"] = df["number_count"] / (df["body_word_count"] + 1)
        
        # Date features
        df["date_count"] = df["body_text"].str.count(r'\\d{1,2}[./-]\\d{1,2}[./-]\\d{2,4}')
        df["date_density"] = df["date_count"] / (df["body_word_count"] + 1)
        
        # Quote features
        df["quote_count"] = df["body_text"].str.count(r'"')
        df["quote_density"] = df["quote_count"] / (df["body_word_count"] + 1)
        
        # Emphasis features
        df["bold_count"] = df["body_text"].str.count(r'<b>|<strong>')
        df["italic_count"] = df["body_text"].str.count(r'<i>|<em>')
        df["emphasis_density"] = (df["bold_count"] + df["italic_count"]) / (df["body_word_count"] + 1)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features."""
    
    # Word count interactions
    if "body_word_count" in df.columns:
        df["word_count_squared"] = df["body_word_count"] ** 2
        df["word_count_log"] = np.log(df["body_word_count"] + 1)
        
        # Interaction with other features
        if "url_depth" in df.columns:
            df["word_count_url_depth"] = df["body_word_count"] * df["url_depth"]
        
        if "heading_count" in df.columns:
            df["word_count_headings"] = df["body_word_count"] * df["heading_count"]
    
    # URL depth interactions
    if "url_depth" in df.columns:
        df["url_depth_squared"] = df["url_depth"] ** 2
        df["url_depth_log"] = np.log(df["url_depth"] + 1)
    
    # Content quality interactions
    if "text_quality_score" in df.columns and "readability_score" in df.columns:
        df["quality_readability_interaction"] = df["text_quality_score"] * df["readability_score"]
    
    return df


# Helper functions
def calculate_keyword_density(text: str, keyword: str) -> float:
    """Calculate keyword density."""
    if pd.isna(text) or pd.isna(keyword):
        return 0.0
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    keyword_count = text_lower.count(keyword_lower)
    word_count = len(text_lower.split())
    
    return keyword_count / word_count if word_count > 0 else 0.0


def calculate_keyword_frequency(text: str, keyword: str) -> int:
    """Calculate keyword frequency."""
    if pd.isna(text) or pd.isna(keyword):
        return 0
    
    text_lower = text.lower()
    keyword_lower = keyword.lower()
    
    return text_lower.count(keyword_lower)


def calculate_text_quality(row) -> float:
    """Calculate text quality score."""
    if pd.isna(row.get("body_text")):
        return 0.0
    
    text = row["body_text"]
    word_count = row.get("body_word_count", 0)
    
    if word_count == 0:
        return 0.0
    
    # Simple quality metrics
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = word_count / (sentence_count + 1)
    
    # Quality score based on sentence length and structure
    if avg_sentence_length < 10:
        return 0.3
    elif avg_sentence_length < 20:
        return 0.7
    else:
        return 1.0


def calculate_readability(row) -> float:
    """Calculate readability score."""
    if pd.isna(row.get("body_text")):
        return 0.0
    
    text = row["body_text"]
    word_count = row.get("body_word_count", 0)
    
    if word_count == 0:
        return 0.0
    
    # Simple readability based on sentence length
    sentence_count = text.count('.') + text.count('!') + text.count('?')
    avg_sentence_length = word_count / (sentence_count + 1)
    
    # Readability score (inverse of sentence length)
    if avg_sentence_length < 15:
        return 1.0
    elif avg_sentence_length < 25:
        return 0.7
    else:
        return 0.3


def calculate_url_depth(url: str) -> int:
    """Calculate URL depth."""
    if pd.isna(url):
        return 0
    
    try:
        parsed = urlparse(url)
        path = parsed.path
        if path == '/' or path == '':
            return 0
        return path.count('/')
    except Exception:
        return 0


def extract_domain(url: str) -> str:
    """Extract domain from URL."""
    if pd.isna(url):
        return ""
    
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return ""


def keyword_in_url(url: str, keyword: str) -> int:
    """Check if keyword is in URL."""
    if pd.isna(url) or pd.isna(keyword):
        return 0
    
    url_lower = url.lower()
    keyword_lower = keyword.lower()
    
    return 1 if keyword_lower in url_lower else 0
