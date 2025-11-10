"""Progress tracking nodes for pipeline execution."""

import time
import logging
from typing import Dict, List
from tqdm import tqdm
import pandas as pd

logger = logging.getLogger(__name__)

class ProgressTracker:
    """Track progress of long-running operations."""
    
    def __init__(self, total_items: int, operation_name: str):
        self.total_items = total_items
        self.operation_name = operation_name
        self.current_item = 0
        self.start_time = time.time()
        self.pbar = None
        
    def start(self):
        """Start progress tracking."""
        self.pbar = tqdm(
            total=self.total_items,
            desc=f"🔄 {self.operation_name}",
            unit="items",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )
        logger.info(f"Starting {self.operation_name}: {self.total_items} items")
        
    def update(self, increment: int = 1, message: str = ""):
        """Update progress."""
        self.current_item += increment
        if self.pbar:
            self.pbar.update(increment)
            if message:
                self.pbar.set_postfix_str(message)
                
    def finish(self):
        """Finish progress tracking."""
        if self.pbar:
            self.pbar.close()
        elapsed = time.time() - self.start_time
        logger.info(f"Completed {self.operation_name} in {elapsed:.1f}s")

def track_serp_progress(seed_df: pd.DataFrame) -> Dict:
    """Track progress for SERP data collection."""
    keywords = seed_df['keyword'].dropna().unique()
    progress = ProgressTracker(len(keywords), "DataForSEO API")
    
    return {
        "total_keywords": len(keywords),
        "progress_tracker": progress,
        "message": f"Tracking progress for {len(keywords)} keywords"
    }

def track_content_progress(content_df: pd.DataFrame) -> Dict:
    """Track progress for content extraction."""
    progress = ProgressTracker(len(content_df), "Web Scraping")
    
    return {
        "total_articles": len(content_df),
        "progress_tracker": progress,
        "message": f"Tracking progress for {len(content_df)} articles"
    }

def track_llm_progress(articles_df: pd.DataFrame) -> Dict:
    """Track progress for LLM feature extraction."""
    progress = ProgressTracker(len(articles_df), "OpenAI API")
    
    return {
        "total_articles": len(articles_df),
        "progress_tracker": progress,
        "message": f"Tracking progress for {len(articles_df)} articles"
    }

def generate_progress_summary(serp_df: pd.DataFrame, content_df: pd.DataFrame, llm_df: pd.DataFrame) -> Dict:
    """Generate progress summary."""
    return {
        "serp_keywords": serp_df['keyword'].nunique() if not serp_df.empty else 0,
        "content_articles": len(content_df) if not content_df.empty else 0,
        "llm_articles": len(llm_df) if not llm_df.empty else 0,
        "completion_rate": "100%" if not llm_df.empty else "0%",
        "message": "Progress tracking completed"
    }
