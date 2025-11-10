"""
Precise cost calculator with token counting
Tracks costs in real-time during pipeline execution.
"""

import json
import time
import logging
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

# Current API pricing (as of 2024)
@dataclass
class APIPricing:
    # DataForSEO API pricing
    dataforseo_serp_live: float = 0.0012  # per 100 results
    dataforseo_serp_task: float = 0.0003  # per result
    dataforseo_keyword_data: float = 0.002  # per keyword
    dataforseo_backlinks: float = 0.006  # per domain
    dataforseo_onpage: float = 0.002  # per page
    
    # OpenAI API pricing (per 1K tokens unless stated)
    # GPT-4 family
    openai_gpt4_input: float = 0.03
    openai_gpt4_output: float = 0.06
    openai_gpt4_turbo_input: float = 0.01
    openai_gpt4_turbo_output: float = 0.03
    # GPT-5-mini (per 1M tokens → convert to per 1K)
    openai_gpt5mini_input_k: float = 0.25 / 1000.0  # $0.25 per 1M → $0.00025 per 1K
    openai_gpt5mini_output_k: float = 2.00 / 1000.0  # $2.00 per 1M → $0.002 per 1K

class CostTracker:
    """Real-time cost tracking during pipeline execution."""
    
    def __init__(self):
        self.pricing = APIPricing()
        self.costs = {
            'dataforseo': {
                'serp_requests': 0,
                'keyword_requests': 0,
                'backlink_requests': 0,
                'onpage_requests': 0,
                'total_cost': 0.0
            },
            'openai': {
                'input_tokens': 0,
                'output_tokens': 0,
                'requests': 0,
                'total_cost': 0.0
            },
            'total_cost': 0.0
        }
        self.start_time = None
        self.log_file = f"logs/cost_tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    def start_tracking(self):
        """Start cost tracking session."""
        self.start_time = time.time()
        logger.info("Cost tracking started")
    
    def track_dataforseo_serp(self, num_requests: int, results_per_request: int = 40):
        """Track DataForSEO SERP API costs."""
        # Using SERP Task pricing (more accurate for our use case)
        cost = num_requests * self.pricing.dataforseo_serp_task
        self.costs['dataforseo']['serp_requests'] += num_requests
        self.costs['dataforseo']['total_cost'] += cost
        self._update_total()
        logger.info(f"DataForSEO SERP: {num_requests} requests, ${cost:.4f}")
        return cost
    
    def track_openai_request(self, input_tokens: int, output_tokens: int, model: str = "gpt-5-mini"):
        """Track OpenAI API costs."""
        if model.startswith("gpt-5-mini"):
            input_cost = (input_tokens / 1000) * self.pricing.openai_gpt5mini_input_k
            output_cost = (output_tokens / 1000) * self.pricing.openai_gpt5mini_output_k
        elif model.startswith("gpt-4-turbo"):
            input_cost = (input_tokens / 1000) * self.pricing.openai_gpt4_turbo_input
            output_cost = (output_tokens / 1000) * self.pricing.openai_gpt4_turbo_output
        else:  # GPT-4
            input_cost = (input_tokens / 1000) * self.pricing.openai_gpt4_input
            output_cost = (output_tokens / 1000) * self.pricing.openai_gpt4_output
        
        total_cost = input_cost + output_cost
        
        self.costs['openai']['input_tokens'] += input_tokens
        self.costs['openai']['output_tokens'] += output_tokens
        self.costs['openai']['requests'] += 1
        self.costs['openai']['total_cost'] += total_cost
        self._update_total()
        
        logger.info(f"OpenAI {model}: {input_tokens} input + {output_tokens} output tokens = ${total_cost:.4f}")
        return total_cost
    
    def _update_total(self):
        """Update total cost."""
        self.costs['total_cost'] = (
            self.costs['dataforseo']['total_cost'] + 
            self.costs['openai']['total_cost']
        )
    
    def get_current_costs(self) -> Dict:
        """Get current cost summary."""
        runtime = time.time() - self.start_time if self.start_time else 0
        return {
            'runtime_seconds': runtime,
            'runtime_minutes': runtime / 60,
            'costs': self.costs.copy(),
            'timestamp': datetime.now().isoformat()
        }
    
    def save_costs(self):
        """Save cost tracking to file."""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.get_current_costs(), f, indent=2)
            logger.info(f"Cost tracking saved to {self.log_file}")
        except Exception as e:
            logger.error(f"Failed to save cost tracking: {e}")
    
    def print_summary(self):
        """Print cost summary."""
        costs = self.get_current_costs()
        
        print("\n" + "="*60)
        print("💰 KOSTEN-TRACKING ZUSAMMENFASSUNG")
        print("="*60)
        
        print(f"\n⏱️  Laufzeit: {costs['runtime_minutes']:.1f} Minuten")
        
        print(f"\n🔍 DataForSEO API:")
        print(f"  SERP Requests: {self.costs['dataforseo']['serp_requests']:,}")
        print(f"  Keyword Requests: {self.costs['dataforseo']['keyword_requests']:,}")
        print(f"  Backlink Requests: {self.costs['dataforseo']['backlink_requests']:,}")
        print(f"  On-Page Requests: {self.costs['dataforseo']['onpage_requests']:,}")
        print(f"  DataForSEO Gesamt: ${self.costs['dataforseo']['total_cost']:.2f}")
        
        print(f"\n🤖 OpenAI API:")
        print(f"  Requests: {self.costs['openai']['requests']:,}")
        print(f"  Input Tokens: {self.costs['openai']['input_tokens']:,}")
        print(f"  Output Tokens: {self.costs['openai']['output_tokens']:,}")
        print(f"  OpenAI Gesamt: ${self.costs['openai']['total_cost']:.2f}")
        
        print(f"\n💵 GESAMTKOSTEN: ${self.costs['total_cost']:.2f}")
        print("="*60)

def estimate_pipeline_costs(n_keywords: int = 100, serp_per_keyword: int = 40, 
                          articles_per_keyword: int = 80, features_per_article: int = 6,
                          model: str = "gpt-5-mini") -> Dict:
    """
    Estimate costs for full pipeline execution.
    
    Args:
        n_keywords: Number of keywords
        serp_per_keyword: SERP results per keyword
        articles_per_keyword: Articles to analyze per keyword
        features_per_article: LLM features per article
    
    Returns:
        Cost estimation
    """
    pricing = APIPricing()
    
    # DataForSEO costs
    total_serp_requests = n_keywords * serp_per_keyword
    dataforseo_cost = total_serp_requests * pricing.dataforseo_serp_task
    
    # OpenAI costs (estimated tokens)
    total_articles = n_keywords * articles_per_keyword
    total_llm_requests = total_articles * features_per_article
    
    # Estimate tokens per request (based on our prompts)
    avg_input_tokens = 2000  # Title + body excerpt
    avg_output_tokens = 50   # JSON response
    
    total_input_tokens = total_llm_requests * avg_input_tokens
    total_output_tokens = total_llm_requests * avg_output_tokens
    
    if model.startswith("gpt-5-mini"):
        openai_cost = (
            (total_input_tokens / 1000) * pricing.openai_gpt5mini_input_k +
            (total_output_tokens / 1000) * pricing.openai_gpt5mini_output_k
        )
    elif model.startswith("gpt-4-turbo"):
        openai_cost = (
            (total_input_tokens / 1000) * pricing.openai_gpt4_turbo_input +
            (total_output_tokens / 1000) * pricing.openai_gpt4_turbo_output
        )
    else:
        openai_cost = (
            (total_input_tokens / 1000) * pricing.openai_gpt4_input +
            (total_output_tokens / 1000) * pricing.openai_gpt4_output
        )
    
    total_cost = dataforseo_cost + openai_cost
    
    return {
        'parameters': {
            'keywords': n_keywords,
            'serp_per_keyword': serp_per_keyword,
            'articles_per_keyword': articles_per_keyword,
            'features_per_article': features_per_article,
            'total_articles': total_articles,
            'total_llm_requests': total_llm_requests
        },
        'dataforseo': {
            'serp_requests': total_serp_requests,
            'cost': dataforseo_cost
        },
        'openai': {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'requests': total_llm_requests,
            'cost': openai_cost
        },
        'total_cost': total_cost,
        'cost_per_keyword': total_cost / n_keywords,
        'cost_per_article': total_cost / total_articles
    }

def print_cost_estimation(estimation: Dict):
    """Print detailed cost estimation."""
    print("\n" + "="*70)
    print("💰 KOSTEN-SCHÄTZUNG FÜR VOLLSTÄNDIGE PIPELINE")
    print("="*70)
    
    params = estimation['parameters']
    print(f"\n📊 Parameter:")
    print(f"  Keywords: {params['keywords']:,}")
    print(f"  SERP pro Keyword: {params['serp_per_keyword']:,}")
    print(f"  Artikel pro Keyword: {params['articles_per_keyword']:,}")
    print(f"  Features pro Artikel: {params['features_per_article']:,}")
    print(f"  Gesamt Artikel: {params['total_articles']:,}")
    print(f"  Gesamt LLM Requests: {params['total_llm_requests']:,}")
    
    print(f"\n🔍 DataForSEO API:")
    print(f"  SERP Requests: {estimation['dataforseo']['serp_requests']:,}")
    print(f"  Kosten: ${estimation['dataforseo']['cost']:.2f}")
    
    print(f"\n🤖 OpenAI API:")
    print(f"  Input Tokens: {estimation['openai']['input_tokens']:,}")
    print(f"  Output Tokens: {estimation['openai']['output_tokens']:,}")
    print(f"  Requests: {estimation['openai']['requests']:,}")
    print(f"  Kosten: ${estimation['openai']['cost']:.2f}")
    
    print(f"\n💵 GESAMTKOSTEN: ${estimation['total_cost']:.2f}")
    print(f"  Kosten pro Keyword: ${estimation['cost_per_keyword']:.2f}")
    print(f"  Kosten pro Artikel: ${estimation['cost_per_article']:.4f}")
    
    print("\n⚠️  WICHTIGE HINWEISE:")
    print("  - Preise basieren auf aktuellen API-Tarifen (2024)")
    print("  - Token-Schätzungen basieren auf durchschnittlichen Prompt-Längen")
    print("  - Tatsächliche Kosten können um ±20% variieren")
    print("  - DataForSEO Mindestguthaben: $50")
    print("="*70)

# Global cost tracker instance
cost_tracker = CostTracker()

if __name__ == "__main__":
    # Test cost estimation
    estimation = estimate_pipeline_costs(n_keywords=100)
    print_cost_estimation(estimation)
    
    # Test with different scenarios
    print("\n" + "="*50)
    print("KOSTEN-VERGLEICH VERSCHIEDENER SZENARIEN")
    print("="*50)
    
    scenarios = [
        ("Minimal (20 Keywords)", 20, 20, 40, 6),
        ("Klein (50 Keywords)", 50, 30, 50, 6),
        ("Standard (100 Keywords)", 100, 40, 80, 6),
        ("Groß (200 Keywords)", 200, 40, 80, 6)
    ]
    
    for name, n_kw, serp, articles, features in scenarios:
        est = estimate_pipeline_costs(n_kw, serp, articles, features)
        print(f"\n{name}:")
        print(f"  Keywords: {n_kw}, Kosten: ${est['total_cost']:.2f}")
        print(f"  Pro Keyword: ${est['cost_per_keyword']:.2f}")
