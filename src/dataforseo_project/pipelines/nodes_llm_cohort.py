import os, json, time, pandas as pd, openai
from typing import Dict
import logging

logger = logging.getLogger(__name__)

# OpenAI API Key aus Umgebungsvariablen laden
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
openai.api_key = api_key

SYSTEM_PROMPT = (
  "Du agierst als wissenschaftlich geschulter Rater für Inhaltsqualität. "
  "Ziel ist hohe Reliabilität und Replizierbarkeit. Verwende eine einheitliche Bewertungsrubrik mit Skalenankern "
  "und halte Dich strikt an das Ausgabeschema. Gib ausschließlich ein einzelnes JSON-Objekt zurück, ohne Markdown oder Erklärungen. "
  "Keine Spekulationen oder unberündeten Annahmen; bewerte nur auf Basis des bereitgestellten Titels und Textauszugs."
)

PROMPT = (
  "Du bist ein wissenschaftlicher Gutachter. Bewerte den Artikel-Ausschnitt streng nach den Definitionen. "
  "Gib ausschließlich JSON zurück (kein Fließtext). Erlaube Werte im [0,1]-Intervall mit drei Dezimalstellen.\n\n"
  "Definitionen:\n"
  "- headline_quality: Wahrnehmte Qualität der Überschrift (Klarheit, Präzision, Relevanz).\n"
  "- topic_coherence: Semantische Kohärenz zwischen Überschrift und Inhalt.\n"
  "- clickbait_score: Grad des Clickbaits (höher = stärkerer Clickbait).\n"
  "- information_density: Dichte an informativen Aussagen relativ zur Textlänge.\n"
  "- entity_richness: Vielfalt und Dichte benannter Entitäten (Personen/Organisationen/Orte).\n"
  "- keyword_alignment: Semantische Ausrichtung auf das Ziel-Keyword des Artikels.\n\n"
  "Skalenanker (alle Dimensionen): 0.000=stark defizitär; 0.250=unterdurchschnittlich; 0.500=mittel; 0.750=gut; 1.000=exzellent.\n"
  "Kalibrierung: Vermeide Extrema ohne starke Evidenz im Text; runde immer auf drei Dezimalstellen.\n\n"
  "Ausgabeschema (strikt): Ein einzelnes JSON-Objekt mit genau diesen Schlüsseln und Float-Werten in [0,1]:\n"
  "{{\n  \"headline_quality\": float,\n  \"topic_coherence\": float,\n  \"clickbait_score\": float,\n  \"information_density\": float,\n  \"entity_richness\": float,\n  \"keyword_alignment\": float\n}}\n\n"
  "Titel: \"{title}\"\n\nTextauszug:\n\"\"\"{excerpt}\"\"\"\n\nJSON:"
)
SAFE = ["headline_quality","topic_coherence","clickbait_score","information_density","entity_richness","keyword_alignment"]

def _truncate(txt: str, max_chars: int = 20000):
    if not txt: return ""
    if len(txt) <= max_chars: return txt
    half = max_chars // 2
    return txt[:half] + " … " + txt[-half:]

def llm_features_cohort(content_df: pd.DataFrame, model: str = "gpt-5-mini", max_calls: int = 300) -> pd.DataFrame:
    """Extract LLM features with live token tracking."""
    logger.info(f"Starting LLM feature extraction with {model}")
    logger.info(f"Processing {len(content_df)} articles (max {max_calls})")
    
    # Token tracking
    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    
    # GPT-5-mini Preise (pro 1M tokens)
    input_cost_per_1m = 0.25   # $0.25 per 1M input tokens
    output_cost_per_1m = 2.00  # $2.00 per 1M output tokens
    
    # Incremental checkpointing
    cache_file = os.getenv("LLM_CACHE_FILE") or "data/02_intermediate/llm_features_cache.csv"
    existing_parquet = "data/02_intermediate/cohort_llm_features.parquet"
    
    processed = set()
    existing_df = None
    
    # Load existing Parquet file if it exists (from run_llm_all_articles.py or previous runs)
    # WICHTIG: String-Konvertierung für konsistenten Vergleich (wie in run_llm_all_articles.py)
    if os.path.exists(existing_parquet):
        try:
            existing_df = pd.read_parquet(existing_parquet)
            for t in existing_df.itertuples():
                kw = str(getattr(t, "keyword", None)) if getattr(t, "keyword", None) is not None else None
                url = str(getattr(t, "url", None)) if getattr(t, "url", None) is not None else None
                processed.add((kw, url))
            logger.info(f"Found existing Parquet file with {len(existing_df)} rows: {existing_parquet}")
        except Exception:
            logger.warning("Could not read existing Parquet file; proceeding without resume")
            existing_df = None
    
    # Also load CSV cache (for incremental checkpointing)
    if os.path.exists(cache_file):
        try:
            cache_df = pd.read_csv(cache_file)
            for t in cache_df.itertuples():
                kw = str(getattr(t, "keyword", None)) if getattr(t, "keyword", None) is not None else None
                url = str(getattr(t, "url", None)) if getattr(t, "url", None) is not None else None
                processed.add((kw, url))
            logger.info(f"Resuming with {len(processed)} already cached rows (from cache + parquet)")
        except Exception:
            logger.warning("Could not read cache file; proceeding without resume")
    
    rows = []
    calls = 0
    total_articles = min(len(content_df), max_calls)
    
    try:
        for i, r in enumerate(content_df.itertuples()):
            if calls >= total_articles: break
            # WICHTIG: String-Konvertierung für konsistenten Vergleich (wie in run_llm_all_articles.py)
            kw = str(getattr(r, "keyword", None)) if getattr(r, "keyword", None) is not None else None
            url = str(getattr(r, "url", None)) if getattr(r, "url", None) is not None else None
            key = (kw, url)
            if key in processed:
                continue
            title = getattr(r, "title_snippet", "") or ""
            excerpt = _truncate(r.body or "")
            prompt = PROMPT.format(title=title, excerpt=excerpt)
            try:
                # Prefer strict JSON mode if supported by the SDK/model
                try:
                    resp = openai.chat.completions.create(
                        model=model,
                        messages=[
                            {"role":"system","content":SYSTEM_PROMPT},
                            {"role":"user","content":prompt}
                        ],
                        temperature=1,
                        response_format={"type": "json_object"}
                    )
                except TypeError:
                    # Fallback for older SDKs without response_format
                    resp = openai.chat.completions.create(
                        model=model,
                        messages=[
                            {"role":"system","content":SYSTEM_PROMPT},
                            {"role":"user","content":prompt}
                        ],
                        temperature=1
                    )
                
                # Track tokens (offizielle OpenAI API Response - 100% exakt)
                input_tokens = resp.usage.prompt_tokens
                output_tokens = resp.usage.completion_tokens
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Calculate cost
                input_cost = (input_tokens / 1_000_000) * input_cost_per_1m
                output_cost = (output_tokens / 1_000_000) * output_cost_per_1m
                total_cost += input_cost + output_cost
                
                txt = resp.choices[0].message.content.strip()
                if txt.startswith("```"):
                    txt = txt.strip("`")
                    j = txt.find("{")
                    txt = txt[j:]
                data = json.loads(txt)
            except openai.APIError as e:
                print(f"Warning: OpenAI API error for URL {r.url}: {e}")
                continue
            except json.JSONDecodeError as e:
                print(f"Warning: JSON decode error for URL {r.url}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Unexpected error for URL {r.url}: {e}")
                continue
            rec = {"keyword": getattr(r, "keyword"), "url": getattr(r, "url")}
            for k in SAFE:
                v = data.get(k)
                try:
                    v = float(v) if v is not None else None
                    if v is not None:
                        if v < 0: v = 0.0
                        if v > 1: v = 1.0
                        v = round(v, 3)
                except Exception:
                    v = None
                rec[k] = v
            rows.append(rec)
            # Incremental write to cache
            try:
                df_append = pd.DataFrame([rec])
                write_header = not os.path.exists(cache_file)
                df_append.to_csv(cache_file, mode="a", header=write_header, index=False)
                processed.add(key)
            except Exception:
                pass
            calls += 1
            time.sleep(0.05)
            
            # Live progress with cost tracking
            if calls % 5 == 0 or calls == total_articles:
                progress = (calls / total_articles) * 100
                estimated_total_cost = (total_cost / calls) * total_articles if calls > 0 else 0
                print(f"\r🔄 Progress: {calls}/{total_articles} ({progress:.1f}%) | "
                      f"💰 Current: ${total_cost:.4f} | "
                      f"📊 Estimated Total: ${estimated_total_cost:.4f} | "
                      f"🔢 Tokens: {total_input_tokens:,} in, {total_output_tokens:,} out",
                      end="", flush=True)
    except KeyboardInterrupt:
        print("\n⏹️ Abbruch erkannt. Teil-Ergebnisse wurden im Cache gesichert und werden zurückgegeben.")
        # proceed to return partial results
    
    print()  # New line after progress
    logger.info("LLM feature extraction completed")
    
    # Print final summary
    print(f"\n📊 Final Summary:")
    print(f"   Requests: {calls}")
    print(f"   Input Tokens: {total_input_tokens:,}")
    print(f"   Output Tokens: {total_output_tokens:,}")
    print(f"   Total Cost: ${total_cost:.4f}")
    print(f"   Avg per Request: ${total_cost/calls:.4f}" if calls > 0 else "   Avg per Request: $0.0000")

    # Merge new results with existing Parquet file (genau wie run_llm_all_articles.py)
    if rows:
        new_df = pd.DataFrame(rows)
        # WICHTIG: String-Konvertierung für konsistenten Merge (wie in run_llm_all_articles.py)
        new_df['keyword'] = new_df['keyword'].astype(str)
        new_df['url'] = new_df['url'].astype(str)
        
        if existing_df is not None and len(existing_df) > 0:
            # String-Konvertierung auch für existing_df
            existing_df = existing_df.copy()
            existing_df['keyword'] = existing_df['keyword'].astype(str)
            existing_df['url'] = existing_df['url'].astype(str)
            
            # Merge: combine existing and new, remove duplicates (keep new if duplicate)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["keyword", "url"], keep="last")
            logger.info(f"Merged {len(new_df)} new rows with {len(existing_df)} existing rows → {len(combined)} total")
            return combined
        else:
            return new_df
    else:
        # No new rows processed - return existing if available, otherwise empty
        if existing_df is not None and len(existing_df) > 0:
            logger.info(f"No new articles processed, returning existing {len(existing_df)} rows from Parquet")
            return existing_df
        else:
            return pd.DataFrame(columns=["keyword", "url", *SAFE])
