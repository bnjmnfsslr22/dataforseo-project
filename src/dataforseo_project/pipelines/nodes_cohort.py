import datetime as dt
import json
import os
import time
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth

ADV = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
TASK_POST = "https://api.dataforseo.com/v3/serp/google/organic/task_post"
TASK_GET_ADV = "https://api.dataforseo.com/v3/serp/google/organic/task_get/advanced/{task_id}"
TASK_GET_ADV_BY_TAG = "https://api.dataforseo.com/v3/serp/google/organic/task_get/advanced"
MAX_RETRIES = int(os.getenv("DATAFORSEO_MAX_RETRIES", "3"))
REQUEST_TIMEOUT = int(os.getenv("DATAFORSEO_TIMEOUT_SECONDS", "30"))


def _is_article_content(url: str, title: str) -> bool:
    """
    Filter for article content only (scientific validity).
    Only excludes obvious non-article domains, keeps all text content.
    """
    if not url or not title:
        return False
    
    # Only exclude obvious non-article domains
    exclude_domains = [
        'youtube.com', 'youtu.be', 'vimeo.com',
        'instagram.com', 'facebook.com', 'twitter.com', 'tiktok.com',
        'amazon.', 'ebay.', 'shop.', 'buy.',
        'images.google.', 'google.com/images',
        'maps.google.', 'google.com/maps',
        'news.google.', 'google.com/news'
    ]
    
    url_lower = url.lower()
    for domain in exclude_domains:
        if domain in url_lower:
            return False
    
    # Keep all text content - no title filtering
    # This ensures we don't exclude articles about videos/images
    return True


def get_serp_cohort(seed_df: pd.DataFrame, dfs_params: dict, cohort_params: dict) -> pd.DataFrame:
    """
    Holt pro Keyword die Top-100 SERP-Items (rank_absolute, url, title).
    Kein target, komplette Kohorte für faire Vergleiche innerhalb des Keywords.
    """
    # Fallback zu Umgebungsvariablen wenn nicht in params
    username = dfs_params.get("username") or os.getenv("DATAFORSEO_USERNAME")
    password = dfs_params.get("password") or os.getenv("DATAFORSEO_PASSWORD")
    
    if not username or not password:
        raise ValueError("DataForSEO credentials not found in parameters or environment variables")
    
    auth = HTTPBasicAuth(username, password)
    # Extract parameters - handle both direct params and nested structure
    loc = dfs_params.get("location_name", "Germany")
    lang = dfs_params.get("language_name", "German")
    # Enforce maximum results per keyword (default 40)
    max_per_kw = int(dfs_params.get("results_per_keyword", 40))
    now = dt.datetime.utcnow().isoformat()
    rows = []

    # Accept dict output from keyword generator and convert to DataFrame
    if isinstance(seed_df, dict):
        kw_items = seed_df.get("keywords", [])
        if isinstance(kw_items, list):
            try:
                seed_df = pd.DataFrame([
                    {"keyword": (it.get("keyword") if isinstance(it, dict) else str(it))}
                    for it in kw_items
                ])
            except Exception:
                seed_df = pd.DataFrame({"keyword": []})
        else:
            seed_df = pd.DataFrame({"keyword": []})

    if "keyword" not in seed_df.columns:
        raise KeyError("keyword")

    use_task_api = bool(dfs_params.get("use_task_api", False))
    keywords = list(seed_df["keyword"].dropna().unique())
    if use_task_api:
        # Batch Task-API with tag-based polling for robustness
        run_tag = f"kedro_seed_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        task_payloads = []
        for kw in keywords:
            task_payloads.append({
                "keyword": kw,
                "location_name": loc,
                "language_name": lang,
                "device": "desktop",
                "depth": int(max_per_kw),
                "exclude_serp_element_types": ["video", "image", "shopping"],
                "tag": run_tag,
                "exclude_serp_element_types": ["video", "image", "shopping"]
            })
        # Post tasks in smaller batches to avoid rate limits
        batch_size = int(dfs_params.get("task_batch_size", 10))
        batch_pause = float(dfs_params.get("task_batch_pause_seconds", 2.0))
        for i in range(0, len(task_payloads), batch_size):
            batch = task_payloads[i:i+batch_size]
            try:
                pr = requests.post(TASK_POST, json=batch, auth=auth, timeout=REQUEST_TIMEOUT)
                if pr.status_code != 200:
                    print(f"Warning: task_post batch HTTP {pr.status_code} | headers={dict(pr.headers)} | body={pr.text[:500]}")
            except Exception as e:
                print(f"Warning: task_post batch failed: {e}")
            time.sleep(batch_pause)

        # Poll by tag until we get results or timeout
        collected = set()
        desired = int(max_per_kw) if max_per_kw else 20
        poll_cycles = int(dfs_params.get("poll_max_cycles", MAX_RETRIES * 20))
        for attempt in range(1, poll_cycles + 1):
            try:
                # Try GET by tag
                gr = requests.get(TASK_GET_ADV_BY_TAG, params={"tag": run_tag}, auth=auth, timeout=REQUEST_TIMEOUT)
                if gr.status_code == 200:
                    data = gr.json()
                else:
                    # Fallback: POST by tag
                    gr2 = requests.post(TASK_GET_ADV_BY_TAG, json=[{"tag": run_tag}], auth=auth, timeout=REQUEST_TIMEOUT)
                    if gr2.status_code != 200:
                        print(f"Warning: task_get by tag HTTP {gr.status_code}/{gr2.status_code} | headers={dict(gr.headers)} | body={gr.text[:300]}")
                        time.sleep(min(5.0, 1.0 + 0.5 * attempt))
                        continue
                    data = gr2.json()
                tasks = data.get("tasks", [])
                for t in tasks:
                    tsc = t.get("status_code")
                    if tsc != 20000:
                        msg = t.get("status_message") or t.get("result", [{}])[0].get("status_message") if t.get("result") else None
                        print(f"Info: task not ready status_code={tsc} message={msg}")
                        continue
                    result = t.get("result") or []
                    items = result[0].get("items", []) if result else []
                    kw = (result[0].get("keyword") if result else None) or t.get("data", {}).get("keyword")
                    if not kw:
                        continue
                    # if already complete, skip
                    if kw in collected:
                        continue
                    if max_per_kw and isinstance(items, list):
                        items = items[:max_per_kw]
                    for it in items:
                        urlv = it.get("url")
                        rank = it.get("rank_absolute")
                        title = it.get("title")
                        if not urlv or not rank:
                            continue
                        rows.append({
                            "keyword": kw,
                            "url": urlv,
                            "rank_absolute": int(rank),
                            "title_snippet": title or "",
                            "captured_at_utc": now
                        })
                    # mark collected only when we have enough FILTERED items
                    filtered_items = [it for it in items if _is_article_content(it.get("url"), it.get("title"))]
                    if filtered_items and len(filtered_items) >= desired:
                        collected.add(kw)
                # Break if we have results for all keywords
                if len(collected) >= len(keywords):
                    break
            except Exception:
                pass
            # Live progress
            try:
                total = len(keywords)
                done = len(collected)
                pct = (done / total * 100.0) if total else 0.0
                status = {
                    "timestamp": dt.datetime.utcnow().isoformat(),
                    "attempt": attempt,
                    "keywords_total": total,
                    "keywords_done": done,
                    "percent": round(pct, 1),
                    "remaining_samples": sorted([k for k in keywords if k not in collected])[:10]
                }
                print(f"\r🔎 SERP polling: {done}/{total} ({pct:.1f}%)", end="", flush=True)
                with open("logs/serp_progress.json", "w", encoding="utf-8") as f:
                    json.dump(status, f, ensure_ascii=False, indent=2)
            except Exception:
                pass
            time.sleep(min(8.0, 1.0 + 0.5 * attempt))

        print()  # newline after progress

        # For any missing keywords, print warnings and fallback to live endpoint
        missing = [k for k in keywords if k not in collected]
        for k in missing:
            print(f"Warning: Polling failed for keyword '{k}' (tag={run_tag})")
        # Fallback: try live ADV for missing keywords to avoid empty cohort
        for kw in missing:
            payload = [{
                "keyword": kw,
                "location_name": loc,
                "language_name": lang,
                "device": "desktop",
                "depth": int(max_per_kw),
                "exclude_serp_element_types": ["video", "image", "shopping"]
            }]
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    r = requests.post(ADV, json=payload, auth=auth, timeout=REQUEST_TIMEOUT)
                    if r.status_code != 200:
                        print(f"Warning: live ADV HTTP {r.status_code} | headers={dict(r.headers)} | body={r.text[:300]}")
                        raise requests.exceptions.RequestException(f"HTTP {r.status_code}")
                    data = r.json()
                    sc = data.get("status_code") or data.get("tasks", [{}])[0].get("status_code")
                    if sc != 20000:
                        print(f"Warning: live ADV API status {sc} | body={str(data)[:300]}")
                        raise requests.exceptions.RequestException(f"API status {sc}")
                    result = data["tasks"][0].get("result") or []
                    items = result[0].get("items", []) if result else []
                    if max_per_kw and isinstance(items, list):
                        items = items[:max_per_kw]
                    for it in items:
                        urlv = it.get("url")
                        rank = it.get("rank_absolute")
                        title = it.get("title")
                        if not urlv or not rank:
                            continue
                        rows.append({
                            "keyword": kw,
                            "url": urlv,
                            "rank_absolute": int(rank),
                            "title_snippet": title or "",
                            "captured_at_utc": now
                        })
                    break
                except Exception:
                    time.sleep(min(3.0, 1.0 + 0.5 * attempt))
                    continue

    else:
        for kw in keywords:
            # Live endpoint path
            payload = [{
                "keyword": kw,
                "location_name": loc,
                "language_name": lang,
                "device": "desktop",
                "depth": int(max_per_kw),
                "exclude_serp_element_types": ["video", "image", "shopping"]
            }]
            success = False
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    print(f"DEBUG: Making API call for '{kw}' (attempt {attempt}/{MAX_RETRIES})")
                    r = requests.post(ADV, json=payload, auth=auth, timeout=REQUEST_TIMEOUT)
                    print(f"DEBUG: API call completed for '{kw}' - status: {r.status_code}")
                    if r.status_code != 200:
                        print(f"HTTP {r.status_code} | Headers: {dict(r.headers)} | Body: {r.text[:200]}")
                        raise requests.exceptions.RequestException(f"HTTP {r.status_code}")
                    data = r.json()
                    sc = data.get("status_code") or data.get("tasks", [{}])[0].get("status_code")
                    if sc != 20000:
                        print(f"API Status {sc} | Message: {data.get('status_message', 'N/A')} | Body: {str(data)[:200]}")
                        raise requests.exceptions.RequestException(f"API status {sc}")
                    result = data["tasks"][0].get("result") or []
                    items = result[0].get("items", []) if result else []
                    if max_per_kw and isinstance(items, list):
                        items = items[:max_per_kw]
                    for it in items:
                        url = it.get("url")
                        rank = it.get("rank_absolute")
                        title = it.get("title")
                        if not url or not rank:
                            continue
                        if not _is_article_content(url, title or ""):
                            continue
                        rows.append({
                            "keyword": kw,
                            "url": url,
                            "rank_absolute": int(rank),
                            "title_snippet": title or "",
                            "captured_at_utc": now
                        })
                    success = True
                    break
                except requests.exceptions.Timeout:
                    print(f"Warning: Timeout for keyword '{kw}' (try {attempt}/{MAX_RETRIES}) - API took too long (> {REQUEST_TIMEOUT}s)")
                except requests.exceptions.RequestException as e:
                    print(f"Warning: Request error for keyword '{kw}' (try {attempt}/{MAX_RETRIES}): {e}")
                    # Check for rate limiting
                    if "429" in str(e) or "403" in str(e):
                        print(f"RATE LIMIT DETECTED for '{kw}' - waiting longer...")
                        time.sleep(10)
                    # Check for connection issues
                    if "Connection" in str(e) or "Network" in str(e):
                        print(f"NETWORK ISSUE for '{kw}' - waiting before retry...")
                        time.sleep(5)  # Wait longer for rate limits
                except Exception as e:
                    print(f"Warning: Unexpected error for keyword '{kw}' (try {attempt}/{MAX_RETRIES}): {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(1.5 ** attempt)
            if not success:
                print(f"Warning: Skipping keyword '{kw}' after {MAX_RETRIES} failed attempts")

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["keyword","url","rank_absolute","title_snippet","captured_at_utc"]) 
    # Deduplicate and sort for determinism
    df = df.drop_duplicates(subset=["keyword", "url"])  # ensure unique URLs per keyword
    try:
        df["rank_absolute"] = df["rank_absolute"].astype(int)
    except Exception:
        pass
    df = df.sort_values(["keyword", "rank_absolute"])  # keep top ranks first

    # Optional strict mode: keep ONLY keywords that have at least max_per_kw results
    strict = bool(dfs_params.get("strict_results_per_keyword", False))
    if strict and max_per_kw:
        counts = df.groupby("keyword").size()
        valid_keywords = set(counts[counts >= max_per_kw].index)
        df = df[df["keyword"].isin(valid_keywords)]
        # Enforce exact top max_per_kw per keyword
        df = df.groupby("keyword", as_index=False, group_keys=False).head(max_per_kw)

    # Optional enforcement of target number of keywords (scientific consistency)
    enforce_kw_count = bool(dfs_params.get("enforce_keyword_count", False))
    target_keywords = int(dfs_params.get("target_keywords", 100))
    if enforce_kw_count:
        got = int(df["keyword"].nunique())
        if got < target_keywords:
            raise ValueError(
                f"SERP cohort has only {got} keywords with >= {max_per_kw} results; "
                f"target is {target_keywords}. Consider increasing retries/timeouts, "
                f"adding more seed keywords, or disabling strict mode."
            )
        # If more than target, keep first N keywords deterministically
        if got > target_keywords:
            keep = list(df["keyword"].drop_duplicates().sort_values())[:target_keywords]
            df = df[df["keyword"].isin(keep)]

    return df
