import re, pandas as pd
import logging

logger = logging.getLogger(__name__)

STOP = {"der","die","das","ein","eine","und","oder","im","in","am","vom","von","mit","für","auf","an","den","dem","des","zur","zum","gegen","nach"}
TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")

def _tokens(text: str):
    return [w.lower() for w in TOKEN_RE.findall(text or "")]

def _jaccard(a, b):
    A, B = set(a), set(b)
    return len(A & B) / max(1, len(A | B))

def _top_tokens(text: str, n: int = 5):
    words = [w for w in _tokens(text) if w not in STOP]
    freq = {}
    for w in words: freq[w] = freq.get(w, 0) + 1
    return [w for w,_ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]

def _readability(text: str):
    sents = re.split(r"[.!?]", text or "")
    sents = [s for s in sents if s.strip()]
    words = _tokens(text or "")
    if not words:
        return 0.0  # Leerer Text ist nicht lesbar (niedrigster Wert, nicht mittlerer)
    avg_w = sum(len(w) for w in words)/len(words)
    avg_s = (len(words)/len(sents)) if sents else len(words)
    score = 1.0/(1.0 + 0.1*avg_w + 0.02*avg_s)
    return max(0.0, min(1.0, score))

def extract_struct_features_cohort(content_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract structural features from article content.
    Only processes URLs with non-empty body content.
    """
    rows = []
    skipped = 0
    processed = 0
    
    for r in content_df.itertuples():
        # Skip URLs without content
        body = r.body or ""
        if not body or len(body.strip()) == 0:
            skipped += 1
            continue
        
        title_tokens = _top_tokens(getattr(r, "title_snippet", "") or "", 5)
        body_tokens = _top_tokens(body, 20)
        jac = _jaccard(title_tokens, body_tokens)
        body_all = _tokens(body)
        body_len = len(body_all)
        if title_tokens and body_len:
            hits = sum(1 for t in set(title_tokens) if t in set(body_all))
            kd = min(0.05, hits / max(1, body_len))
        else:
            kd = 0.0
        rows.append({
            "keyword": r.keyword,
            "url": r.url,
            "keyword_alignment_struct": jac,
            "keyword_density_struct": kd,
            "readability_struct": _readability(body),
            "body_word_count": body_len,
            "url_depth": r.url_depth,
        })
        processed += 1
    
    # Log summary
    logger.info(f"Structural features extracted: {processed} URLs processed, {skipped} URLs skipped (empty body)")
    
    return pd.DataFrame(rows)
