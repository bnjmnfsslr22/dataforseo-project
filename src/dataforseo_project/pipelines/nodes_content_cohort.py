import time, re, requests, pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urlparse

HEADERS = {"User-Agent": "Mozilla/5.0 (SEO)"}

def _clean_text(txt: str) -> str:
    return re.sub(r"\s+", " ", txt or "").strip()

def _extract_main_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script","style","noscript"]):
        tag.decompose()
    paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
    paras = [p for p in paras if len(p.split()) > 3]
    body = " ".join(paras) if paras else soup.get_text(" ", strip=True)
    return _clean_text(body)

def fetch_cohort_content(serp_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in serp_df.itertuples():
        url = r.url
        body = ""
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            if resp.status_code == 200:
                body = _extract_main_text(resp.text)
        except Exception:
            body = ""
        url_depth = max(0, urlparse(url).path.count("/") - 1)
        rows.append({
            "keyword": r.keyword,
            "url": url,
            "rank_absolute": int(r.rank_absolute),
            "title_snippet": getattr(r, "title_snippet", "") or "",
            "body": body,
            "url_depth": url_depth
        })
        time.sleep(0.2)
    # Ensure schema even when empty
    if not rows:
        return pd.DataFrame(columns=["keyword","url","rank_absolute","title_snippet","body","url_depth"])
    return pd.DataFrame(rows)
