"""
FastAPI backend for Museum Semantic Search.

Start with:
    uvicorn server:app --reload --port 8000
"""

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
COMBINED_FILE = BASE / "combined_collection.json"
EMBEDDINGS_FILE = BASE / "embeddings.npy"
INDEX_FILE = BASE / "embeddings_index.json"
HARVARD_CACHE_DIR = BASE / "harvard_image_cache"

# ── Museum label mapping ──────────────────────────────────────────────────
MUSEUM_LABELS = {
    "met": "The Met",
    "va": "V&A",
    "aic": "Art Institute",
    "rijks": "Rijksmuseum",
    "jps": "Japan Search",
    "harvard": "Harvard",
    "europeana": "Europeana",
    "npm": "NPM 故宮",
}

# ── Quality weighting (from merge_and_embed.py) ──────────────────────────
NOISY_TITLE_TOKENS = {"download", "presentation", "lower", "image size"}


def compute_quality_weight(richness: int, title: str) -> float:
    if richness >= 6:
        weight = 1.00
    elif richness == 5:
        weight = 0.95
    elif richness == 4:
        weight = 0.85
    elif richness == 3:
        weight = 0.70
    else:
        weight = 0.50

    t = (title or "").strip().lower()
    if not t or t == "?" or any(tok in t for tok in NOISY_TITLE_TOKENS):
        weight *= 0.5

    return max(weight, 0.30)


def cosine_similarity_matrix(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
    normed = matrix / norms
    return normed @ query_norm


# ── Noisy record filters ─────────────────────────────────────────────────
NPM_NOISY_TOKENS = {"download", "verification", "presentation size", "error replay", "refresh"}


def _is_noisy_record(record: dict) -> bool:
    """Filter out NPM records with scraper-artifact titles."""
    if record.get("source_museum_code") != "npm":
        return False
    title = (record.get("title") or "").lower()
    return any(tok in title for tok in NPM_NOISY_TOKENS)


# ── Load data at startup ─────────────────────────────────────────────────
print("Loading combined_collection.json ...")
with open(COMBINED_FILE, encoding="utf-8") as f:
    _collection = json.load(f)
_raw_records = _collection["records"]
_filtered = [r for r in _raw_records if not _is_noisy_record(r)]
print(f"Filtered out {len(_raw_records) - len(_filtered)} noisy NPM records")
ALL_RECORDS = _filtered
RECORDS_BY_ID = {r["id"]: r for r in ALL_RECORDS}

print("Loading embeddings.npy ...")
EMBEDDINGS_MATRIX = np.load(EMBEDDINGS_FILE)

print("Loading embeddings_index.json ...")
with open(INDEX_FILE, encoding="utf-8") as f:
    EMBEDDINGS_INDEX = json.load(f)

# Build richness lookup
RICHNESS_LOOKUP = {r["id"]: r.get("metadata_richness", 5) for r in ALL_RECORDS}

print(f"Ready: {len(ALL_RECORDS)} records, embeddings shape {EMBEDDINGS_MATRIX.shape}")


# ── Helpers ───────────────────────────────────────────────────────────────

# Matches /full/full/ or /full/1400,400/ or /full/!400,400/ etc.
_HARVARD_IIIF_RE = re.compile(r"/full/(?:full|!?\d+,\d*)/0/")


def _shrink_harvard_image(url: str) -> str:
    """Downsize Harvard IIIF URLs to 400px wide thumbnails."""
    if "ids.lib.harvard.edu" in url:
        return _HARVARD_IIIF_RE.sub("/full/!400,400/0/", url)
    return url


def format_artwork(record: dict, score: float = None) -> dict:
    """Convert an ArtworkRecord dict to the frontend format."""
    code = record.get("source_museum_code", "")
    image = record.get("image_url_small") or record.get("image_url", "")
    image = _shrink_harvard_image(image)
    # Use local cache for Harvard images if available
    if code == "harvard":
        cached = HARVARD_CACHE_DIR / f"{record['id']}.jpg"
        if cached.exists():
            image = f"http://localhost:8000/harvard-cache/{record['id']}.jpg"
    result = {
        "id": record["id"],
        "title": record.get("title", ""),
        "artist": record.get("artist", ""),
        "date": record.get("date", ""),
        "culture": record.get("culture", ""),
        "museum": code,
        "museum_label": MUSEUM_LABELS.get(code, record.get("source_museum", code)),
        "image": image,
        "medium": record.get("medium", ""),
    }
    if score is not None:
        result["score"] = round(float(score), 4)
    return result


# Filter artworks that have images
ARTWORKS_WITH_IMAGES = [
    r for r in ALL_RECORDS
    if r.get("has_image") and (r.get("image_url") or r.get("image_url_small"))
]
print(f"Artworks with images: {len(ARTWORKS_WITH_IMAGES)}")

# Group by museum for diverse sampling
_BY_MUSEUM: dict[str, list] = {}
for r in ARTWORKS_WITH_IMAGES:
    _BY_MUSEUM.setdefault(r.get("source_museum_code", ""), []).append(r)
print(f"Museums: {', '.join(f'{k}({len(v)})' for k, v in _BY_MUSEUM.items())}")


# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(title="Museum Semantic Search API")  # v2 cache fix

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Mount Harvard image cache as static files
if HARVARD_CACHE_DIR.exists():
    app.mount("/harvard-cache", StaticFiles(directory=str(HARVARD_CACHE_DIR)), name="harvard-cache")
    print(f"Mounted Harvard cache: {len(list(HARVARD_CACHE_DIR.glob('*.jpg')))} images")
else:
    print("No Harvard cache directory found — Harvard images will use proxy/IIIF")


@app.get("/")
async def homepage():
    """Serve the museum search frontend."""
    return FileResponse(BASE / "museum-search.html")


class SearchRequest(BaseModel):
    query: str
    museum: str = "all"


@app.get("/api/artworks")
def get_artworks():
    """Return a diverse random sample of 200 artworks across all museums."""
    sample_size = 200
    num_museums = len(_BY_MUSEUM)
    if num_museums == 0:
        return []
    per_museum = max(sample_size // num_museums, 5)
    sampled = []
    for code, records in _BY_MUSEUM.items():
        n = min(per_museum, len(records))
        sampled.extend(random.sample(records, n))
    # If we have room, fill remaining slots from all artworks
    remaining = sample_size - len(sampled)
    if remaining > 0:
        sampled_ids = {r["id"] for r in sampled}
        extras = [r for r in ARTWORKS_WITH_IMAGES if r["id"] not in sampled_ids]
        sampled.extend(random.sample(extras, min(remaining, len(extras))))
    random.shuffle(sampled)
    return [format_artwork(r) for r in sampled[:sample_size]]


# ── Retrieve-and-Rerank architecture ─────────────────────────────────────
#
# Search uses a two-stage pipeline:
#
# Stage 1 — RETRIEVE: Embed the query with OpenAI text-embedding-3-small,
#   compute cosine similarity against all 6,964 pre-computed artwork embeddings,
#   apply quality weighting, and return the top 150 candidates. This is fast
#   (~100ms) but only matches semantic fragments — it can't understand the
#   *relationships* between concepts in a query like "tension between humans
#   and nature" (it will return landscapes OR portraits, not both together).
#
# Stage 2 — RERANK: Send the 150 candidates to Claude (claude-sonnet-4-20250514)
#   with the original query. Claude reads each artwork's metadata and scores
#   how well the artwork matches the COMPLETE intent of the query, considering
#   concept relationships, not just keyword overlap. Results are re-sorted by
#   Claude's scores and the top 80 are returned.
#
# If ANTHROPIC_API_KEY is not set, Stage 2 is skipped and embedding-only
# results are returned (functional but lower quality).
# ─────────────────────────────────────────────────────────────────────────


def _build_artwork_summary(record: dict) -> str:
    """Build a concise text summary of an artwork for the re-ranker."""
    parts = []
    if record.get("title"):
        parts.append(f"Title: {record['title']}")
    if record.get("artist") and record["artist"] != "Unknown":
        parts.append(f"Artist: {record['artist']}")
    if record.get("date"):
        parts.append(f"Date: {record['date']}")
    if record.get("culture"):
        parts.append(f"Culture: {record['culture']}")
    if record.get("medium"):
        parts.append(f"Medium: {record['medium']}")
    if record.get("classification"):
        parts.append(f"Classification: {record['classification']}")
    if record.get("tags"):
        tags = record["tags"]
        if isinstance(tags, list):
            parts.append(f"Tags: {', '.join(tags[:10])}")
    if record.get("description"):
        desc = record["description"][:200]
        parts.append(f"Description: {desc}")
    return " | ".join(parts)


def _rerank_with_llm(query: str, candidates: list[tuple[dict, float]]) -> list[tuple[str, int, str, float]]:
    """
    Call Claude to re-rank candidates by query relevance.

    Returns list of (id, llm_score, reason, raw_cosine_score) sorted by llm_score desc.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return []

    client = anthropic.Anthropic(api_key=api_key)

    # Build artwork list for the prompt — split into batches of 75
    batch_size = 75
    all_results = []

    # Map id -> raw_score for later
    raw_scores = {rec["id"]: score for rec, score in candidates}

    for batch_start in range(0, len(candidates), batch_size):
        batch = candidates[batch_start:batch_start + batch_size]
        artwork_lines = []
        for i, (record, _score) in enumerate(batch):
            summary = _build_artwork_summary(record)
            artwork_lines.append(f'{i+1}. [ID: {record["id"]}] {summary}')

        artworks_text = "\n".join(artwork_lines)

        prompt = f"""You are an art curator evaluating how well each artwork matches a visitor's search intent.

The visitor searched for: "{query}"

For each artwork below, rate how well it matches the COMPLETE intent of the search query on a scale of 0-100.
Consider not just individual keywords but the RELATIONSHIPS between concepts in the query.
For example, "the quiet tension between humans and nature" means artworks showing humans AND nature TOGETHER with a sense of quiet tension — not just nature alone, not just humans alone.
"feminine beauty" means artworks depicting or evoking feminine beauty — not just any artwork by a female artist.

Score guide:
- 90-100: Directly and powerfully embodies the full query intent
- 70-89: Strongly related, captures most of the query's meaning
- 50-69: Partially relevant, matches some aspects
- 30-49: Tangentially related
- 0-29: Not relevant

Artworks:
{artworks_text}

Respond with ONLY a JSON array, no other text:
[{{"id": "artwork_id", "score": 85, "reason": "one sentence explanation"}}]"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8192,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text.strip()
            # Parse JSON — handle potential markdown code fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            batch_results = json.loads(text)
            for item in batch_results:
                aid = item["id"]
                all_results.append((
                    aid,
                    int(item.get("score", 0)),
                    item.get("reason", ""),
                    raw_scores.get(aid, 0.0),
                ))
        except Exception as e:
            print(f"Re-rank batch failed: {e}")
            # Fall back: include these candidates with score=0 so they
            # appear at the end rather than being lost
            for record, score in batch:
                all_results.append((record["id"], 0, "", score))

    # Sort by LLM score descending
    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results


@app.post("/api/search")
def search_artworks(req: SearchRequest):
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"error": "OPENAI_API_KEY not set"}

    client = OpenAI(api_key=api_key)

    # ── Stage 1: RETRIEVE via embedding similarity ──
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[req.query],
    )
    query_vec = np.array(response.data[0].embedding, dtype=np.float32)

    sims = cosine_similarity_matrix(query_vec, EMBEDDINGS_MATRIX)

    # Quality weighting
    scores = np.zeros(len(sims), dtype=np.float32)
    for i, item in enumerate(EMBEDDINGS_INDEX):
        rid = item["id"]
        richness = RICHNESS_LOOKUP.get(rid, 5)
        title = item.get("title", "")
        qw = compute_quality_weight(richness, title)
        scores[i] = sims[i] * qw

    top_indices = np.argsort(scores)[::-1]

    # Build candidate pool — top 150 for re-ranking (wider net than final 80)
    candidates = []
    for idx in top_indices:
        if len(candidates) >= 150:
            break
        idx = int(idx)
        item = EMBEDDINGS_INDEX[idx]
        rid = item["id"]

        if req.museum != "all" and item.get("source_museum_code") != req.museum:
            continue

        record = RECORDS_BY_ID.get(rid)
        if not record:
            continue
        if not record.get("has_image"):
            continue
        if not (record.get("image_url") or record.get("image_url_small")):
            continue

        candidates.append((record, float(scores[idx])))

    # ── Stage 2: RERANK with Claude ──
    reranked = _rerank_with_llm(req.query, candidates)

    if reranked:
        # Use LLM scores — build results from reranked order
        results = []
        for aid, llm_score, reason, raw_score in reranked[:80]:
            record = RECORDS_BY_ID.get(aid)
            if not record:
                continue
            art = format_artwork(record, score=llm_score / 100.0)
            art["raw_score"] = round(raw_score, 4)
            results.append(art)

        # Apply diversity sampling on reranked results
        # (already sorted by LLM score, just cap per-museum)
        if req.museum == "all" and len(results) > 20:
            max_per = int(80 * 0.4)
            museum_counts: dict[str, int] = {}
            diverse = []
            deferred = []
            for r in results:
                c = museum_counts.get(r["museum"], 0)
                if c < max_per:
                    diverse.append(r)
                    museum_counts[r["museum"]] = c + 1
                else:
                    deferred.append(r)
            remaining = 80 - len(diverse)
            if remaining > 0:
                diverse.extend(deferred[:remaining])
            results = diverse[:80]

        return results

    # Fallback: no ANTHROPIC_API_KEY or re-ranking failed — use embedding scores
    if req.museum == "all" and len(candidates) > 20:
        results = _diversity_sample(candidates, limit=80)
    else:
        results = [format_artwork(r, score=s) for r, s in candidates[:80]]
    return results


# ── Harvard image proxy ───────────────────────────────────────────────────
# Harvard's IIIF server aggressively rate-limits browser requests (429).
# This proxy serializes all Harvard image fetches through a single thread
# with a minimum interval between requests.

_harvard_proxy_lock = asyncio.Lock()
_harvard_last_request = 0.0
_HARVARD_MIN_INTERVAL = 0.5  # seconds between requests


@app.get("/api/image-proxy")
async def image_proxy(url: str = Query(...)):
    """Proxy image requests for rate-limited servers (Harvard IIIF).
    Serializes requests with a lock and retries on 429."""
    import httpx

    # Only proxy Harvard URLs — reject anything else
    if "ids.lib.harvard.edu" not in url:
        return Response(status_code=400, content="Only Harvard URLs are proxied")

    global _harvard_last_request

    max_retries = 3
    for attempt in range(max_retries):
        async with _harvard_proxy_lock:
            # Enforce minimum interval between requests
            now = time.monotonic()
            wait = _HARVARD_MIN_INTERVAL - (now - _harvard_last_request)
            if wait > 0:
                await asyncio.sleep(wait)
            _harvard_last_request = time.monotonic()

            try:
                async with httpx.AsyncClient(timeout=15.0) as client:
                    resp = await client.get(url)
            except Exception:
                return Response(status_code=502, content="Failed to fetch image")

        if resp.status_code == 429:
            # Back off and retry
            await asyncio.sleep(2.0 * (attempt + 1))
            continue

        if resp.status_code != 200:
            return Response(status_code=resp.status_code, content=resp.content)

        content_type = resp.headers.get("content-type", "image/jpeg")
        return Response(
            content=resp.content,
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # All retries exhausted
    return Response(status_code=429, content="Harvard rate limit — try again later")


def _diversity_sample(candidates: list, limit: int = 80) -> list:
    """
    Pick top results but cap any single museum at 40% of the output.
    Remaining slots are filled by the next-best results from other museums.
    """
    max_per_museum = int(limit * 0.4)
    museum_counts: dict[str, int] = {}
    selected = []
    deferred = []

    for record, score in candidates:
        code = record.get("source_museum_code", "")
        count = museum_counts.get(code, 0)
        if count < max_per_museum:
            selected.append(format_artwork(record, score=score))
            museum_counts[code] = count + 1
        else:
            deferred.append(format_artwork(record, score=score))
        if len(selected) >= limit:
            break

    # Fill remaining slots from deferred if we haven't reached limit
    remaining = limit - len(selected)
    if remaining > 0:
        selected.extend(deferred[:remaining])

    return selected
