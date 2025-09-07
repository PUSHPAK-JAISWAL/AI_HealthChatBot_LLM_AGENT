# app/main.py
"""
Single-file FastAPI app for:
AI-Driven Public Health Chatbot (HealthAlert)

- Sessions, users
- Document ingestion (Chroma or in-memory fallback)
- RAG + Ollama (gemma3:4b) chat endpoint
- Trusted web search (WHO + gov.in)
- On-demand alert scan endpoint that classifies outbreaks and saves alerts to Mongo
- Token-aware session trimming (defaults to 128k tokens) with optional summarization

New additions:
- Public API key registration endpoint (/public/register)
- Public chat endpoint (/public/chat) that external apps can call with X-API-KEY
- Alert scanning respects published article dates; older pages (default > ALERT_MAX_AGE_DAYS) are excluded unless allow_old=true

ENV:
- MONGO_URI (default: mongodb://localhost:27017)
- DB_NAME (default: health_alert)
- OLLAMA_URL (default: http://localhost:11434)
- OLLAMA_MODEL (default: gemma3:4b)
- CHROMA_PERSIST_DIR (default: ./chromadb_persist)
- SESSION_CONTEXT_MAX_TOKENS (default: 128000)
- SUMMARIZE_OLD_MESSAGES (default: "false")
- ALERT_MAX_AGE_DAYS (default: 30)
"""

import os
import time
import uuid
import json
import traceback
import asyncio
import re
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone, timedelta

from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# networking / scraping
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

# Mongo
import motor.motor_asyncio

# Embeddings
from sentence_transformers import SentenceTransformer
import numpy as np

# Optional packages (chromadb / googlesearch); app will fallback if missing
try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None

try:
    from googlesearch import search as google_search
except Exception:
    google_search = None

# token counting (optional tiktoken)
try:
    import tiktoken
except Exception:
    tiktoken = None

# optional date parsing
try:
    from dateutil import parser as dateutil_parser
except Exception:
    dateutil_parser = None

load_dotenv()

# ---------- Config ----------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "health_alert")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chromadb_persist")
SESSION_CONTEXT_MAX_TOKENS = int(os.getenv("SESSION_CONTEXT_MAX_TOKENS", 128000))
SUMMARIZE_OLD_MESSAGES = os.getenv("SUMMARIZE_OLD_MESSAGES", "false").lower() in ("1", "true", "yes")
ALERT_MAX_AGE_DAYS = int(os.getenv("ALERT_MAX_AGE_DAYS", 30))

# trusted domains for searching health news / advisories
HEALTH_DOMAIN_WHITELIST = {
    "who.int",
    "mohfw.gov.in",
    "gov.in",
    "nic.in",
    "nhm.gov.in",
    # add state domains like "health.odisha.gov.in" as needed
}

# ---------- App & DB ----------
app = FastAPI(title="HealthAlert — AI Public Health Chatbot")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_col = db["users"]
sessions_col = db["sessions"]
docs_meta_col = db["docs_meta"]
alerts_col = db["alerts"]
api_keys_col = db["api_keys"]

# ---------- Embedding + Chroma init (with fallback) ----------
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

chroma_client = None
collection = None
collection_lock = asyncio.Lock()

# in-memory fallback
_inmem_docs: Dict[str, Dict[str, Any]] = {}
_inmem_ids: List[str] = []
_inmem_embs: List[np.ndarray] = []

def create_chroma_client():
    global chroma_client, collection
    if chromadb is None:
        print("chromadb not installed — using in-memory fallback")
        chroma_client = None
        collection = None
        return
    try:
        settings = Settings(persist_directory=CHROMA_PERSIST_DIR)
        chroma_client = chromadb.Client(settings)
        collection = chroma_client.get_or_create_collection(name="health_docs")
        print("Chroma initialized with persist dir:", CHROMA_PERSIST_DIR)
        return
    except Exception as e:
        print("Chroma new-style init failed:", e)
        traceback.print_exc()
    try:
        chroma_client = chromadb.Client()
        collection = chroma_client.get_or_create_collection(name="health_docs")
        print("Chroma initialized with default client")
        return
    except Exception as e:
        print("Chroma default init failed:", e)
        traceback.print_exc()
    chroma_client = None
    collection = None
    print("Chroma unavailable — using in-memory fallback")

create_chroma_client()

def embed_text_sync(text: str) -> List[float]:
    vec = embedder.encode([text], convert_to_numpy=True)[0]
    return vec.astype("float32").tolist()

def _inmem_add(doc_id: str, title: str, content: str, source: Optional[str], emb: List[float]):
    _inmem_docs[doc_id] = {"title": title, "content": content, "source": source, "embedding": np.array(emb, dtype="float32")}
    _inmem_ids.append(doc_id)
    _inmem_embs.append(np.array(emb, dtype="float32"))

def _inmem_query(qemb: np.ndarray, k: int):
    if len(_inmem_embs) == 0:
        return []
    arr = np.vstack(_inmem_embs)
    dists = np.linalg.norm(arr - qemb.astype("float32"), axis=1)
    idxs = np.argsort(dists)[:k]
    results = []
    for idx in idxs:
        doc_id = _inmem_ids[int(idx)]
        meta = _inmem_docs[doc_id]
        results.append({
            "id": doc_id,
            "score": float(dists[int(idx)]),
            "meta": {"title": meta.get("title"), "source": meta.get("source")},
            "document": meta.get("content")
        })
    return results

async def add_doc(title: str, content: str, source: Optional[str] = None) -> str:
    doc_id = str(uuid.uuid4())
    emb = embed_text_sync(content)
    if collection is not None:
        try:
            # chroma is synchronous; keep within lock to avoid races
            async with collection_lock:
                collection.add(ids=[doc_id], documents=[content], metadatas=[{"title": title, "source": source}], embeddings=[emb])
        except Exception:
            traceback.print_exc()
            _inmem_add(doc_id, title, content, source, emb)
    else:
        _inmem_add(doc_id, title, content, source, emb)
    # store metadata in Mongo
    await docs_meta_col.insert_one({"_id": doc_id, "title": title, "source": source, "content": content, "created_at": time.time()})
    return doc_id

async def retrieve_similar(query: str, k: int = 4):
    qemb = np.array(embed_text_sync(query), dtype="float32")
    if collection is not None:
        try:
            res = collection.query(query_embeddings=[qemb.tolist()], n_results=k, include=["metadatas", "documents", "distances"])
            items = []
            if not res or "ids" not in res:
                return []
            for i in range(len(res["ids"])):
                for j, doc_id in enumerate(res["ids"][i]):
                    items.append({
                        "id": doc_id,
                        "score": float(res["distances"][i][j]),
                        "meta": res["metadatas"][i][j],
                        "document": res["documents"][i][j]
                    })
            return items
        except Exception:
            traceback.print_exc()
    # fallback
    return _inmem_query(qemb, k)

# ---------- Models ----------
class UserCreate(BaseModel):
    user_id: str
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    extra: Optional[Dict[str, Any]] = {}

class SessionCreate(BaseModel):
    user_id: str
    title: Optional[str] = "Chat Session"
    meta: Optional[Dict[str, Any]] = {}

class MessageIn(BaseModel):
    user_id: str
    text: str

class DocIn(BaseModel):
    title: str
    content: str
    source: Optional[str] = None

# public API models
class PublicRegister(BaseModel):
    user_id: Optional[str] = None
    name: Optional[str] = None

class PublicMessage(BaseModel):
    text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    title: Optional[str] = None

# ---------- Utilities ----------
def is_trusted_domain(url: str) -> bool:
    try:
        host = urlparse(url).hostname or ""
        host = host.lower()
        for d in HEALTH_DOMAIN_WHITELIST:
            if host == d or host.endswith("." + d):
                return True
        return False
    except Exception:
        return False

def _try_parse_date_string(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.strip()
    # try dateutil if available
    if dateutil_parser is not None:
        try:
            dt = dateutil_parser.parse(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            pass
    # try iso format
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        pass
    # look for YYYY-MM-DD-ish patterns
    m = re.search(r'(\d{4}-\d{2}-\d{2})([ T]\d{2}:\d{2}:\d{2})?', s)
    if m:
        try:
            dt = datetime.fromisoformat(m.group(0))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            pass
    # fallback None
    return None

def extract_published_ts_from_soup(soup: BeautifulSoup) -> Optional[float]:
    # check common meta tags
    meta_props = [
        ('meta', {'property': 'article:published_time'}),
        ('meta', {'name': 'article:published_time'}),
        ('meta', {'name': 'pubdate'}),
        ('meta', {'name': 'publish-date'}),
        ('meta', {'name': 'publication_date'}),
        ('meta', {'name': 'date'}),
        ('meta', {'property': 'og:published_time'}),
        ('meta', {'property': 'og:updated_time'}),
        ('meta', {'name': 'DC.Date'}),
    ]
    for tag_name, attrs in meta_props:
        try:
            t = soup.find(tag_name, attrs=attrs)
            if t and t.get("content"):
                ts = _try_parse_date_string(t.get("content"))
                if ts:
                    return ts
        except Exception:
            continue
    # <time datetime="...">
    try:
        time_tag = soup.find("time")
        if time_tag:
            dt = time_tag.get("datetime") or time_tag.get_text()
            ts = _try_parse_date_string(dt)
            if ts:
                return ts
    except Exception:
        pass
    # other possible meta properties
    try:
        t = soup.find("meta", attrs={"property": "article:modified_time"})
        if t and t.get("content"):
            ts = _try_parse_date_string(t.get("content"))
            if ts:
                return ts
    except Exception:
        pass
    # last-resort: find a date-like pattern near top of article
    try:
        body_text = soup.get_text()[:1000]
        m = re.search(r'(\d{4}-\d{2}-\d{2})', body_text)
        if m:
            ts = _try_parse_date_string(m.group(1))
            if ts:
                return ts
    except Exception:
        pass
    return None

def web_search_health(query: str, num: int = 6):
    """
    Returns list of dicts: {"url": url, "snippet": snippet, "published_ts": <float|None>, "published_iso": <str|None>}
    Only returns trusted domain results (HEALTH_DOMAIN_WHITELIST).
    """
    results = []
    if google_search is None:
        return results
    try:
        urls = list(google_search(query, num_results=num))
    except Exception:
        urls = []
    now_ts = time.time()
    cutoff_seconds = ALERT_MAX_AGE_DAYS * 86400
    for url in urls:
        if not is_trusted_domain(url):
            continue
        try:
            r = requests.get(url, timeout=8)
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            desc_tag = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
            desc = desc_tag.get("content", "").strip() if desc_tag else ""
            if not desc:
                p = soup.find("p")
                desc = p.get_text().strip()[:800] if p else ""
            published_ts = extract_published_ts_from_soup(soup)
            published_iso = None
            if published_ts:
                try:
                    published_iso = datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat()
                except Exception:
                    published_iso = None
            # by default exclude pages older than ALERT_MAX_AGE_DAYS (unless caller overrides)
            results.append({"url": url, "snippet": desc, "published_ts": published_ts, "published_iso": published_iso})
        except Exception:
            continue
    return results

def call_ollama_generate(prompt: str, timeout: int = 60) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    r = requests.post(url, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "response" in data:
        return data["response"]
    return json.dumps(data)

def build_alert_prompt(region: str, web_results: list, retrieved_docs: list, now_ts: float) -> str:
    sys = (
        "You are a public-health monitoring assistant. "
        "Given web findings and internal documents, classify if there is an outbreak and assign a severity level. "
        "Return JSON ONLY (no extra text)."
    )
    parts = [f"SYSTEM: {sys}\nREGION: {region}\nTIMESTAMP: {now_ts}\n"]
    if web_results:
        parts.append("WEB_FINDINGS:")
        for i, w in enumerate(web_results, start=1):
            pub = w.get("published_iso") or "unknown"
            parts.append(f"\n[{i}] {w.get('url')}\nPublished: {pub}\n{w.get('snippet')}\n")
    if retrieved_docs:
        parts.append("\nLOCAL_DOCS:")
        for i, d in enumerate(retrieved_docs, start=1):
            meta = d.get("meta", {}) or {}
            title = meta.get("title") or ""
            src = meta.get("source") or ""
            content = (d.get("document") or "")[:2000]
            parts.append(f"\n--- Doc {i} ---\nTitle: {title}\nSource: {src}\nContent: {content}\n")
    parts.append(
        "\nINSTRUCTIONS: Output ONLY valid JSON with exactly these fields:\n"
        '{\n'
        '  "severity": "none|low|moderate|high|critical",\n'
        '  "confidence": 0.0,\n'
        '  "summary": "short summary (<=200 chars)",\n'
        '  "recommended_actions": ["..."],\n'
        '  "should_send_alert": true,\n'
        '  "references": [{"url":"...","title":"..."}]\n'
        '}\n'
        "Do NOT add any commentary or explanation outside the JSON."
    )
    return "\n".join(parts)

async def scan_region_for_alert(region: str = "India", query: Optional[str] = None, k_docs: int = 4, allow_old: bool = False):
    """Run a web + local-doc scan for the given region, call LLM to classify and store the alert.

    Now respects article publish timestamps and by default excludes results older than ALERT_MAX_AGE_DAYS.
    Set allow_old=True to include older pages (not recommended unless explicitly desired).
    """
    now_ts = time.time()
    q = query or f"outbreak OR surge OR cluster OR 'disease outbreak' site:gov.in OR site:who.int {region}"
    web_results_raw = web_search_health(q, num=8)
    # filter by publish date
    filtered_web_results = []
    cutoff_seconds = ALERT_MAX_AGE_DAYS * 86400
    filtered_out_count = 0
    for w in web_results_raw:
        pub_ts = w.get("published_ts")
        if pub_ts is None:
            # unknown publish date: include but mark as unknown
            filtered_web_results.append(w)
            continue
        age = now_ts - float(pub_ts)
        if age <= cutoff_seconds or allow_old:
            filtered_web_results.append(w)
        else:
            filtered_out_count += 1

    retrieved = await retrieve_similar(q, k=k_docs)
    prompt = build_alert_prompt(region, filtered_web_results, retrieved, now_ts)
    try:
        raw = call_ollama_generate(prompt, timeout=60)
    except Exception as e:
        raise RuntimeError(f"Ollama generate failed: {e}")
    parsed = {}
    try:
        parsed = json.loads(raw)
    except Exception:
        m = re.search(r'(\{.*\})', raw, re.S)
        if m:
            try:
                parsed = json.loads(m.group(1))
            except Exception:
                parsed = {"severity": "unknown", "confidence": 0.0, "summary": raw[:200], "recommended_actions": [], "should_send_alert": False, "references": []}
        else:
            parsed = {"severity": "unknown", "confidence": 0.0, "summary": raw[:200], "recommended_actions": [], "should_send_alert": False, "references": []}
    alert_doc = {
        "region": region,
        "query": q,
        "web_results": filtered_web_results,
        "filtered_out_count": filtered_out_count,
        "retrieved_docs": [{"id": r.get("id"), "meta": r.get("meta")} for r in retrieved],
        "llm_raw": raw,
        "alert": parsed,
        "created_at": now_ts,
    }
    # store and return inserted id
    res = await alerts_col.insert_one(alert_doc)
    alert_doc["_id"] = res.inserted_id
    return alert_doc

# ---------- Token counting helpers ----------
# Try to initialize a tiktoken encoder; fallback to a heuristic if unavailable
_enc = None
if tiktoken is not None:
    try:
        # prefer cl100k_base for general-purpose tokenization
        _enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        try:
            # try to pick encoding for model (best-effort)
            _enc = tiktoken.encoding_for_model(OLLAMA_MODEL)
        except Exception:
            _enc = None

def count_tokens(text: str) -> int:
    if not text:
        return 0
    if _enc is not None:
        try:
            return len(_enc.encode(text))
        except Exception:
            pass
    # fallback heuristic: assume ~4 characters per token
    return max(1, int(len(text) / 4))

# Summarization helper (optional): condense text using Ollama
def summarize_text_via_llm(text: str, max_chars: int = 1000) -> str:
    """
    Summarize input text using Ollama. Returns summary string.
    This is a synchronous call and may take time; only used if SUMMARIZE_OLD_MESSAGES is enabled.
    """
    prompt = (
        "You are an assistant that summarizes conversation history for context retention.\n"
        "Create a concise factual summary preserving key facts, named places, dates, and decisions. "
        "Limit to 150-200 words or fewer. Return ONLY the summary text.\n\n"
        "CONVERSATION:\n" + text + "\n\nSUMMARY:"
    )
    try:
        raw = call_ollama_generate(prompt, timeout=30)
        # try to return raw as-is (strip)
        return raw.strip()
    except Exception:
        return text[:max_chars]

# ---------- Token-aware session trimming (with optional summarization) ----------
async def trim_session_messages(session_id: str):
    s = await sessions_col.find_one({"_id": session_id})
    if not s:
        return
    messages = s.get("messages", [])
    if not messages:
        return

    # compute token counts per message
    token_counts = [count_tokens(m.get("text", "")) for m in messages]
    total_tokens = sum(token_counts)
    if total_tokens <= SESSION_CONTEXT_MAX_TOKENS:
        return

    # If summarization is enabled, we'll iteratively summarize the oldest chunk
    if SUMMARIZE_OLD_MESSAGES:
        # loop until under limit or no more messages to compress
        while messages and total_tokens > SESSION_CONTEXT_MAX_TOKENS:
            # take oldest N messages to summarize (tune N; start with 8)
            take = min(len(messages), 8)
            chunk_msgs = messages[:take]
            chunk_text = "\n".join([f"{m.get('role','')}: {m.get('text','')}" for m in chunk_msgs])
            summary = summarize_text_via_llm(chunk_text)  # may be long; it's okay
            # remove those messages
            removed_tokens = sum(token_counts[:take])
            messages = messages[take:]
            token_counts = token_counts[take:]
            # insert summary message at beginning as a system-ish message
            summary_msg = {"role": "system", "user_id": None, "text": f"[SUMMARY OF OLDER MESSAGES]: {summary}", "ts": time.time()}
            messages.insert(0, summary_msg)
            summary_tokens = count_tokens(summary_msg["text"])
            token_counts.insert(0, summary_tokens)
            total_tokens = sum(token_counts)
            # if summarization didn't help (rare), fallback to removing oldest messages without summarization
            if total_tokens > SESSION_CONTEXT_MAX_TOKENS and len(messages) <= 1:
                # remove oldest until under limit
                while messages and total_tokens > SESSION_CONTEXT_MAX_TOKENS:
                    removed = messages.pop(0)
                    # recompute tokens
                    token_counts = [count_tokens(m.get("text", "")) for m in messages]
                    total_tokens = sum(token_counts)
                break
    else:
        # simple removal: drop oldest messages until under token limit
        while messages and total_tokens > SESSION_CONTEXT_MAX_TOKENS:
            removed = messages.pop(0)
            total_tokens -= count_tokens(removed.get("text", ""))
    # store trimmed messages back to DB
    await sessions_col.update_one({"_id": session_id}, {"$set": {"messages": messages}})

# ---------- API Endpoints ----------
@app.post("/user", summary="Create or upsert user")
async def create_user(payload: UserCreate):
    doc = payload.dict()
    doc["_id"] = payload.user_id
    await users_col.update_one({"_id": payload.user_id}, {"$set": doc}, upsert=True)
    return {"status": "ok", "user_id": payload.user_id}

@app.get("/user/{user_id}")
async def get_user(user_id: str):
    u = await users_col.find_one({"_id": user_id}, {"_id": 0})
    if not u:
        raise HTTPException(status_code=404, detail="user not found")
    return u

@app.delete("/user/{user_id}")
async def delete_user(user_id: str):
    await users_col.delete_one({"_id": user_id})
    await sessions_col.delete_many({"user_id": user_id})
    return {"status": "deleted"}

@app.post("/session", summary="Create a chat session")
async def create_session(payload: SessionCreate):
    session_id = str(uuid.uuid4())
    doc = {"_id": session_id, "user_id": payload.user_id, "title": payload.title, "meta": payload.meta, "messages": [], "created_at": time.time()}
    await sessions_col.insert_one(doc)
    return {"session_id": session_id}

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    s = await sessions_col.find_one({"_id": session_id}, {"_id": 0})
    if not s:
        raise HTTPException(status_code=404, detail="session not found")
    return s

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    await sessions_col.delete_one({"_id": session_id})
    return {"status": "deleted"}

@app.post("/session/{session_id}/message", summary="Send a user message (chat + RAG)")
async def post_message(session_id: str, payload: MessageIn):
    session = await sessions_col.find_one({"_id": session_id})
    if not session:
        raise HTTPException(status_code=404, detail="session not found")

    user_msg = {"role": "user", "user_id": payload.user_id, "text": payload.text, "ts": time.time()}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": user_msg}})

    # Trim session to configured token budget (may summarize older messages if enabled)
    await trim_session_messages(session_id)

    retrieved = await retrieve_similar(payload.text, k=4)
    web_results = web_search_health(payload.text, num=3)

    user = await users_col.find_one({"_id": payload.user_id}) or {}
    user_memory_text = json.dumps(user.get("memory", {}), indent=2) if user.get("memory") else ""

    session = await sessions_col.find_one({"_id": session_id})
    session_messages = session.get("messages", []) if session else []

    system_instructions = (
        "You are a public-health assistant helping rural and semi-urban citizens with preventive healthcare, disease symptoms, and vaccination schedules. "
        "Be concise, provide actionable steps, cite sources from the provided local docs or web results, and avoid hallucinations."
    )
    parts = [f"SYSTEM: {system_instructions}\n"]
    if user_memory_text:
        parts.append(f"USER_MEMORY:\n{user_memory_text}\n")
    if retrieved:
        parts.append("RETRIEVED_DOCS:")
        for i, r in enumerate(retrieved, 1):
            parts.append(f"\n--- Doc {i} ---\nTitle: {r.get('meta', {}).get('title')}\nSource: {r.get('meta', {}).get('source')}\nContent: {r.get('document')}\n")
    if web_results:
        parts.append("WEB_RESULTS:")
        for i, w in enumerate(web_results, 1):
            parts.append(f"\n[{i}] {w.get('url')} - {w.get('snippet')[:300]}\n")
    if session_messages:
        parts.append("\nCONVERSATION_HISTORY:")
        for m in session_messages[-12:]:  # include last 12 messages for context
            parts.append(f"\n{m.get('role','').upper()}: {m.get('text')}")
    parts.append(f"\nUSER: {payload.text}\n")
    parts.append("\nASSISTANT: (Be concise, provide actionable steps, cite URLs where used.)\n")
    prompt = "\n".join(parts)

    try:
        assistant_text = call_ollama_generate(prompt, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    assistant_msg = {"role": "assistant", "text": assistant_text, "ts": time.time()}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": assistant_msg}})

    # Optional: memory extraction could be added here (omitted for brevity)
    return {"assistant": assistant_text, "retrieved": retrieved, "web_results": web_results}

# ---------- Public API key management & public chat ----------
async def _get_api_key_doc(api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    doc = await api_keys_col.find_one({"_key": api_key})
    return doc

@app.post("/public/register", summary="Create a public API key (simple)")
async def public_register(payload: PublicRegister):
    # Create a simple API key tied to optional user_id
    api_key = uuid.uuid4().hex
    doc = {
        "_id": str(uuid.uuid4()),
        "_key": api_key,
        "user_id": payload.user_id,
        "name": payload.name,
        "active": True,
        "created_at": time.time(),
    }
    await api_keys_col.insert_one(doc)
    # return only the key (store in DB so you can revoke later)
    return {"status": "ok", "api_key": api_key, "note": "Store this key securely; it's shown only once here."}

@app.post("/public/chat", summary="Public chat endpoint — use X-API-KEY header")
async def public_chat(payload: PublicMessage, x_api_key: Optional[str] = Header(None)):
    api_key = x_api_key
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-KEY header")
    key_doc = await _get_api_key_doc(api_key)
    if not key_doc or not key_doc.get("active", False):
        raise HTTPException(status_code=401, detail="Invalid or inactive API key")

    # Allow reuse of existing sessions; if not provided create a new session for this API client
    session_id = payload.session_id
    if session_id:
        sess = await sessions_col.find_one({"_id": session_id})
        if not sess:
            # create new session if provided id not found
            session_id = None

    if not session_id:
        session_id = str(uuid.uuid4())
        title = payload.title or f"PublicSession-{session_id[:8]}"
        doc = {"_id": session_id, "user_id": payload.user_id or key_doc.get("user_id") or "public_user", "title": title, "meta": {"public_key_id": key_doc.get("_id")}, "messages": [], "created_at": time.time()}
        await sessions_col.insert_one(doc)

    # Insert user message into session
    user_msg = {"role": "user", "user_id": payload.user_id or key_doc.get("user_id") or "public_user", "text": payload.text, "ts": time.time()}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": user_msg}})

    # Trim session
    await trim_session_messages(session_id)

    # RAG + web
    retrieved = await retrieve_similar(payload.text, k=4)
    web_results = web_search_health(payload.text, num=3)

    # Build prompt similar to internal flow
    user = await users_col.find_one({"_id": payload.user_id}) or {}
    user_memory_text = json.dumps(user.get("memory", {}), indent=2) if user.get("memory") else ""
    session = await sessions_col.find_one({"_id": session_id})
    session_messages = session.get("messages", []) if session else []

    system_instructions = (
        "You are a public-health assistant helping rural and semi-urban citizens with preventive healthcare, disease symptoms, and vaccination schedules. "
        "Be concise, provide actionable steps, cite sources from the provided local docs or web results, and avoid hallucinations."
    )
    parts = [f"SYSTEM: {system_instructions}\n"]
    if user_memory_text:
        parts.append(f"USER_MEMORY:\n{user_memory_text}\n")
    if retrieved:
        parts.append("RETRIEVED_DOCS:")
        for i, r in enumerate(retrieved, 1):
            parts.append(f"\n--- Doc {i} ---\nTitle: {r.get('meta', {}).get('title')}\nSource: {r.get('meta', {}).get('source')}\nContent: {r.get('document')}\n")
    if web_results:
        parts.append("WEB_RESULTS:")
        for i, w in enumerate(web_results, 1):
            parts.append(f"\n[{i}] {w.get('url')} - {w.get('snippet')[:300]}\n")
    if session_messages:
        parts.append("\nCONVERSATION_HISTORY:")
        for m in session_messages[-12:]:
            parts.append(f"\n{m.get('role','').upper()}: {m.get('text')}")
    parts.append(f"\nUSER: {payload.text}\n")
    parts.append("\nASSISTANT: (Be concise, provide actionable steps, cite URLs where used.)\n")
    prompt = "\n".join(parts)

    try:
        assistant_text = call_ollama_generate(prompt, timeout=60)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    assistant_msg = {"role": "assistant", "text": assistant_text, "ts": time.time()}
    await sessions_col.update_one({"_id": session_id}, {"$push": {"messages": assistant_msg}})

    return {"assistant": assistant_text, "session_id": session_id, "retrieved": retrieved, "web_results": web_results}

@app.post("/docs", summary="Ingest a doc into Chroma DB")
async def ingest_doc(payload: DocIn):
    doc_id = await add_doc(payload.title, payload.content, payload.source)
    return {"status": "ok", "doc_id": doc_id}

@app.get("/alerts/scan", summary="Trigger health scan for region (on-demand)")
async def alerts_scan(region: Optional[str] = "India", query: Optional[str] = None, k_docs: int = 4, allow_old: bool = False):
    try:
        doc = await scan_region_for_alert(region=region, query=query, k_docs=k_docs, allow_old=allow_old)
        # return richer payload: parsed alert, web findings, retrieved docs, raw llm output, and saved id
        return {
            "status": "ok",
            "alert_id": str(doc.get("_id")),
            "region": doc.get("region"),
            "query": doc.get("query"),
            "created_at": doc.get("created_at"),
            "alert": doc.get("alert"),
            "web_results": doc.get("web_results"),
            "retrieved_docs": doc.get("retrieved_docs"),
            "llm_raw": doc.get("llm_raw"),
            "filtered_out_count": doc.get("filtered_out_count", 0),
            "note": f"By default pages older than {ALERT_MAX_AGE_DAYS} days are excluded unless allow_old=true."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts/latest", summary="Get latest stored alert for a region")
async def get_latest_alert(region: Optional[str] = "India"):
    doc = await alerts_col.find_one({"region": region}, sort=[("created_at", -1)])
    if not doc:
        return {"status": "none", "message": "no alerts found"}
    return {"status": "ok", "alert_id": str(doc.get("_id")), "alert": doc.get("alert"), "created_at": doc.get("created_at")}

@app.get("/health")
async def health():
    api_keys_count = await api_keys_col.count_documents({})
    return {
        "status": "ok",
        "ollama": OLLAMA_URL,
        "model": OLLAMA_MODEL,
        "chroma": bool(collection),
        "session_context_max_tokens": SESSION_CONTEXT_MAX_TOKENS,
        "summarize_old_messages": SUMMARIZE_OLD_MESSAGES,
        "alert_max_age_days": ALERT_MAX_AGE_DAYS,
        "public_api_keys": api_keys_count,
    }

# ---------- Run instructions ----------
# Save as app/main.py. Install requirements (see requirements.txt from earlier message) and run:
# uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
