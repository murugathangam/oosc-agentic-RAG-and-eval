from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import chromadb
import requests
import os
import json
import time
import re
import numpy as np

app = FastAPI()

# CORS — required so the browser-based UI (a separate origin) can call this API.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY = os.getenv("API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-120b"
WEB_MODEL = "groq/compound-mini"


def groq_post(payload: dict, max_retries: int = 3):
    """Wraps requests.post to Groq with automatic retry on rate-limit (429)
    errors. Groq's error message includes how long to wait — we parse it
    and sleep, rather than failing the whole request on a transient limit."""
    header = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    for attempt in range(max_retries + 1):
        r = requests.post(GROQ_URL, headers=header, json=payload, timeout=30)
        if r.status_code != 429:
            return r
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        message = data.get("error", {}).get("message", "")
        match = re.search(r"try again in ([\d.]+)s", message)
        wait_seconds = float(match.group(1)) + 0.5 if match else 2 * (attempt + 1)
        if attempt < max_retries:
            time.sleep(wait_seconds)
        else:
            return r  # give up, let caller handle the still-429 response
    return r

client = chromadb.PersistentClient()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
collection = client.get_or_create_collection("my_collection", embedding_function=None)

MAX_HISTORY_TURNS = 3   # keep at most the last 3 question/answer pairs — lower reduces tokens per request and cuts rate-limit risk on free/dev tiers
chat_history = []        # list of {"role": "user"/"assistant", "content": ...}

def add_to_history(question: str, answer: str):
    chat_history.append({"role": "user", "content": question})
    chat_history.append({"role": "assistant", "content": answer})
    # trim to the last MAX_HISTORY_TURNS exchanges (2 messages per exchange)
    max_messages = MAX_HISTORY_TURNS * 2
    while len(chat_history) > max_messages:
        chat_history.pop(0)


# ---------------------------------------------------------------------
# CHUNKING + STORAGE
# ---------------------------------------------------------------------

class Material(BaseModel):
    material: str
    chunk_size: int = 200
    overlap: int = 50
    source_name: str = "doc"

def chunk_text(material: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    doc = []
    pointer = 0
    step = chunk_size - overlap
    while pointer < len(material):
        doc.append(material[pointer:pointer + chunk_size])
        pointer += step
    return doc

@app.post("/chunk1")
def chunker(chunk_material: Material):
    chunks = chunk_text(chunk_material.material, chunk_material.chunk_size, chunk_material.overlap)
    embeddings = embedding_model.encode(chunks)
    ids = [f"{chunk_material.source_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"source": chunk_material.source_name, "chunk_index": i} for i in range(len(chunks))]
    collection.upsert(ids=ids, embeddings=embeddings.tolist(), documents=chunks, metadatas=metadatas)
    return {"message": "chunked and stored", "num_chunks": len(chunks)}


# ---------------------------------------------------------------------
# RETRIEVAL TOOL — returns metadata too, for citation
# ---------------------------------------------------------------------

def retrieve(query: str, k: int = 3):
    query_embedding = embedding_model.encode([query])
    results = collection.query(query_embeddings=query_embedding.tolist(), n_results=k)
    return {
        "documents": results["documents"][0] if results["documents"] else [],
        "ids": results["ids"][0] if results["ids"] else [],
        "distances": results["distances"][0] if results["distances"] else [],
        "metadatas": results["metadatas"][0] if results["metadatas"] else [],
    }

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0

TOOLS = [{
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Search the student's uploaded study materials for relevant passages. Call this whenever you need information to answer the question. You may call it multiple times with different queries if the first result isn't sufficient.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "reason": {"type": "string", "description": "One short sentence explaining why you're running this specific search — what you're trying to find or clarify."}
            },
            "required": ["query", "reason"]
        }
    }
}]


# ---------------------------------------------------------------------
# DIFFICULTY-AWARE EXPLANATION
# ---------------------------------------------------------------------

DIFFICULTY_PROMPTS = {
    "simple": "Explain this in very simple terms, as if for a beginner or younger student. Avoid jargon; use short sentences and, if helpful, an everyday analogy.",
    "standard": "Explain this clearly and concisely, at a normal high-school/early-college level.",
    "advanced": "Give a thorough, technically detailed explanation, including derivations or precise terminology where relevant, suitable for an advanced student.",
}

def get_system_prompt(difficulty: str) -> str:
    style = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["standard"])
    return (
        "You are a study assistant. Use the retrieve tool to search the student's notes before answering. "
        "Call retrieve as many times as needed with refined queries if results aren't relevant enough, "
        "but don't call it more than necessary. "
        "IMPORTANT: When you use information from a retrieved passage in your answer, cite it inline in the form "
        "[Source: <source name>, chunk <chunk_index>], using the source and chunk_index values given to you in the tool results. "
        "Once you have enough information, answer directly without calling the tool. "
        f"{style}"
    )


# ---------------------------------------------------------------------
# AGENTIC LOOP AGAINST LOCAL NOTES
# ---------------------------------------------------------------------

MAX_TOOL_CALLS = 6

def run_agentic_chat(question: str, difficulty: str = "standard"):
    trajectory = []
    seen_chunk_ids = set()
    sources_used = {}
    retrieved_texts = {}   # chunk_id -> text, deduped, used to build a real faithfulness context
    original_embedding = embedding_model.encode([question])[0]
    prev_query_embeddings = []

    messages = [
        {"role": "system", "content": get_system_prompt(difficulty)},
        *chat_history,
        {"role": "user", "content": question}
    ]

    header = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    tool_call_count = 0

    while tool_call_count < MAX_TOOL_CALLS:
        payload = {"model": MODEL, "messages": messages, "tools": TOOLS, "tool_choice": "auto", "temperature": 0.2}
        r = groq_post(payload)
        data = r.json()

        if "choices" not in data:
            return {"answer": f"[Groq error: {data}]", "trajectory": trajectory, "sources_used": [], "retrieved_texts": []}

        msg = data["choices"][0]["message"]
        messages.append(msg)

        tool_calls = msg.get("tool_calls")
        if not tool_calls:
            final_answer = msg.get("content", "")
            return {
                "answer": final_answer,
                "trajectory": trajectory,
                "messages": messages,
                "sources_used": list(sources_used.values()),
                "retrieved_texts": list(retrieved_texts.values()),
            }

        for tc in tool_calls:
            tool_call_count += 1
            args = json.loads(tc["function"]["arguments"])
            query = args.get("query", question)
            reason = args.get("reason", "")

            current_embedding = embedding_model.encode([query])[0]
            result = retrieve(query, k=3)
            best_distance = min(result["distances"]) if result["distances"] else 999
            new_ids = [i for i in result["ids"] if i not in seen_chunk_ids]
            redundant = len(new_ids) == 0 and tool_call_count > 1
            drift = cosine_sim(current_embedding, original_embedding)
            loop = any(cosine_sim(current_embedding, p) > 0.95 for p in prev_query_embeddings)

            trajectory.append({
                "step": tool_call_count,
                "query": query,
                "reason": reason,
                "retrieved_ids": result["ids"],
                "best_distance": best_distance,
                "new_chunks_found": len(new_ids),
                "redundant_step": redundant,
                "drift_from_original": drift,
                "loop_detected": loop,
            })

            for cid, meta, doc in zip(result["ids"], result["metadatas"], result["documents"]):
                seen_chunk_ids.add(cid)
                sources_used[cid] = {"source": meta.get("source"), "chunk_index": meta.get("chunk_index")}
                retrieved_texts[cid] = doc
            prev_query_embeddings.append(current_embedding)

            tool_payload = {
                "documents": result["documents"],
                "distances": result["distances"],
                "metadatas": result["metadatas"],
            }
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps(tool_payload)
            })

    messages.append({"role": "user", "content": "Please answer now with what you have, citing sources as instructed."})
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2}
    r = groq_post(payload)
    data = r.json()
    final_answer = data["choices"][0]["message"]["content"] if "choices" in data else f"[error: {data}]"
    return {
        "answer": final_answer,
        "trajectory": trajectory,
        "messages": messages,
        "hit_ceiling": True,
        "sources_used": list(sources_used.values()),
        "retrieved_texts": list(retrieved_texts.values()),
    }


def compute_trajectory_metrics(trajectory: list[dict]) -> dict:
    if not trajectory:
        return {"total_steps": 0, "unnecessary_steps": 0, "loop_detected": False, "drift_compounded": 0.0, "avg_retrieval_distance": 999.0, "efficiency_score": 1.0}
    total_steps = len(trajectory)
    unnecessary_steps = sum(1 for s in trajectory if s["redundant_step"])
    looped = any(s["loop_detected"] for s in trajectory)
    drift_values = [s["drift_from_original"] for s in trajectory]
    drift_compounded = (drift_values[0] - drift_values[-1]) if len(drift_values) > 1 else 0.0
    avg_dist = sum(s["best_distance"] for s in trajectory) / total_steps
    return {
        "total_steps": total_steps,
        "unnecessary_steps": unnecessary_steps,
        "loop_detected": looped,
        "drift_compounded": drift_compounded,
        "avg_retrieval_distance": avg_dist,
        "efficiency_score": round(1 / (1 + unnecessary_steps + (2 if looped else 0)), 3),
    }


def check_faithfulness(answer: str, context_text: str, context_label: str = "retrieved passages") -> dict:
    """Judges whether ANSWER is actually supported by context_text — works
    identically whether the context came from the student's own notes or
    from a web search, since both are just 'what was actually found'."""
    if not context_text.strip():
        return {"raw_verdict": "SCORE: 0\nREASON: No context was retrieved to check the answer against.", "score": 0}

    judge_messages = [
        {"role": "system", "content": f"You are a strict fact-checker judging whether an ANSWER is grounded in the given {context_label}. Reply EXACTLY as:\nSCORE: <0-10>\nREASON: <one sentence>"},
        {"role": "user", "content": f"{context_label.upper()}:\n{context_text}\n\nANSWER:\n{answer}"}
    ]
    payload = {"model": MODEL, "messages": judge_messages, "temperature": 0.0}
    r = groq_post(payload)
    data = r.json()
    verdict = data["choices"][0]["message"]["content"] if "choices" in data else f"[error: {data}]"

    score = None
    if "SCORE:" in verdict:
        try:
            score = int(verdict.split("SCORE:")[1].split("\n")[0].strip())
        except Exception:
            score = None

    return {"raw_verdict": verdict, "score": score}


# ---------------------------------------------------------------------
# WEB SEARCH FALLBACK — only when local notes are insufficient
# ---------------------------------------------------------------------

WEB_FALLBACK_DISTANCE_THRESHOLD = 1.1

def needs_web_fallback(metrics: dict) -> bool:
    if metrics["total_steps"] == 0:
        return True
    return metrics["avg_retrieval_distance"] > WEB_FALLBACK_DISTANCE_THRESHOLD

def web_search_fallback(question: str, difficulty: str = "standard") -> dict:
    header = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    style = DIFFICULTY_PROMPTS.get(difficulty, DIFFICULTY_PROMPTS["standard"])
    payload = {
        "model": WEB_MODEL,
        "messages": [
            {"role": "system", "content": f"The student's uploaded notes did not cover this question well. Search the web for a reliable, accurate answer. {style} State your answer clearly."},
            {"role": "user", "content": question}
        ],
        "search_settings": {
            "include_domains": ["arxiv.org", "paperswithcode.com", "docs.python.org", "wikipedia.org", "stackoverflow.com", "github.com", "ocw.mit.edu", "britannica.com"]
        }
    }
    r = groq_post(payload)
    data = r.json()

    if "choices" not in data:
        return {"answer": f"[Web search error: {data}]", "web_sources": [], "web_context_text": ""}

    message = data["choices"][0]["message"]
    answer = message.get("content", "")

    web_sources = []
    web_snippets = []
    executed = data["choices"][0].get("message", {}).get("executed_tools") or data.get("executed_tools")
    if executed:
        for tool_call in executed:
            results = tool_call.get("search_results", {}).get("results", []) if isinstance(tool_call, dict) else []
            for res in results:
                url = res.get("url")
                if url:
                    web_sources.append(url)
                snippet = res.get("content") or res.get("snippet") or res.get("description") or ""
                if url or snippet:
                    web_snippets.append(f"{url}: {snippet}".strip(": "))

    web_context_text = "\n\n".join(web_snippets)
    return {"answer": answer, "web_sources": web_sources, "web_context_text": web_context_text}


# ---------------------------------------------------------------------
# STUDENT-FACING CONFIDENCE LABEL
# ---------------------------------------------------------------------

def build_confidence_label(faithfulness_score, source: str, hit_ceiling: bool) -> str:
    if source == "web":
        base = "This answer used a web search because your notes didn't cover this topic well."
        if faithfulness_score is None:
            return f"{base} Could not verify how well it matches the sites it found — double-check against a trusted source."
        if faithfulness_score >= 8:
            return f"{base} It stayed closely grounded in what it found on the web."
        if faithfulness_score >= 5:
            return f"{base} Some parts may go beyond what the web search actually found — verify key details."
        return f"{base} It may have added information beyond what the web search actually found — verify this carefully."
    if faithfulness_score is None:
        return "Could not verify how well-grounded this answer is — review it carefully."
    if hit_ceiling:
        return "This answer required extensive searching and may be incomplete — consider asking your teacher."
    if faithfulness_score >= 8:
        return "This answer is well-grounded in your notes."
    if faithfulness_score >= 5:
        return "This answer is partially grounded in your notes — some parts may not be fully covered. Verify key details."
    return "This answer may not be well-supported by your notes. Please verify with your teacher or textbook."


# ---------------------------------------------------------------------
# ENDPOINT
# ---------------------------------------------------------------------

class Question(BaseModel):
    message: str
    difficulty: str = "standard"

@app.post("/agent_chat")
def agent_chat(request: Question):
    result = run_agentic_chat(request.message, request.difficulty)
    metrics = compute_trajectory_metrics(result["trajectory"])

    source = "notes"
    web_sources = []
    answer = result["answer"]
    hit_ceiling = result.get("hit_ceiling", False)
    sources_used = result.get("sources_used", [])

    if needs_web_fallback(metrics):
        web_result = web_search_fallback(request.message, request.difficulty)
        answer = web_result["answer"]
        web_sources = web_result["web_sources"]
        source = "web"
        sources_used = []
        faithfulness = check_faithfulness(answer, web_result.get("web_context_text", ""), context_label="web search results")
    else:
        notes_context = "\n\n".join(result.get("retrieved_texts", []))
        faithfulness = check_faithfulness(answer, notes_context, context_label="your notes")

    confidence_label = build_confidence_label(faithfulness.get("score"), source, hit_ceiling)

    add_to_history(request.message, answer)

    return {
        "answer": answer,
        "source": source,
        "sources_used": sources_used,
        "web_sources": web_sources,
        "difficulty": request.difficulty,
        "trajectory": result["trajectory"],
        "metrics": metrics,
        "faithfulness": faithfulness,
        "confidence_label": confidence_label,
        "hit_ceiling": hit_ceiling,
    }

@app.get("/")
def root():
    return {"status": "ok"}


# ---------------------------------------------------------------------
# CLEARING ENDPOINTS
# ---------------------------------------------------------------------

@app.post("/clear_history")
def clear_history():
    chat_history.clear()
    return {"message": "conversation history cleared"}

@app.post("/clear_documents")
def clear_documents():
    global collection
    client.delete_collection("my_collection")
    collection = client.get_or_create_collection("my_collection", embedding_function=None)
    chat_history.clear()  # old conversation no longer makes sense against a wiped knowledge base
    return {"message": "all stored documents cleared"}