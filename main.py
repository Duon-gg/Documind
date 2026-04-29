"""FastAPI backend for DocuMind RAG chatbot."""

import subprocess
import json
import shutil
import time
import re
import sys
from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT_DIR = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_DIR / "data"
STORAGE_DIR = PROJECT_DIR / "storage"
VENV_PYTHON = sys.executable  # Use current Python interpreter

GROQ_DAILY_LIMIT = 100_000  # Groq free tier TPD

app = FastAPI(title="DocuMind API")


# ── Token Tracker ──
class TokenTracker:
    """Track token usage across all chat requests in this session."""

    def __init__(self):
        self.total_tokens = 0
        self.total_requests = 0
        self.history = []

    def add(self, prompt_tokens: int, completion_tokens: int, elapsed: float):
        total = prompt_tokens + completion_tokens
        self.total_tokens += total
        self.total_requests += 1
        self.history.append({
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "elapsed": elapsed,
            "timestamp": time.strftime("%H:%M:%S"),
        })

    def get_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> dict:
        total = prompt_tokens + completion_tokens
        remaining = max(0, GROQ_DAILY_LIMIT - self.total_tokens)
        remaining_pct = round(remaining / GROQ_DAILY_LIMIT * 100, 1)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total,
            "total_session_tokens": self.total_tokens,
            "total_requests": self.total_requests,
            "groq_limit": GROQ_DAILY_LIMIT,
            "remaining_percent": remaining_pct,
        }


tracker = TokenTracker()


def fmt_size(b: int) -> str:
    if b < 1024: return f"{b} B"
    if b < 1048576: return f"{b/1024:.1f} KB"
    return f"{b/1048576:.1f} MB"


# ── Static files ──
app.mount("/static", StaticFiles(directory=str(PROJECT_DIR / "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(str(PROJECT_DIR / "static" / "index.html"))


@app.get("/status")
async def status():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [{"name": f.name, "size": fmt_size(f.stat().st_size)} for f in DATA_DIR.glob("*.pdf")]
    kb_ready = STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir())
    return {"files": files, "kb_ready": kb_ready}


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...)):
    t0 = time.time()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.filename and f.filename.endswith(".pdf"):
            dest = DATA_DIR / f.filename
            dest.write_bytes(await f.read())
    all_files = [{"name": p.name, "size": fmt_size(p.stat().st_size)} for p in DATA_DIR.glob("*.pdf")]
    return {"files": all_files, "elapsed": round(time.time() - t0, 2)}


@app.post("/build")
async def build():
    t0 = time.time()
    script = f'import sys; sys.path.insert(0,"{PROJECT_DIR.as_posix()}"); from rag_engine import RAGEngine; e=RAGEngine(); e.build_engine(); print("__OK__")'
    try:
        r = subprocess.run([str(VENV_PYTHON), "-c", script], capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=300)
        elapsed = round(time.time() - t0, 2)
        if "__OK__" in r.stdout:
            return {"status": "ok", "elapsed": elapsed}
        return {"status": "error", "error": r.stderr[-300:] if r.stderr else "Unknown", "elapsed": elapsed}
    except Exception as e:
        return {"status": "error", "error": str(e), "elapsed": round(time.time() - t0, 2)}


class ChatReq(BaseModel):
    question: str


@app.post("/chat")
async def chat(req: ChatReq):
    t0 = time.time()
    try:
        r = subprocess.run(
            [str(VENV_PYTHON), str(PROJECT_DIR / "query_worker.py"), req.question],
            capture_output=True, text=True, cwd=str(PROJECT_DIR), timeout=180,
        )
        elapsed = round(time.time() - t0, 2)
        if "__RESULT_JSON__" in r.stdout:
            data = json.loads(r.stdout.split("__RESULT_JSON__")[1].strip())
            err = data.get("error") or ""
            # Check for Groq 429 rate limit
            if "429" in err or "rate_limit" in err:
                retry = "unknown"
                m = re.search(r"try again in ([\d]+m[\d.]+s|[\d.]+s)", err)
                if m:
                    retry = m.group(1)
                return JSONResponse({
                    "error": "rate_limit",
                    "message": "Hết quota Groq hôm nay!",
                    "retry_after": retry,
                    "tip": "Dùng câu hỏi ngắn hơn hoặc chờ reset lúc nửa đêm",
                }, status_code=429)
            if err:
                return {"error": err, "elapsed": elapsed}

            steps = data.get("steps", [])
            qtype = steps[0].get("query_type", "explain") if steps else "explain"
            usage_raw = data.get("usage") or {}
            pt = usage_raw.get("prompt_tokens", 0)
            ct = usage_raw.get("completion_tokens", 0)

            # Track tokens
            tracker.add(pt, ct, elapsed)

            return {
                "answer": data.get("answer", ""),
                "query_type": qtype,
                "elapsed": elapsed,
                "steps": [{"step": s["step"], "elapsed": s.get("elapsed", 0)} for s in steps],
                "usage": tracker.get_usage(pt, ct),
            }
        # Check stderr for 429 too
        full_err = (r.stderr or "") + (r.stdout or "")
        if "429" in full_err or "rate_limit" in full_err:
            retry = "unknown"
            m = re.search(r"try again in ([\d]+m[\d.]+s|[\d.]+s)", full_err)
            if m:
                retry = m.group(1)
            return JSONResponse({
                "error": "rate_limit",
                "message": "Hết quota Groq hôm nay!",
                "retry_after": retry,
                "tip": "Dùng câu hỏi ngắn hơn hoặc chờ reset lúc nửa đêm",
            }, status_code=429)
        return {"error": r.stderr[-300:] if r.stderr else "No response", "elapsed": elapsed}
    except subprocess.TimeoutExpired:
        return {"error": "Timeout (>180s)", "elapsed": round(time.time() - t0, 2)}
    except Exception as e:
        return {"error": str(e), "elapsed": round(time.time() - t0, 2)}


@app.get("/stats")
async def get_stats():
    """Return full session token usage statistics."""
    return {
        "total_tokens": tracker.total_tokens,
        "total_requests": tracker.total_requests,
        "groq_limit": GROQ_DAILY_LIMIT,
        "remaining_percent": round(max(0, GROQ_DAILY_LIMIT - tracker.total_tokens) / GROQ_DAILY_LIMIT * 100, 1),
        "history": tracker.history,
    }


@app.get("/files")
async def list_files():
    """List all PDF files in the data directory with sizes."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = [{"name": f.name, "size": fmt_size(f.stat().st_size)} for f in DATA_DIR.glob("*.pdf")]
    return {"files": files}


@app.delete("/delete/{filename}")
async def delete_file(filename: str):
    """Delete a single PDF and wipe the vector store (requires KB rebuild)."""
    t0 = time.time()
    target = DATA_DIR / filename
    if not target.exists():
        return {"status": "error", "error": "File not found"}
    target.unlink()
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return {"status": "deleted", "file": filename, "elapsed": round(time.time() - t0, 2)}


@app.delete("/clear")
async def clear_all():
    """Delete all PDFs and wipe the vector store."""
    t0 = time.time()
    if DATA_DIR.exists():
        for f in DATA_DIR.glob("*.pdf"):
            f.unlink()
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return {"status": "cleared", "elapsed": round(time.time() - t0, 2)}
