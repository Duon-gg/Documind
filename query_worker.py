"""Query worker — runs RAG pipeline in a separate process to avoid DLL conflicts."""
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

from rag_engine import RAGEngine
from agents.actions import AnalyzeQuery, RetrieveAndAnswer, ReviewAnswer, SummarizeDocument

DATA_DIR = Path(__file__).parent.resolve() / "data"

# Token estimation: ~1 token per 4 chars (English), ~1 token per 2 chars (Vietnamese/CJK)
def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii = len(text) - ascii_chars
    return int(ascii_chars / 4 + non_ascii / 2)


async def process_query(question: str) -> dict:
    """Run the pipeline and return results as JSON with token estimates."""
    result = {"steps": [], "answer": "", "error": None, "usage": None}
    prompt_tokens = 0
    completion_tokens = 0

    try:
        # Step 1: Analyze
        t1 = time.time()
        analysis = await AnalyzeQuery().run(question)
        e1 = round(time.time() - t1, 2)
        refined = analysis.get("refined_query", question)
        need_code = analysis.get("need_code", False)
        query_type = analysis.get("query_type", "explain")
        is_summary = analysis.get("is_summary", False)

        p1 = estimate_tokens(question) + 120  # prompt overhead
        c1 = estimate_tokens(json.dumps(analysis))
        prompt_tokens += p1
        completion_tokens += c1

        result["steps"].append({
            "step": 1, "status": "done", "elapsed": e1,
            "query_type": "summary" if is_summary else query_type,
            "need_code": need_code, "is_summary": is_summary,
            "refined": refined,
        })

        # Summary branch: Map Reduce
        if is_summary:
            pdf_files = list(DATA_DIR.glob("*.pdf"))
            if not pdf_files:
                result["error"] = "Không tìm thấy file PDF trong thư mục data/"
                return result

            t2 = time.time()
            summary = await SummarizeDocument().run(str(pdf_files[0]))
            e2 = round(time.time() - t2, 2)

            p2 = estimate_tokens(str(pdf_files[0].stat().st_size)) + 500
            c2 = estimate_tokens(summary)
            prompt_tokens += p2
            completion_tokens += c2

            result["steps"].append({"step": 2, "status": "done", "elapsed": e2, "info": f"Map Reduce on {pdf_files[0].name}"})
            result["steps"].append({"step": 3, "status": "done", "elapsed": 0})
            result["answer"] = summary
            result["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}
            return result

        # Normal RAG branch
        # Step 2: Retrieve
        t2 = time.time()
        raw = await RetrieveAndAnswer().run(query=refined, need_code=need_code)
        e2 = round(time.time() - t2, 2)

        p2 = estimate_tokens(refined) + 300  # RAG context overhead
        c2 = estimate_tokens(raw)
        prompt_tokens += p2
        completion_tokens += c2

        result["steps"].append({"step": 2, "status": "done", "elapsed": e2, "chars": len(raw)})

        # Step 3: Review
        t3 = time.time()
        final = await ReviewAnswer().run(question=question, answer=raw)
        e3 = round(time.time() - t3, 2)

        p3 = estimate_tokens(question + raw) + 100
        c3 = estimate_tokens(final)
        prompt_tokens += p3
        completion_tokens += c3

        result["steps"].append({"step": 3, "status": "done", "elapsed": e3})
        result["answer"] = final
        result["usage"] = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}

    except Exception as e:
        result["error"] = str(e)

    return result


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else ""
    if not question:
        print(json.dumps({"error": "No question provided"}))
        sys.exit(1)

    out = asyncio.run(process_query(question))
    print("__RESULT_JSON__")
    print(json.dumps(out, ensure_ascii=False))
