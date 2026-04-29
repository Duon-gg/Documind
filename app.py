"""DocuMind 🧠 — RAG Study Assistant powered by MetaGPT + Groq."""

import streamlit as st
import subprocess
import json
import sys
from pathlib import Path

DATA_DIR = Path("E:/projects/documind/data")
STORAGE_DIR = Path("E:/projects/documind/storage")
VENV_PYTHON = Path("E:/projects/documind-env/Scripts/python.exe")
PROJECT_DIR = Path("E:/projects/documind")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13141f 0%, #1a1b2e 100%);
    border-right: 1px solid #2d2e3f;
}

.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 500;
}

.status-ready {
    background: rgba(16, 185, 129, 0.12);
    color: #10b981;
    border: 1px solid rgba(16, 185, 129, 0.25);
}

.status-empty {
    background: rgba(245, 158, 11, 0.12);
    color: #f59e0b;
    border: 1px solid rgba(245, 158, 11, 0.25);
}

.step-box {
    background: #1e1f2e;
    border: 1px solid #2d2e3f;
    border-radius: 10px;
    padding: 12px 16px;
    margin: 6px 0;
    font-size: 0.9rem;
}

.step-active {
    border-left: 3px solid #7c3aed;
    background: rgba(124, 58, 237, 0.08);
}

.step-done {
    border-left: 3px solid #10b981;
    opacity: 0.85;
}

.file-chip {
    display: inline-block;
    background: rgba(124, 58, 237, 0.1);
    border: 1px solid rgba(124, 58, 237, 0.2);
    color: #a78bfa;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.8rem;
    margin: 3px 2px;
}

.main-header {
    text-align: center;
    padding: 1.5rem 0 1rem;
}

.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #7c3aed 0%, #a78bfa 50%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}

.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
}

.metric-card {
    background: #1e1f2e;
    border: 1px solid #2d2e3f;
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}

.metric-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #a78bfa;
}

.metric-card .label {
    font-size: 0.8rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────
def get_pdf_names() -> list[str]:
    if not DATA_DIR.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        return []
    return [f.name for f in DATA_DIR.glob("*.pdf")]


def is_kb_ready() -> bool:
    return STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir())


def run_query_subprocess(question: str) -> dict:
    """Run the RAG pipeline in a separate Python process to avoid DLL conflicts."""
    result = subprocess.run(
        [str(VENV_PYTHON), str(PROJECT_DIR / "query_worker.py"), question],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
        timeout=120,
    )

    output = result.stdout
    if "__RESULT_JSON__" in output:
        json_str = output.split("__RESULT_JSON__")[1].strip()
        return json.loads(json_str)
    else:
        return {"error": result.stderr or "Unknown error", "answer": "", "steps": []}


def run_build_subprocess() -> bool:
    """Build the knowledge base in a separate process."""
    script = """
import sys
sys.path.insert(0, "E:/projects/documind")
from rag_engine import RAGEngine
engine = RAGEngine()
engine.build_engine()
print("__BUILD_OK__")
"""
    result = subprocess.run(
        [str(VENV_PYTHON), "-c", script],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_DIR),
        timeout=300,
    )
    return "__BUILD_OK__" in result.stdout


def run_rebuild_subprocess() -> bool:
    """Rebuild KB from scratch in a separate process."""
    import shutil
    if STORAGE_DIR.exists():
        shutil.rmtree(STORAGE_DIR)
        STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    return run_build_subprocess()


# ── Session State ────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "kb_status" not in st.session_state:
    st.session_state.kb_status = "ready" if is_kb_ready() else "empty"


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    st.caption("RAG Study Assistant")
    st.divider()

    st.markdown("### 📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drop PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        saved = 0
        for f in uploaded_files:
            dest = DATA_DIR / f.name
            if not dest.exists():
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(f.getvalue())
                saved += 1
        if saved > 0:
            st.success(f"Saved {saved} new file(s)!")

    pdf_names = get_pdf_names()
    pdf_count = len(pdf_names)

    if pdf_count > 0:
        with st.expander(f"📁 {pdf_count} file(s) in knowledge base", expanded=False):
            for name in pdf_names:
                st.markdown(f'<span class="file-chip">📄 {name}</span>', unsafe_allow_html=True)
    else:
        st.info("No PDF files yet. Upload some above!")

    st.divider()

    st.markdown("### ⚡ Knowledge Base")

    if st.session_state.kb_status == "ready":
        st.markdown('<span class="status-badge status-ready">✅ Ready</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-empty">⚠️ Not Built</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        build_btn = st.button("📚 Build", use_container_width=True, type="primary")
    with col2:
        rebuild_btn = st.button("🔄 Rebuild", use_container_width=True)

    if build_btn or rebuild_btn:
        if pdf_count == 0:
            st.error("Upload PDF files first!")
        else:
            action = "Rebuilding" if rebuild_btn else "Building"
            with st.spinner(f"{action} from {pdf_count} PDF(s)..."):
                try:
                    ok = run_rebuild_subprocess() if rebuild_btn else run_build_subprocess()
                    if ok:
                        st.session_state.kb_status = "ready"
                        st.success("✅ Done!")
                        st.rerun()
                    else:
                        st.error("Build failed. Check console logs.")
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()

    st.markdown("### ℹ️ Tech Stack")
    st.caption("🤖 Llama 3.3 70B (Groq)")
    st.caption("📊 BGE-small-en-v1.5")
    st.caption("💾 ChromaDB")
    st.caption("🏗️ MetaGPT 0.8.2")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>DocuMind 🧠</h1>
    <p>Upload your documents. Ask anything. Get intelligent answers.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="metric-card"><div class="value">{pdf_count}</div><div class="label">Documents</div></div>', unsafe_allow_html=True)
with c2:
    ready = st.session_state.kb_status == "ready"
    st.markdown(f'<div class="metric-card"><div class="value" style="color: {"#10b981" if ready else "#f59e0b"};">{"●" if ready else "○"}</div><div class="label">KB: {"Ready" if ready else "Not Built"}</div></div>', unsafe_allow_html=True)
with c3:
    qcount = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.markdown(f'<div class="metric-card"><div class="value">{qcount}</div><div class="label">Questions Asked</div></div>', unsafe_allow_html=True)

st.markdown("")

# ── Chat History ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🧠"):
        st.markdown(msg["content"])

# ── Chat Input ───────────────────────────────────────────────
if prompt := st.chat_input("Hỏi về tài liệu của bạn..."):

    if st.session_state.kb_status != "ready":
        st.warning("⚠️ Build the knowledge base first! Sidebar → 📚 Build")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🧠"):
            step1 = st.empty()
            step2 = st.empty()
            step3 = st.empty()
            answer_box = st.empty()

            # Show initial progress
            step1.markdown('<div class="step-box step-active">🔍 <strong>Step 1/3:</strong> Analyzing question...</div>', unsafe_allow_html=True)
            step2.markdown('<div class="step-box">📚 <strong>Step 2/3:</strong> Waiting...</div>', unsafe_allow_html=True)
            step3.markdown('<div class="step-box">✨ <strong>Step 3/3:</strong> Waiting...</div>', unsafe_allow_html=True)

            try:
                result = run_query_subprocess(prompt)

                if result.get("error"):
                    step1.empty()
                    step2.empty()
                    step3.empty()
                    answer_box.error(f"❌ {result['error']}")
                else:
                    steps = result.get("steps", [])

                    # Update step indicators
                    if len(steps) >= 1:
                        s = steps[0]
                        step1.markdown(
                            f'<div class="step-box step-done">✅ <strong>Step 1/3:</strong> Analyzed — type: <code>{s.get("query_type","?")}</code> | code: <code>{s.get("need_code","?")}</code></div>',
                            unsafe_allow_html=True,
                        )
                    if len(steps) >= 2:
                        s = steps[1]
                        step2.markdown(
                            f'<div class="step-box step-done">✅ <strong>Step 2/3:</strong> Retrieved — {s.get("chars",0)} chars</div>',
                            unsafe_allow_html=True,
                        )
                    if len(steps) >= 3:
                        step3.markdown(
                            '<div class="step-box step-done">✅ <strong>Step 3/3:</strong> Reviewed & polished</div>',
                            unsafe_allow_html=True,
                        )

                    final = result.get("answer", "No answer generated.")
                    answer_box.markdown(final)
                    st.session_state.messages.append({"role": "assistant", "content": final})

            except subprocess.TimeoutExpired:
                step1.empty()
                step2.empty()
                step3.empty()
                answer_box.error("❌ Query timed out (>120s). Try a simpler question.")
            except Exception as e:
                step1.empty()
                step2.empty()
                step3.empty()
                answer_box.error(f"❌ Error: {e}")
