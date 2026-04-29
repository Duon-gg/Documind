"""RAG Engine - Vector store and retrieval logic using MetaGPT SimpleEngine."""

from metagpt.rag.engines import SimpleEngine
from metagpt.rag.schema import (
    ChromaRetrieverConfig,
    ChromaIndexConfig,
)
from metagpt.rag.factories.llm import get_rag_llm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pypdf import PdfReader
import asyncio
from pathlib import Path

DATA_DIR = Path(__file__).parent.resolve() / "data"
STORAGE_DIR = Path(__file__).parent.resolve() / "storage"

# Local embedding model (free, no API key needed)
EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


class RAGEngine:
    """RAG pipeline built on MetaGPT's SimpleEngine with ChromaDB backend."""

    def __init__(self):
        self.engine = None

    def build_engine(self):
        """Build knowledge base from PDF files in DATA_DIR.

        Scans DATA_DIR for all PDF files, creates embeddings,
        and persists the vector store to STORAGE_DIR via ChromaDB.
        """
        input_files = [str(f) for f in DATA_DIR.glob("*.pdf")]

        if not input_files:
            print("⚠️ No PDF files found in", DATA_DIR)
            print("   Please add PDF files to the data/ folder first.")
            return

        print(f"📄 Found {len(input_files)} PDF file(s):")
        for f in input_files:
            print(f"   - {Path(f).name}")

        print("\n🔨 Building knowledge base...")

        self.engine = SimpleEngine.from_docs(
            input_files=input_files,
            embed_model=EMBED_MODEL,
            llm=get_rag_llm(),
            retriever_configs=[
                ChromaRetrieverConfig(
                    persist_path=str(STORAGE_DIR),
                )
            ],
        )

        print("✅ Knowledge base built successfully!")

    def load_engine(self):
        """Load engine from existing storage, or build if not available.

        Checks if STORAGE_DIR exists and contains data.
        If yes, loads the index from ChromaDB storage.
        If no, falls back to building from scratch.
        """
        if STORAGE_DIR.exists() and any(STORAGE_DIR.iterdir()):
            print("📂 Found existing knowledge base, loading from storage...")
            try:
                self.engine = SimpleEngine.from_index(
                    index_config=ChromaIndexConfig(
                        persist_path=str(STORAGE_DIR),
                    ),
                    embed_model=EMBED_MODEL,
                    llm=get_rag_llm(),
                )
                print("✅ Knowledge base loaded successfully!")
            except Exception as e:
                print(f"⚠️ Failed to load from storage: {e}")
                print("🔄 Rebuilding knowledge base...")
                self.build_engine()
        else:
            print("📂 No existing knowledge base found.")
            self.build_engine()

    async def query(self, question: str) -> str:
        """Query the RAG engine with a question.

        Args:
            question: The question to ask against the knowledge base.

        Returns:
            The answer as a string.

        Raises:
            RuntimeError: If the engine has not been initialized.
        """
        if self.engine is None:
            raise RuntimeError(
                "Engine not initialized. Call load_engine() or build_engine() first."
            )

        response = await self.engine.aquery(question)
        return str(response)

    async def map_reduce_summary(self, pdf_path: str) -> str:
        """Summarize an entire PDF using Map Reduce strategy.

        1. Extract all text page-by-page
        2. Group into chunks of 5 pages
        3. MAP: Summarize each chunk in parallel via LLM
        4. REDUCE: Combine all summaries into a final structured summary

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A comprehensive Vietnamese summary of the entire document.
        """
        from metagpt.llm import LLM
        llm = LLM()

        # 1. Extract text per page
        reader = PdfReader(pdf_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        print(f"📖 Extracted {len(pages)} pages from {Path(pdf_path).name}")

        # 2. Group into chunks of 5 pages
        chunk_size = 5
        chunks = []
        for i in range(0, len(pages), chunk_size):
            chunk = " ".join(pages[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)

        if not chunks:
            return "⚠️ Không thể trích xuất text từ PDF này."

        print(f"🔀 Split into {len(chunks)} chunks for Map phase")

        # 3. MAP: Summarize each chunk in parallel
        async def summarize_chunk(chunk: str, idx: int) -> str:
            prompt = f"""Tóm tắt ngắn gọn đoạn văn sau (tối đa 200 từ):

{chunk[:3000]}

Chỉ trả về tóm tắt, không giải thích thêm."""
            result = await llm.aask(prompt)
            print(f"  ✅ Chunk {idx + 1}/{len(chunks)} done")
            return result

        summaries = await asyncio.gather(*[
            summarize_chunk(chunk, i)
            for i, chunk in enumerate(chunks)
        ])

        print(f"📝 Map phase complete, reducing {len(summaries)} summaries...")

        # 4. REDUCE: Combine all summaries into final
        combined = "\n\n".join([
            f"Phần {i + 1}: {s}"
            for i, s in enumerate(summaries)
        ])

        final_prompt = f"""Dựa trên các tóm tắt từng phần sau,
viết một tóm tắt tổng thể hoàn chỉnh bằng tiếng Việt:

{combined}

Tóm tắt phải:
- Bao gồm tất cả ý chính
- Có cấu trúc rõ ràng (dùng bullet points)
- Ngắn gọn súc tích
- Bằng tiếng Việt"""

        result = await llm.aask(final_prompt)
        print("✅ Reduce phase complete!")
        return result


if __name__ == "__main__":
    engine = RAGEngine()
    engine.load_engine()
    result = asyncio.run(engine.query("Tóm tắt nội dung tài liệu"))
    print(result)
