"""DocuMind Agent Role — orchestrates the 3-step RAG pipeline.

Pipeline:  AnalyzeQuery → RetrieveAndAnswer → ReviewAnswer
"""

from metagpt.roles import Role
from metagpt.schema import Message
from metagpt.logs import logger

from agents.actions import AnalyzeQuery, RetrieveAndAnswer, ReviewAnswer, SummarizeDocument
from pathlib import Path

DATA_DIR = Path("E:/projects/documind/data")


class DocuMindAgent(Role):
    """RAG Study Assistant that analyzes, retrieves, and reviews answers."""

    name: str = "DocuMind"
    profile: str = "RAG Study Assistant"
    goal: str = "Answer user questions accurately using the knowledge base"
    constraints: str = "Always ground answers in retrieved documents"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_actions([AnalyzeQuery, RetrieveAndAnswer, ReviewAnswer])

    async def _act(self) -> Message:
        """Execute the RAG pipeline or Map Reduce summary based on intent.

        Step 1: AnalyzeQuery   — classify & refine the user's question
        If summary detected → SummarizeDocument (Map Reduce)
        Else → Step 2: RetrieveAndAnswer + Step 3: ReviewAnswer
        """
        # Get the latest user query from memory
        memories = self.get_memories(k=1)
        if not memories:
            return Message(content="No query found.", role=self.profile)

        query = memories[-1].content
        logger.info(f"DocuMind processing: {query}")

        # Step 1: Analyze the query
        logger.info("Step 1: Analyzing query...")
        analysis = await AnalyzeQuery().run(query)
        refined = analysis.get("refined_query", query)
        need_code = analysis.get("need_code", False)
        query_type = analysis.get("query_type", "explain")
        is_summary = analysis.get("is_summary", False)
        logger.info(f"  → type={query_type}, need_code={need_code}, is_summary={is_summary}")

        # Summary branch: use Map Reduce instead of RAG
        if is_summary:
            logger.info("📝 Summary detected → running Map Reduce...")
            pdf_files = list(DATA_DIR.glob("*.pdf"))
            if not pdf_files:
                return Message(content="⚠️ Không tìm thấy file PDF trong thư mục data/", role=self.profile)
            # Summarize the first PDF
            result = await SummarizeDocument().run(str(pdf_files[0]))
            logger.info(f"  → summary done, {len(result)} chars")
            return Message(content=result, role=self.profile)

        # Normal RAG branch
        # Step 2: Retrieve & Answer
        logger.info("Step 2: Retrieving context and generating answer...")
        raw_answer = await RetrieveAndAnswer().run(
            query=refined,
            need_code=need_code,
        )
        logger.info(f"  → draft answer length: {len(raw_answer)} chars")

        # Step 3: Review & Polish
        logger.info("Step 3: Reviewing and polishing answer...")
        final_answer = await ReviewAnswer().run(
            question=query,
            answer=raw_answer,
        )
        logger.info("  → final answer ready")

        return Message(content=final_answer, role=self.profile)


async def run_agent(question: str) -> str:
    """Convenience function to run DocuMindAgent with a single question.

    Args:
        question: The user's question string.

    Returns:
        The final polished answer string.
    """
    agent = DocuMindAgent()
    result = await agent.run(question)
    return result.content if result else "No answer generated."
