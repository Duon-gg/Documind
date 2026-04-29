"""MetaGPT Actions for DocuMind RAG Chatbot.

Three-stage pipeline:
  1. AnalyzeQuery   → classify & refine the user's question
  2. RetrieveAndAnswer → fetch context from RAG + generate answer
  3. ReviewAnswer   → quality-check and polish the final response
"""

import json
from metagpt.actions import Action


class AnalyzeQuery(Action):
    """Analyze and classify the user's query before retrieval."""

    name: str = "AnalyzeQuery"

    async def run(self, query: str) -> dict:
        """Classify the query type and refine it for better retrieval.

        Args:
            query: Raw user question.

        Returns:
            Dict with keys: need_code, query_type, refined_query.
        """
        # Detect summary intent from keywords
        summary_keywords = ["tóm tắt", "toàn bộ", "summarize", "tổng quan", "overview", "nội dung chính", "summary"]
        is_summary = any(kw in query.lower() for kw in summary_keywords)

        prompt = f"""Analyze the following user question and return a JSON object with exactly 4 keys:
- "need_code": true if the user wants a code example, false otherwise
- "query_type": one of "explain", "compare", or "code"
- "is_summary": true if the user wants a summary/overview of the entire document, false otherwise
- "refined_query": the question rewritten to be clearer and more specific for document retrieval

User question: {query}

Return ONLY valid JSON, no markdown, no explanation, no extra text.
Example format: {{"need_code": false, "query_type": "explain", "is_summary": false, "refined_query": "..."}}"""

        response = await self._aask(prompt)

        # Parse JSON from LLM response, handling possible markdown wrapping
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        try:
            result = json.loads(text)
        except json.JSONDecodeError:
            # Fallback if LLM didn't return valid JSON
            result = {
                "need_code": False,
                "query_type": "explain",
                "is_summary": is_summary,
                "refined_query": query,
            }

        # Override is_summary from keyword detection if LLM missed it
        if is_summary and not result.get("is_summary"):
            result["is_summary"] = True

        return result


class RetrieveAndAnswer(Action):
    """Retrieve context from knowledge base and generate an answer."""

    name: str = "RetrieveAndAnswer"

    async def run(self, query: str, need_code: bool = False) -> str:
        """Query the RAG engine and return an answer.

        Args:
            query: The refined query for retrieval.
            need_code: Whether to request a code example in the answer.

        Returns:
            The generated answer string.
        """
        from rag_engine import RAGEngine

        engine = RAGEngine()
        engine.load_engine()

        # Append code request if needed
        full_query = query
        if need_code:
            full_query += "\n\nInclude a Python code example."

        result = await engine.query(full_query)
        return str(result)


class ReviewAnswer(Action):
    """Review and polish the generated answer for completeness."""

    name: str = "ReviewAnswer"

    async def run(self, question: str, answer: str) -> str:
        """Check the answer for completeness and polish it.

        Args:
            question: The original user question.
            answer: The draft answer from RetrieveAndAnswer.

        Returns:
            The final polished answer.
        """
        prompt = f"""You are a quality reviewer. Review the following answer for the given question.

Question: {question}

Draft Answer: {answer}

Your task:
1. Check if the answer fully addresses the question
2. If information is missing or unclear, add supplementary details
3. Fix any formatting issues
4. Keep the answer concise but complete

Return ONLY the final polished answer, nothing else."""

        result = await self._aask(prompt)
        return result


class SummarizeDocument(Action):
    """Summarize an entire PDF document using Map Reduce strategy."""

    name: str = "SummarizeDocument"

    async def run(self, pdf_path: str) -> str:
        """Run Map Reduce summarization on a PDF file.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            A structured Vietnamese summary of the entire document.
        """
        from rag_engine import RAGEngine

        engine = RAGEngine()
        result = await engine.map_reduce_summary(pdf_path)
        return result
