from typing import Any, Protocol

from config import BEDROCK_ANSWER_MODEL_ID, BEDROCK_REGION
from llm_provider import generate_text, parse_json_object


class AnswerLLMProvider(Protocol):
    """Interface for answer generation providers."""

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        ...


class BedrockAnswerProvider:
    """Amazon Bedrock adapter using the Converse Runtime API."""

    def __init__(
        self,
        model_id: str = BEDROCK_ANSWER_MODEL_ID,
        region_name: str = BEDROCK_REGION,
        temperature: float = 0.1,
    ):
        self.model_id = model_id
        self.region_name = region_name
        self.temperature = temperature

    def generate(self, prompt: str, system_instruction: str | None = None) -> str:
        return generate_text(
            model_id=self.model_id,
            region_name=self.region_name,
            prompt=prompt,
            system_instruction=system_instruction,
            temperature=self.temperature,
        )


def build_answer_provider(
    bedrock_model_id: str | None = None,
) -> AnswerLLMProvider:
    return BedrockAnswerProvider(model_id=bedrock_model_id or BEDROCK_ANSWER_MODEL_ID)


def answer_from_grounded_context(
    query: str,
    grounded_context: dict[str, Any],
    provider: AnswerLLMProvider | None = None,
) -> dict[str, Any]:
    """Generate an answer from prebuilt SQL, graph, and hybrid chunks."""
    chunks = grounded_context.get("chunks", [])
    context_text = grounded_context.get("context_text", "No matching records found.")
    llm = provider or build_answer_provider()
    system_instruction = (
        "You are a strict context-grounded assistant. Answer only with facts explicitly stated "
        "in the provided context. Do not add outside knowledge or assumptions. If the context "
        "does not contain the answer, say so clearly."
    )
    prompt = f"""
User Question:
{query}

Grounded Context:
{context_text}

Return one valid JSON object with exactly these fields:
{{
  "answer": "A clear answer using only the grounded context.",
  "source_chunk_names": ["Exact Chunk Name values that support the answer"]
}}
Use exact Chunk Name values from the context. Return an empty array if no chunk supports the answer.
""".strip()

    response = parse_json_object(llm.generate(prompt=prompt, system_instruction=system_instruction))
    source_names = response.get("source_chunk_names", [])
    chunks_by_name = {chunk["Chunk Name"]: chunk for chunk in chunks}
    cited_chunks = [chunks_by_name[name] for name in source_names if name in chunks_by_name]

    source_lines = ["Sources used:"]
    if not cited_chunks:
        source_lines.append("- None")
    for chunk in cited_chunks:
        source_lines.extend(
            [
                f"- Chunk Name: {chunk['Chunk Name']}",
                f"  Source: {chunk['Source']}",
            ]
        )

    return {
        "answer": f"{response['answer']}\n\n" + "\n".join(source_lines),
        "source_chunk_names": [chunk["Chunk Name"] for chunk in cited_chunks],
        "sources": cited_chunks,
        "chunks": chunks,
        "context_text": context_text,
    }


def answer_general_question(
    query: str,
    provider: AnswerLLMProvider | None = None,
) -> dict[str, Any]:
    """Answer a general question directly, without retrieval or grounded context."""
    llm = provider or build_answer_provider()
    system_instruction = (
        "You are a helpful assistant. Answer the user's general question clearly and concisely. "
        "Do not claim to have searched personal records or external sources."
    )
    answer = llm.generate(prompt=query, system_instruction=system_instruction).strip()
    return {
        "answer": answer,
        "source_chunk_names": [],
        "sources": [],
        "chunks": [],
        "context_text": "",
    }


