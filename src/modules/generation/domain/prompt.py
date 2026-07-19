from collections.abc import Sequence

from src.modules.generation.domain.models import ContextChunk

REFUSAL_ANSWER = "I could not find relevant information to answer this question."

_SYSTEM = f"""
You are a grounded RAG answerer.
Return JSON with fields:
- refused: boolean
- answer: string
- cited_indices: list of 1-based context indices

Decision rule:
- If ANY context contains information that helps answer the question,
  set refused=false, write a concise answer from that context, and cite those indices.
- Set refused=true ONLY when none of the contexts support the question.
- When refused=true, answer must be exactly: "{REFUSAL_ANSWER}" and cited_indices=[].

Rules:
1. Use ONLY the provided context. No prior knowledge.
2. Cover every supported answer intent. Do not silently skip an intent.
3. If an answer intent is not supported by context, say that part is not
   supported by the provided context.
4. Prefer answering over refusing when evidence exists.
5. Do not apologize.
6. Never invent citation indices outside 1..N.
""".strip()


class GroundedPromptBuilder:
    def build(
        self,
        query: str,
        contexts: Sequence[ContextChunk],
        *,
        sub_questions: Sequence[str] = (),
    ) -> tuple[str, str]:
        block = "\n\n".join(
            (
                f"[{i}] chunk_id={c.chunk_id} "
                f"document_id={c.document_id} score={c.score}\n{c.content}"
            )
            for i, c in enumerate(contexts, start=1)
        )
        intent_block = "\n".join(
            f"[{i}] {question}" for i, question in enumerate(sub_questions, start=1)
        )
        intent_instruction = (
            f"\n\nANSWER INTENTS:\n{intent_block}\n"
            "Answer each supported intent. If one intent is unsupported, "
            "state that it is not supported by the provided context."
            if sub_questions
            else ""
        )
        user = (
            f"CONTEXT:\n{block}\n\n"
            f"QUESTION: {query}"
            f"{intent_instruction}\n\n"
            f"If insufficient, set refused=true and answer="
            f'"{REFUSAL_ANSWER}" with cited_indices=[].'
        )
        return _SYSTEM, user
