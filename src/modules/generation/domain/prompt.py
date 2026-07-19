from collections.abc import Sequence

from src.modules.generation.domain.models import ContextChunk

REFUSAL_ANSWER = "I could not find relevant information to answer this question."

_SYSTEM = f"""
You are a grounded RAG answerer.
Return JSON with fields:
- refused: boolean
- answer: string
- cited_indices: list of 1-based context indices
- covered_questions: list of 1-based answer-intent indices answered
- unsupported_questions: list of 1-based answer-intent indices not supported by context

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
5. For multi-intent questions, the answer must use one short numbered point
   per supported answer intent. Do not merge intents into one paragraph.
6. If the question asks how to calculate something and context provides a
   formula or steps, include the formula or steps.
7. Do not apologize.
8. Never invent citation indices outside 1..N.
""".strip()

_INTENT_TEXT_SYSTEM = f"""
You answer one RAG retrieval intent.

Rules:
1. Use ONLY the provided context.
2. The context has already been filtered for this intent.
3. If any context contains relevant evidence, answer directly.
4. If the question asks for a formula, calculation, or steps and context has
   formula-like or step-like evidence, include it.
5. If context contains OCR-damaged math, convert it to standard LaTeX math
   notation, but keep the same variables and operations from context.
6. Do not create formulas, variants, or named methods that are not stated in
   context.
7. Do not invent a step-by-step procedure. If context provides a formula but
   not explicit steps, explain the formula and variables only.
8. Do not use bullet or numbered steps unless the context explicitly contains
   procedural steps.
9. Keep the answer to one compact paragraph.
10. If no context supports the question, return exactly:
   {REFUSAL_ANSWER}
""".strip()

_FINAL_TEXT_SYSTEM = f"""
You are a grounded RAG answer refinement agent.

Rules:
1. Use ONLY the provided context and draft answers.
2. Keep exactly the requested answer intents and answer each supported intent.
3. Format the final answer with numbered sections separated by a blank line.
4. If the context contains math, write it as LaTeX.
5. If a calculation intent is supported by context, include the formula and a
   short explanation of the variables.
6. Do not invent a step-by-step procedure. If the context provides a formula
   but not explicit steps, explain the formula and variables only.
7. Do not add unsupported facts or unsupported methods.
8. If none of the contexts support the question, return exactly:
   {REFUSAL_ANSWER}
""".strip()

_MATH_EVIDENCE_SYSTEM = r"""
You are a math OCR normalization agent for a RAG pipeline.

Rules:
1. Return formulas only. Do not summarize prose.
2. Extract only formulas relevant to the question.
3. Convert OCR-damaged math into standard LaTeX display math.
4. Preserve the variables and operations visible in context.
5. Use surrounding text only to infer transpose placement, square-root
   denominator placement, and matrix multiplication order.
6. If context shows Scaled Dot-Product Attention with Q, K, V, softmax, and
   sqrt(dk), normalize it as:
   \[
   \operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
   \]
7. If no relevant formula appears, return exactly: NO_MATH_EVIDENCE
8. Return only the normalized formula and a short variable note.

Example OCR normalization:
- OCR: Attention(Q,K,V ) = softmax( QKT √ dk )V
- LaTeX: \[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]
""".strip()


class GroundedPromptBuilder:
    def build(
        self,
        query: str,
        contexts: Sequence[ContextChunk],
        *,
        sub_questions: Sequence[str] = (),
    ) -> tuple[str, str]:
        user = (
            f"CONTEXT:\n{_context_block(contexts)}\n\n"
            f"QUESTION: {query}"
            f"{_intent_block(sub_questions)}\n\n"
            "Set covered_questions to the answer-intent indices you answered. "
            "Set unsupported_questions to the answer-intent indices that lack "
            "supporting context.\n\n"
            f"If insufficient, set refused=true and answer="
            f'"{REFUSAL_ANSWER}" with cited_indices=[].'
        )
        return _SYSTEM, user

    def build_intent_text_answer(
        self,
        query: str,
        contexts: Sequence[ContextChunk],
        *,
        math_evidence: str = "",
    ) -> tuple[str, str]:
        math_block = (
            "MATH EVIDENCE TO COPY EXACTLY IF RELEVANT:\n" f"{math_evidence}\n\n"
            if math_evidence
            else ""
        )
        user = (
            f"{math_block}"
            f"CONTEXT:\n{_context_block(contexts)}\n\n"
            f"QUESTION: {query}\n\n"
            "Answer this one intent only. If the context includes OCR-damaged "
            "math, normalize it to LaTeX in the answer. If math evidence is "
            "relevant, copy that formula exactly and do not create another. "
            "Do not write bullet or numbered steps unless the context itself "
            "contains procedural steps."
        )
        return _INTENT_TEXT_SYSTEM, user

    def build_final_text_answer(
        self,
        query: str,
        contexts: Sequence[ContextChunk],
        *,
        draft_answer: str,
        sub_questions: Sequence[str],
    ) -> tuple[str, str]:
        user = (
            f"CONTEXT:\n{_context_block(contexts)}\n\n"
            f"QUESTION: {query}"
            f"{_intent_block(sub_questions)}\n\n"
            f"DRAFT ANSWER:\n{draft_answer}\n\n"
            "Return only the final answer text. Use blank lines between "
            "numbered sections. Use LaTeX for formulas."
        )
        return _FINAL_TEXT_SYSTEM, user

    def build_math_evidence(
        self,
        query: str,
        contexts: Sequence[ContextChunk],
    ) -> tuple[str, str]:
        user = (
            f"CONTEXT:\n{_context_block(contexts)}\n\n"
            f"QUESTION: {query}\n\n"
            "Normalize relevant OCR math into LaTeX."
        )
        return _MATH_EVIDENCE_SYSTEM, user


def _context_block(contexts: Sequence[ContextChunk]) -> str:
    return "\n\n".join(
        (
            f"[{i}] chunk_id={c.chunk_id} "
            f"document_id={c.document_id} score={c.score}"
            f"{_intent_label(c)}\n{c.content}"
        )
        for i, c in enumerate(contexts, start=1)
    )


def _intent_block(sub_questions: Sequence[str]) -> str:
    if not sub_questions:
        return ""
    block = "\n".join(
        f"[{i}] {question}" for i, question in enumerate(sub_questions, start=1)
    )
    return (
        f"\n\nANSWER INTENTS:\n{block}\n"
        "Answer each supported intent in its own numbered point. If one intent "
        "is unsupported, state that it is not supported by the provided context."
    )


def _intent_label(context: ContextChunk) -> str:
    indices = context.metadata.get("answer_intent_indices")
    if indices is None:
        indices = context.metadata.get("answer_intent_index")
    if indices is None:
        return ""
    if isinstance(indices, (list, tuple, set, frozenset)):
        values = ", ".join(str(item) for item in indices)
    else:
        values = str(indices)
    return f" answer_intents={values}"
