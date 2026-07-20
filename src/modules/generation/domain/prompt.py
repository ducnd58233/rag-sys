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
6. If MATH EVIDENCE is provided, treat it as the authoritative normalized
   formula for this intent. Copy its formula exactly and do not rewrite it
   from OCR text.
7. When MATH EVIDENCE is provided, do not add any other formula, equation, or
   step list. Explain only the copied formula and its variables.
8. When MATH EVIDENCE is not provided, do not create LaTeX from OCR-damaged
   formula fragments yourself. State that the formula is not safely supported
   if the context is ambiguous.
9. Do not create formulas, variants, or named methods that are not stated in
   context.
10. Do not invent a step-by-step procedure. If context provides a formula but
   not explicit steps, explain the formula and variables only.
11. Do not use bullet or numbered steps unless the context explicitly contains
   procedural steps.
12. Do not mention authors, contribution notes, venue, affiliation, or training
   hardware unless the question asks for those details.
13. Keep the answer compact.
14. If no context supports the question, return exactly:
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
5. Use surrounding text only to disambiguate OCR layout, such as superscripts,
   subscripts, fractions, roots, transposes, parentheses, matrix products,
   concatenation, and projection matrices.
6. If the question does not ask for a formula, calculation, equation, or
   computation, return NO_MATH_EVIDENCE unless the context explicitly states a
   named formula as the direct answer.
7. Do not use memorized domain formulas. Normalize only formulas that are
   visibly present in the supplied context.
8. Do not invent missing variables, operators, denominators, exponents, or
   output multipliers.
9. Do not change multiplication into division, division into multiplication,
   or concatenation into summation unless the context explicitly shows that
   operation.
10. If an OCR fragment is too ambiguous to normalize safely, return the
   safest partial LaTeX plus a short ambiguity note instead of guessing.
11. If no relevant formula appears, return exactly: NO_MATH_EVIDENCE
12. Return only the normalized formula and a short variable note.

""".strip()

_MATH_EVIDENCE_REVIEW_SYSTEM = r"""
You are a math evidence verification agent for a RAG pipeline.
Return structured output with:
- has_math_evidence: boolean
- formula_latex: string

Rules:
1. Compare the candidate math evidence against the supplied context.
2. Set has_math_evidence=false and formula_latex="" if the candidate is prose,
   a definition sentence, or only wraps prose in LaTeX text commands.
3. Set has_math_evidence=false and formula_latex="" if the context does not
   visibly contain a relevant formula, equation, or calculation.
4. Correct OCR layout mistakes only when the source context supports the
   correction through visible symbols, line breaks, or nearby formula prose.
5. Use mathematical notation conventions for OCR layout: stacked text may
   indicate fractions, letters offset after a symbol may indicate superscripts
   or subscripts, and text after a closed parenthesized expression may indicate
   multiplication.
6. Preserve visible trailing factors. If context shows a variable or symbol
   immediately after a closed parenthesized expression and it is not visibly in
   a denominator, keep it as multiplication in formula_latex.
7. Do not use memorized domain formulas. Do not add variables or operations
   absent from context.
8. Do not change multiplication into division, division into multiplication,
   or concatenation into summation unless the context explicitly shows that
   operation.
9. Valid formula evidence must contain an equation or mathematical expression,
   not just explanatory text.
10. If the candidate contains one or more equations and the same function names,
   variables, and operations are visible in context, set formula_latex to the
   candidate unchanged.
11. If the candidate is wrong but safely correctable from context, set
   formula_latex to only the corrected LaTeX formula and a short variable note.
12. If ambiguity remains, use the safest partial LaTeX plus a short ambiguity
    note instead of guessing.
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
            "math and math evidence is present, copy that formula exactly and "
            "do not create another formula or a step list. If math evidence "
            "is absent, do not normalize OCR-damaged formulas yourself. "
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

    def build_math_evidence_review(
        self,
        query: str,
        contexts: Sequence[ContextChunk],
        candidate: str,
    ) -> tuple[str, str]:
        user = (
            f"CONTEXT:\n{_context_block(contexts)}\n\n"
            f"QUESTION: {query}\n\n"
            f"CANDIDATE MATH EVIDENCE:\n{candidate}\n\n"
            "Verify the candidate against the context."
        )
        return _MATH_EVIDENCE_REVIEW_SYSTEM, user


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
