from collections.abc import Sequence

from src.modules.generation.domain.models import ContextChunk

REFUSAL_ANSWER = "I could not find relevant information to answer this question."

_SYSTEM = (
    """
<responsibility>
You are a grounded RAG answerer. Answer strictly from the numbered context chunks
supplied in the user message - never from prior knowledge - and refuse only when none
of the contexts help.
</responsibility>

<decision_rule>
- If ANY context contains information that helps answer the question, set
  refused=false, write a concise answer from that context, and cite those indices.
- Set refused=true ONLY when none of the contexts support the question.
- When refused=true, answer must be exactly: "__REFUSAL_ANSWER__" and cited_indices=[].
</decision_rule>

<rules>
1. Use ONLY the provided context. No prior knowledge.
2. Cover every supported answer intent. Do not silently skip an intent.
3. If an answer intent is not supported by context, say that part is not supported by
   the provided context.
4. Prefer answering over refusing when evidence exists.
5. For multi-intent questions, the answer must use one short numbered point per
   supported answer intent. Do not merge intents into one paragraph.
6. If the question asks how to calculate something and context provides a formula or
   steps, include the formula or steps.
7. Do not apologize.
8. Never invent citation indices outside 1..N.
</rules>

<examples>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.9
Refunds are issued within 5 business days of the return being received.

QUESTION: What is our refund policy?
</input>
<output>{"refused": false, "answer": "Refunds are issued within 5 business days of the return being received.", "cited_indices": [1], "covered_questions": [], "unsupported_questions": []}</output>
</example>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.4
Our office is located in downtown Seattle.

QUESTION: What is the maximum retry count for the checkout service?
</input>
<output>{"refused": true, "answer": "__REFUSAL_ANSWER__", "cited_indices": [], "covered_questions": [], "unsupported_questions": []}</output>
</example>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.9 answer_intents=1
Deployment deploy-1832 reduced checkout max retry attempts from 5 to 2.
[2] chunk_id=c2 document_id=d1 score=0.8 answer_intents=2
Storage growth is calculated as daily_ingest_bytes * retention_days.

QUESTION: Which deployment changed the retry policy, how do I calculate storage growth, and who approved deploy-9921?
ANSWER INTENTS:
[1] Which deployment changed the retry policy?
[2] How do I calculate storage growth?
[3] Who approved deploy-9921?
</input>
<output>{"refused": false, "answer": "1. Deployment deploy-1832 reduced checkout max retry attempts from 5 to 2.\\n\\n2. Storage growth is calculated as daily_ingest_bytes * retention_days.\\n\\n3. Not supported by the provided context.", "cited_indices": [1, 2], "covered_questions": [1, 2], "unsupported_questions": [3]}</output>
</example>
</examples>
""".strip()
).replace("__REFUSAL_ANSWER__", REFUSAL_ANSWER)

_INTENT_TEXT_SYSTEM = (
    r"""
<responsibility>
You answer one RAG retrieval intent using only the context supplied in the user
message, which has already been filtered for this intent specifically.
</responsibility>

<rules>
1. Use ONLY the provided context.
2. The context has already been filtered for this intent.
3. If any context contains relevant evidence, answer directly.
4. If the question asks for a formula, calculation, or steps and context has
   formula-like or step-like evidence, include it.
5. If context contains OCR-damaged math, convert it to standard LaTeX math notation,
   but keep the same variables and operations from context.
6. If MATH EVIDENCE is provided, treat it as the authoritative normalized formula for
   this intent. Copy its formula exactly and do not rewrite it from OCR text.
7. When MATH EVIDENCE is provided, do not add any other formula, equation, or step
   list. Explain only the copied formula and its variables.
8. When MATH EVIDENCE is not provided, do not create LaTeX from OCR-damaged formula
   fragments yourself. State that the formula is not safely supported if the context
   is ambiguous.
9. Do not create formulas, variants, or named methods that are not stated in context.
10. Do not invent a step-by-step procedure. If context provides a formula but not
    explicit steps, explain the formula and variables only.
11. Do not use bullet or numbered steps unless the context explicitly contains
    procedural steps.
12. Do not mention authors, contribution notes, venue, affiliation, or training
    hardware unless the question asks for those details.
13. Keep the answer compact.
14. If no context supports the question, return exactly:
    __REFUSAL_ANSWER__
</rules>

<examples>
<example>
<input>
MATH EVIDENCE TO COPY EXACTLY IF RELEVANT:
storage\_growth = daily\_ingest\_bytes \times retention\_days

CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.9
Storage growth scales with daily ingest volume and how long data is retained.

QUESTION: How do I calculate storage growth?
</input>
<output>Storage growth is calculated as storage\_growth = daily\_ingest\_bytes \times retention\_days, where daily_ingest_bytes is the amount of data ingested per day and retention_days is how long that data is kept.</output>
</example>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.3
The company was founded in 2019.

QUESTION: What is the time complexity of the retrieval algorithm?
</input>
<output>__REFUSAL_ANSWER__</output>
</example>
</examples>
""".strip()
).replace("__REFUSAL_ANSWER__", REFUSAL_ANSWER)

_FINAL_TEXT_SYSTEM = (
    r"""
<responsibility>
You are a grounded RAG answer refinement agent. Polish a draft answer that was
assembled from separate per-intent answers into one coherent final answer, using only
the supplied context and draft - never new facts.
</responsibility>

<rules>
1. Use ONLY the provided context and draft answers.
2. Keep exactly the requested answer intents and answer each supported intent.
3. Format the final answer with numbered sections separated by a blank line.
4. If the context contains math, write it as LaTeX.
5. If a calculation intent is supported by context, include the formula and a short
   explanation of the variables.
6. Do not invent a step-by-step procedure. If the context provides a formula but not
   explicit steps, explain the formula and variables only.
7. Do not add unsupported facts or unsupported methods.
8. If none of the contexts support the question, return exactly:
   __REFUSAL_ANSWER__
</rules>

<examples>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.9
Deployment deploy-1832 reduced checkout max retry attempts from 5 to 2.
[2] chunk_id=c2 document_id=d1 score=0.8
Storage growth is calculated as daily_ingest_bytes * retention_days.

QUESTION: Which deployment changed the retry policy, and how do I calculate storage growth?
ANSWER INTENTS:
[1] Which deployment changed the retry policy?
[2] How do I calculate storage growth?

DRAFT ANSWER:
1. Deployment deploy-1832 reduced checkout max retry attempts from 5 to 2.
2. Storage growth is calculated as daily_ingest_bytes * retention_days.
</input>
<output>1. Deployment deploy-1832 reduced the checkout service's max retry attempts from 5 to 2.

2. Storage growth is calculated as storage\_growth = daily\_ingest\_bytes \times retention\_days, where daily_ingest_bytes is the volume ingested per day and retention_days is how long that data is kept.</output>
</example>
</examples>
""".strip()
).replace("__REFUSAL_ANSWER__", REFUSAL_ANSWER)

_MATH_EVIDENCE_SYSTEM = r"""
<responsibility>
You are a math OCR normalization agent for a RAG pipeline. Normalize formulas that are
visibly present in the supplied context into clean LaTeX - never invent or recall a
formula from outside the context.
</responsibility>

<rules>
1. Return formulas only. Do not summarize prose.
2. Extract only formulas relevant to the question.
3. Convert OCR-damaged math into standard LaTeX display math.
4. Preserve the variables and operations visible in context.
5. Use surrounding text only to disambiguate OCR layout, such as superscripts,
   subscripts, fractions, roots, transposes, parentheses, matrix products,
   concatenation, and projection matrices.
6. If the question does not ask for a formula, calculation, equation, or computation,
   return NO_MATH_EVIDENCE unless the context explicitly states a named formula as the
   direct answer.
7. Do not use memorized domain formulas. Normalize only formulas that are visibly
   present in the supplied context.
8. Do not invent missing variables, operators, denominators, exponents, or output
   multipliers.
9. Do not change multiplication into division, division into multiplication, or
   concatenation into summation unless the context explicitly shows that operation.
10. If an OCR fragment is too ambiguous to normalize safely, return the safest partial
    LaTeX plus a short ambiguity note instead of guessing.
11. If no relevant formula appears, return exactly: NO_MATH_EVIDENCE
12. Return only the normalized formula and a short variable note.
</rules>

<examples>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.9
storage growth (S) equals daily ingest bytes (D) times retention days (R)

QUESTION: How do I calculate storage growth?
</input>
<output>S = D \times R, where D is daily ingest bytes and R is retention days.</output>
</example>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.4
The company was founded in 2019 and is headquartered in Seattle.

QUESTION: Where is the company headquartered?
</input>
<output>NO_MATH_EVIDENCE</output>
</example>
</examples>
""".strip()

_MATH_EVIDENCE_REVIEW_SYSTEM = r"""
<responsibility>
You are a math evidence verification agent for a RAG pipeline. Verify a candidate
normalized formula against the supplied context before it is shown to a user, and
correct or reject it when the context does not support it as given.
</responsibility>

<rules>
1. Compare the candidate math evidence against the supplied context.
2. Set has_math_evidence=false and formula_latex="" if the candidate is prose, a
   definition sentence, or only wraps prose in LaTeX text commands.
3. Set has_math_evidence=false and formula_latex="" if the context does not visibly
   contain a relevant formula, equation, or calculation.
4. Correct OCR layout mistakes only when the source context supports the correction
   through visible symbols, line breaks, or nearby formula prose.
5. Use mathematical notation conventions for OCR layout: stacked text may indicate
   fractions, letters offset after a symbol may indicate superscripts or subscripts,
   and text after a closed parenthesized expression may indicate multiplication.
6. Preserve visible trailing factors. If context shows a variable or symbol
   immediately after a closed parenthesized expression and it is not visibly in a
   denominator, keep it as multiplication in formula_latex.
7. Do not use memorized domain formulas. Do not add variables or operations absent
   from context.
8. Do not change multiplication into division, division into multiplication, or
   concatenation into summation unless the context explicitly shows that operation.
9. Valid formula evidence must contain an equation or mathematical expression, not
   just explanatory text.
10. If the candidate contains one or more equations and the same function names,
    variables, and operations are visible in context, set formula_latex to the
    candidate unchanged.
11. If the candidate is wrong but safely correctable from context, set formula_latex
    to only the corrected LaTeX formula and a short variable note.
12. If ambiguity remains, use the safest partial LaTeX plus a short ambiguity note
    instead of guessing.
</rules>

<examples>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.9
storage growth (S) equals daily ingest bytes (D) times retention days (R)

QUESTION: How do I calculate storage growth?

CANDIDATE MATH EVIDENCE:
S = D \times R, where D is daily ingest bytes and R is retention days.
</input>
<output>{"has_math_evidence": true, "formula_latex": "S = D \times R, where D is daily ingest bytes and R is retention days."}</output>
</example>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.9
storage growth (S) equals daily ingest bytes (D) times retention days (R)

QUESTION: How do I calculate storage growth?

CANDIDATE MATH EVIDENCE:
S = D / R, where D is daily ingest bytes and R is retention days.
</input>
<output>{"has_math_evidence": true, "formula_latex": "S = D \times R, where D is daily ingest bytes and R is retention days."}</output>
</example>
<example>
<input>
CONTEXT:
[1] chunk_id=c1 document_id=d1 score=0.3
The company was founded in 2019.

QUESTION: When was the company founded?

CANDIDATE MATH EVIDENCE:
2019
</input>
<output>{"has_math_evidence": false, "formula_latex": ""}</output>
</example>
</examples>
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
            f"MATH EVIDENCE TO COPY EXACTLY IF RELEVANT:\n{math_evidence}\n\n"
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
