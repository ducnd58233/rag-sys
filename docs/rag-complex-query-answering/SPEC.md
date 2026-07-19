# Spec: Complex Query Answering for RAG

## Assumptions

1. The current target is the `rag-sys` repository under this workspace.
2. The first implementation should keep the existing `/generation/ask` response contract: `query`, `answer`, `citations`, and `refused`.
3. We should improve complex questions without adding a heavyweight multi-agent runtime yet.
4. Pytest is out of scope for the current phase because tests were removed by request.

Correct these before implementation if any assumption is wrong.

## Objective

Improve `/generation/ask` for compound and multi-hop questions such as `What is transformers and how to calculate attention?`.

Today the system answers simple questions well, but complex questions can be under-answered because the current path performs one retrieval for the original query, then asks the LLM to produce one concise answer from a flat context list.

The expected behavior is:

- Detect when a query contains multiple answer intents.
- Retrieve evidence for each intent, not only the highest-scoring intent.
- Produce an answer that covers every supported intent.
- Preserve grounded citations and refuse only unsupported parts or unsupported whole queries.
- Record metrics and traces for each decomposition and retrieval step.

## Current Behavior

Observed code path:

- API endpoint: `rag-sys/src/modules/generation/api/router.py`
- Use case: `rag-sys/src/modules/generation/app/use_cases/answer_question.py`
- Prompt builder: `rag-sys/src/modules/generation/domain/prompt.py`
- Retrieval adapter: `rag-sys/src/modules/generation/infra/retrieval_adapter.py`
- Retrieval use case: `rag-sys/src/modules/retrieval/app/use_cases/retrieve.py`
- Fusion: `rag-sys/src/modules/retrieval/infra/fusion/reciprocal_rank.py`

The retrieval use case embeds the whole query once, runs lexical and dense search in parallel, fuses both ranked lists with reciprocal rank fusion, applies an optional fused-score gate, and returns `top_k` chunks. The generation use case then sends those chunks to the chat model once.

This is a 2-step RAG shape. LangChain describes 2-step RAG as predictable and fast, but less flexible than agentic or hybrid RAG for ambiguous, multi-source, or iterative quality-control workflows. Source: https://docs.langchain.com/oss/python/langchain/retrieval

## Research Findings

For multi-step QA, one-step retrieve-and-read can be insufficient because what should be retrieved may depend on what has already been derived. IRCoT addresses this by interleaving retrieval with reasoning steps and reports retrieval and downstream QA gains on multi-hop datasets. Source: https://arxiv.org/abs/2212.10509

FLARE frames active RAG as deciding when and what to retrieve during generation, rather than retrieving once at the beginning. It uses anticipated future content as retrieval queries for long-form knowledge-intensive generation. Source: https://arxiv.org/abs/2305.06983

Self-RAG is a stronger research direction that trains a model to retrieve, generate, and critique with reflection tokens. It is useful conceptually, but it is not a small code-only feature for this repo because it assumes model behavior/training not present here. Source: https://arxiv.org/abs/2310.11511

RAG-Fusion combines multiple generated queries with reciprocal rank fusion. It can improve coverage and comprehensiveness, but generated queries can drift off topic if relevance is weak. Source: https://arxiv.org/abs/2402.03367

LlamaIndex's Sub Question Query Engine follows a practical product pattern: break a complex query into sub-questions, execute those questions, then synthesize the final response. Source: https://developers.llamaindex.ai/python/examples/query_engine/sub_question_query_engine/

## Recommended Design

Implement "query decomposition RAG" first, then add iterative retrieval only after we can observe and evaluate failures.

Proposed pipeline:

1. Analyze the user query.
2. If simple, use the current single-hop path.
3. If compound or multi-hop, generate 2-5 self-contained sub-questions.
4. Always include the original query as one retrieval query.
5. Retrieve for each sub-question using the existing retriever.
6. Merge and deduplicate chunks by `chunk_id`.
7. Re-rank merged chunks using the original query and sub-question coverage.
8. Ask the LLM to answer all detected intents with citations.
9. If an intent lacks context, say that part is not supported instead of dropping it silently.

This adds a deterministic bounded workflow, not an open-ended agent loop.

## Why Not Multi-Agent First

A multi-agent design is not necessary for this failure mode. The issue is retrieval coverage and answer synthesis, not tool orchestration across independent specialists.

Use a single use-case pipeline with small internal components:

- `IQueryAnalyzer` for decomposition.
- `IContextRetriever` remains the retrieval boundary.
- `IContextMerger` for dedupe and scoring.
- `GroundedPromptBuilder` updated to require coverage across query intents.

Add multi-agent orchestration later only if the system needs multiple tools, planning, browsing, code execution, or long-running research tasks. LangChain describes agentic RAG as more flexible but with variable latency; that tradeoff is not justified for this bounded `/ask` problem yet. Source: https://docs.langchain.com/oss/python/langchain/retrieval

## Iterative RAG and Boosted Multi-Hop

Iterative RAG means the system can retrieve more than once. Each step uses the current evidence, intermediate answer, or missing information to form the next retrieval query.

Multi-hop means the answer requires multiple facts or sub-answers that may live in different chunks or documents.

"Boosted multi-hop" in this project should mean a bounded multi-hop implementation that boosts coverage and ranking for evidence connected to detected sub-questions:

- Decompose the original query.
- Retrieve per sub-question.
- Boost chunks that appear across multiple retrieval lists.
- Boost chunks whose metadata or document IDs connect to already selected evidence.
- Preserve RRF so dense and lexical search still combine cleanly.
- Cap iterations and sub-queries to control latency and cost.

## Configuration

Add generation settings:

- `CHAT_COMPLEX_RAG_ENABLED=true`
- `CHAT_QUERY_DECOMPOSITION_MAX_SUBQUESTIONS=4`
- `CHAT_COMPLEX_RAG_MAX_RETRIEVAL_QUERIES=5`
- `CHAT_COMPLEX_RAG_PER_QUERY_TOP_K=6`
- `CHAT_COMPLEX_RAG_FINAL_TOP_K=12`
- `CHAT_COMPLEX_RAG_MAX_ITERATIONS=2`

Fix or confirm existing env naming:

- Code reads `CHAT_TOP_K` through `ChatSettings.top_k`.
- `.env.example` currently has `CHAT_DEFAULT_TOP_K=8`, which does not map to `ChatSettings.top_k`.

## Project Structure

Suggested files:

- `rag-sys/src/modules/generation/app/query_analysis.py`
- `rag-sys/src/modules/generation/app/context_merge.py`
- `rag-sys/src/modules/generation/app/use_cases/answer_question.py`
- `rag-sys/src/modules/generation/app/llm_schema.py`
- `rag-sys/src/modules/generation/domain/prompt.py`
- `rag-sys/src/shared/configs/settings.py`
- `rag-sys/src/shared/observability/metrics.py`
- `rag-sys/src/shared/observability/setup.py`
- `rag-sys/.env.example`

## Code Style

Follow current repo style:

- Use dataclasses for application DTOs.
- Keep FastAPI route handlers thin.
- Keep domain/application code independent of FastAPI.
- Use protocols for ports.
- Use Pydantic schemas for structured LLM output.

Example pattern already present:

```python
class IContextRetriever(Protocol):
    async def retrieve(...) -> Sequence[ContextChunk]: ...
```

## Commands

Available repo commands:

- `make docker-up`
- `make docker-down`
- `make obs-up`
- `make obs-up-core`
- `uv run app`
- `uv run dev`
- `uv run worker`
- `uv run poe lint`
- `uv run poe migrate-up`

Verification for current no-pytest phase:

- `uv run python -m compileall src`
- `uv run python -c "<small import smoke>"`
- `docker compose -f deployments/observability/docker-compose.observability.yml -f deployments/observability/docker-compose.langfuse.yml --profile llm config -q`
- `git diff --check`

## Testing Strategy

Because tests are currently removed by request, implementation should use:

- Manual API checks with seeded Transformer document data.
- Structured output smoke checks for decomposition schema.
- Observability checks in Grafana/Tempo for per-step spans.
- Later, when tests return, add unit tests for decomposition parsing, context merge, and prompt coverage.

## Metrics and Traces

Add spans:

- `generation.ask`
- `generation.query_analysis`
- `generation.retrieve.original`
- `generation.retrieve.subquestion`
- `generation.context_merge`
- `generation.answer_synthesis`

Add bounded metric attributes:

- `strategy`: `single_hop` or `decomposed`
- `outcome`: current outcome enum
- `subquestion.count`
- `retrieval.query.count`
- `context.count`
- `citation.count`

Do not label metrics with raw query text, org ID, document ID, chunk ID, or request ID.

## Boundaries

Always:

- Preserve grounded citations.
- Keep latency bounded with max sub-questions and max iterations.
- Keep existing `/generation/ask` response compatible unless explicitly approved.
- Trace each major step so dashboard latency explains where time is spent.

Ask first:

- Add reranker dependency or model service.
- Change API response schema to expose sub-questions and partial answers.
- Add a full agent runtime.
- Add paid external model calls for query analysis.

Never:

- Log raw user query or full prompts to metrics labels.
- Generate unsupported claims without citations.
- Run unbounded retrieval loops.
- Commit secrets or real Langfuse keys.
- Reintroduce pytest/tests in the current phase without approval.

## Success Criteria

For the sample query `What is transformers and how to calculate attention?`, the answer should include both:

- A grounded explanation of Transformer architecture.
- A grounded explanation of scaled dot-product attention calculation, including the `softmax(QK^T / sqrt(d_k))V` formula when present in retrieved context.

The final citations should include at least one context supporting Transformer architecture and at least one context supporting attention calculation.

Operational criteria:

- Simple questions still use the single-hop path by default.
- Complex questions retrieve from at least two generated/self-contained retrieval queries when enabled.
- Total retrieval queries are capped.
- Ask latency, decomposition latency, retrieval latency per query, and LLM synthesis latency appear in traces.
- Dashboard can distinguish `single_hop` vs `decomposed` traffic.

## Open Questions

1. Do we want `/generation/ask` to return sub-questions/debug evidence in API responses, or keep that only in traces/logs?
2. Should complex RAG be enabled by default in local development or behind `CHAT_COMPLEX_RAG_ENABLED=false` first?
3. Are we allowed to add a reranker dependency/service, or should phase 1 only use RRF and simple coverage scoring?
4. Should unsupported sub-questions produce a partial answer, or should the whole request refuse?
