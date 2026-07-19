# Tasks: Complex Query Answering for RAG

## Task 1: Add Complex RAG Settings

**Status:** Complete

**Description:** Add bounded complex-RAG configuration to `ChatSettings` and `.env.example`. Fix the existing env naming mismatch by replacing or documenting `CHAT_DEFAULT_TOP_K` as `CHAT_TOP_K`.

**Acceptance criteria:**
- [x] `ChatSettings` exposes complex RAG toggles and caps with validation.
- [x] `.env.example` contains matching `CHAT_` env names.
- [x] Defaults keep simple behavior predictable and latency bounded.

**Verification:**
- [x] Compile check: `uv run python -m compileall src`
- [x] Settings smoke: instantiate `Settings()` and print `settings.chat.top_k`.
- [x] No `tests/` directory is added.

**Dependencies:** None

**Files likely touched:**
- `rag-sys/src/shared/configs/settings.py`
- `rag-sys/.env.example`

**Estimated scope:** S

**Delivery branch:** `feature/rag-complex-query-task-1-settings`

## Task 2: Add Query Analysis Component

**Status:** Complete

**Description:** Add a small query analyzer that returns whether the query is simple or decomposed, plus 2-5 self-contained sub-questions for complex queries. Use the existing chat model structured-output path. Fall back to single-hop if analysis fails.

**Acceptance criteria:**
- [x] A new structured schema represents `is_complex`, `sub_questions`, and a short `reason`.
- [x] Analyzer includes the original query in downstream retrieval planning but not as a generated sub-question duplicate.
- [x] Empty, duplicate, and over-limit sub-questions are filtered deterministically.
- [x] Failure in analysis does not fail `/generation/ask`; it uses single-hop.

**Verification:**
- [x] Compile check: `uv run python -m compileall src`
- [x] Import smoke for the new query analysis module.
- [x] Manual smoke can classify `How to calculate attention?` as simple and `What is transformers and how to calculate attention?` as complex.
- [x] No `tests/` directory is added.

**Dependencies:** Task 1

**Files likely touched:**
- `rag-sys/src/modules/generation/app/query_analysis.py`
- `rag-sys/src/modules/generation/app/llm_schema.py`
- `rag-sys/src/modules/generation/__init__.py`

**Estimated scope:** M

**Delivery branch:** `feature/rag-complex-query-task-2-query-analysis`

## Task 3: Add Context Merge and Coverage Scoring

**Status:** Complete

**Description:** Add a deterministic context merge component for fan-out retrieval results. It deduplicates by `chunk_id`, keeps the best score, tracks which retrieval queries found each chunk, and orders chunks by coverage plus score.

**Acceptance criteria:**
- [x] Duplicate chunks collapse into one context item.
- [x] Chunks matched by multiple retrieval queries get a deterministic boost.
- [x] Final context count is capped by `CHAT_COMPLEX_RAG_FINAL_TOP_K`.
- [x] Metadata records bounded internal fields only if needed for prompting or tracing.

**Verification:**
- [x] Compile check: `uv run python -m compileall src`
- [x] Import smoke for the context merge module.
- [x] Manual smoke with sample in-memory chunks shows stable ordering and dedupe.
- [x] No `tests/` directory is added.

**Dependencies:** Task 1

**Files likely touched:**
- `rag-sys/src/modules/generation/app/context_merge.py`
- `rag-sys/src/modules/generation/domain/models.py`

**Estimated scope:** M

**Delivery branch:** `feature/rag-complex-query-task-3-context-merge`

## Task 4: Wire Decomposed Retrieval Into Ask

**Status:** Complete

**Description:** Update `AnswerQuestionUseCase` to choose between single-hop and decomposed retrieval. For complex queries, retrieve original query plus sub-questions concurrently, merge contexts, then continue to answer synthesis.

**Acceptance criteria:**
- [x] Single-hop behavior is preserved when complex RAG is disabled or analysis returns simple.
- [x] Complex path retrieves from the original query and capped sub-questions.
- [x] Retrieval fan-out is bounded and uses existing `IContextRetriever`.
- [x] If no merged context is found, the existing refusal behavior remains.

**Verification:**
- [x] Compile check: `uv run python -m compileall src`
- [x] Import smoke for `AnswerQuestionUseCase`.
- [x] Manual use-case check with a simple query returns the same response shape.
- [x] Manual use-case check with the compound Transformer query triggers the decomposed path.
- [x] No `tests/` directory is added.

**Dependencies:** Tasks 1, 2, 3

**Files likely touched:**
- `rag-sys/src/modules/generation/app/use_cases/answer_question.py`
- `rag-sys/src/modules/generation/__init__.py`

**Estimated scope:** M

**Delivery branch:** `feature/rag-complex-query-task-4-decomposed-ask`

## Task 5: Update Multi-Intent Grounded Synthesis

**Status:** Complete

**Description:** Update the grounded prompt and structured answer schema so the model must cover each detected intent when context supports it. Unsupported intents should be stated as unsupported by the provided context rather than silently omitted.

**Acceptance criteria:**
- [x] Prompt includes the original question and detected sub-questions.
- [x] Structured output can represent cited indices and intent coverage without changing the public API.
- [x] Final answer for the Transformer sample covers architecture and attention calculation when both are supported.
- [x] Citation validation still rejects answers with no valid citations.

**Verification:**
- [x] Compile check: `uv run python -m compileall src`
- [x] Import smoke for prompt and schema modules.
- [x] Manual use-case check confirms answer includes the attention formula when retrieved context contains it.
- [x] No `tests/` directory is added.

**Dependencies:** Task 4

**Files likely touched:**
- `rag-sys/src/modules/generation/domain/prompt.py`
- `rag-sys/src/modules/generation/app/llm_schema.py`
- `rag-sys/src/modules/generation/app/use_cases/answer_question.py`

**Estimated scope:** M

**Delivery branch:** `feature/rag-complex-query-task-5-synthesis`

## Task 6: Add Metrics and Traces for Complex RAG

**Status:** Code complete; runtime trace check pending Task 7

**Description:** Add spans and metrics for query analysis, retrieval fan-out, context merge, and answer synthesis. Update dashboard panels to distinguish `single_hop` and `decomposed` traffic.

**Acceptance criteria:**
- [x] Spans include `generation.query_analysis`, `generation.retrieve.original`, `generation.retrieve.subquestion`, `generation.context_merge`, and `generation.answer_synthesis`.
- [x] Ask metrics include bounded `strategy` labels.
- [x] New metrics count sub-questions, retrieval queries, merged contexts, and strategy outcomes.
- [x] Dashboard can show latency and volume by strategy.

**Verification:**
- [x] Compile check: `uv run python -m compileall src`
- [x] Dashboard JSON parse check.
- [x] Compose config check: `docker compose -f deployments/observability/docker-compose.observability.yml -f deployments/observability/docker-compose.langfuse.yml --profile llm config -q`
- [ ] Manual trace check in Grafana/Tempo after one complex ask.
- [x] No `tests/` directory is added.

**Dependencies:** Task 4

**Files likely touched:**
- `rag-sys/src/modules/generation/app/use_cases/answer_question.py`
- `rag-sys/src/shared/observability/metrics.py`
- `rag-sys/src/shared/observability/setup.py`
- `rag-sys/deployments/observability/grafana/dashboards/70-llm-rag.json`
- `rag-sys/deployments/observability/grafana/dashboards/00-overview.json`
- `rag-sys/deployments/observability/prometheus/rules/recording.slo.yml`

**Estimated scope:** M

**Delivery branch:** `feature/rag-complex-query-task-6-observability`

## Task 7: Update Docs and Run No-Pytest Verification

**Description:** Document the complex RAG settings, expected behavior, manual verification steps, and known limits. Run the allowed verification commands and capture manual evidence for PR review.

**Acceptance criteria:**
- [ ] README or observability docs mention complex RAG settings and dashboard panels.
- [ ] Manual verification steps include the Transformer compound query.
- [ ] PR notes include whether the answer covers both requested intents.
- [ ] No new test files or `tests/` directory are created.

**Verification:**
- [ ] Compile check: `uv run python -m compileall src`
- [ ] Import smoke for changed modules.
- [ ] Dashboard JSON parse check.
- [ ] Compose config check.
- [ ] Diff check: `git diff --check`
- [ ] Manual `/generation/ask` check against seeded Transformer data.

**Dependencies:** Tasks 1-6

**Files likely touched:**
- `rag-sys/README.md`
- `rag-sys/deployments/observability/README.md`
- PR description or implementation notes

**Estimated scope:** S

**Delivery branch:** `feature/rag-complex-query-task-7-docs-verification`

## Checkpoints

## Checkpoint: After Tasks 1-3

- [x] Foundation compiles.
- [x] Settings instantiate correctly.
- [x] Query analyzer and context merge import cleanly.
- [x] No public API contract changed.

## Checkpoint: After Tasks 4-5

- [x] Simple ask path still works.
- [x] Complex ask path retrieves more than one query.
- [x] Transformer compound query returns both architecture and attention calculation when evidence exists.
- [x] Public response schema remains `query`, `answer`, `citations`, `refused`.

## Checkpoint: After Tasks 6-7

- [ ] Traces show every major complex RAG step.
- [ ] Dashboard distinguishes `single_hop` and `decomposed`.
- [ ] No-pytest verification commands pass.
- [ ] Manual evidence is ready for review.
