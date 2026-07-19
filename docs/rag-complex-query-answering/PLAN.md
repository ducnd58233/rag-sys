# Implementation Plan: Complex Query Answering for RAG

## Overview

Build a bounded complex-query path for `/generation/ask` in `rag-sys`. Simple queries keep the current single-hop behavior. Compound or multi-hop queries use query decomposition, bounded fan-out retrieval, context merge, and a stricter synthesis prompt so the final answer covers each supported intent with citations.

This plan does not add `tests/` and does not run `pytest`.

## Architecture Decisions

- Keep one deterministic pipeline instead of adding a multi-agent runtime. The current failure is retrieval coverage and synthesis coverage, not general tool planning.
- Keep the existing `/generation/ask` response contract: `query`, `answer`, `citations`, `refused`.
- Add internal components under `src/modules/generation/app/` for query analysis and context merge.
- Use the existing `IContextRetriever` boundary, so retrieval internals stay unchanged for phase 1.
- Keep complex RAG bounded by config: max sub-questions, max retrieval queries, per-query top-k, final top-k, and max iterations.
- Add observability for strategy and per-step timing without high-cardinality labels.
- Do not add a reranker dependency in phase 1. Use existing RRF output plus deterministic coverage scoring first.

## Dependency Graph

```text
ChatSettings config
    |
    +-- Query analysis schema and component
    |       |
    |       +-- AnswerQuestionUseCase strategy routing
    |
    +-- Context merge component
            |
            +-- Fan-out retrieval for decomposed queries
                    |
                    +-- Prompt synthesis for intent coverage
                            |
                            +-- Metrics, traces, dashboard updates
                                    |
                                    +-- Manual verification and docs
```

## Task List

### Phase 1: Foundation

- [ ] Task 1: Add complex RAG settings and fix chat top-k env naming
- [ ] Task 2: Add query decomposition schema and analyzer
- [ ] Task 3: Add context merge and coverage scoring

### Checkpoint: Foundation

- [ ] `uv run python -m compileall src` passes.
- [ ] Import smoke for generation modules passes.
- [ ] Simple ask code path still routes through current single-hop behavior when complex RAG is disabled.

### Phase 2: Ask Pipeline

- [ ] Task 4: Wire decomposed retrieval into `AnswerQuestionUseCase`
- [ ] Task 5: Update grounded prompt for multi-intent synthesis and partial unsupported coverage

### Checkpoint: Core Flow

- [ ] Manual `/generation/ask` check answers both parts of `What is transformers and how to calculate attention?` when relevant chunks exist.
- [ ] Answer cites chunks for both Transformer architecture and attention calculation.
- [ ] Simple query output does not regress in shape or refusal behavior.

### Phase 3: Observability and Docs

- [ ] Task 6: Add complex RAG metrics and traces
- [ ] Task 7: Update dashboard/docs and run no-pytest verification

### Checkpoint: Ready for Review

- [ ] `uv run python -m compileall src` passes.
- [ ] Import smoke passes.
- [ ] Dashboard JSON parses.
- [ ] Observability compose config validates.
- [ ] `git diff --check` passes.
- [ ] Manual ask result and trace evidence are recorded in PR notes.

## Vertical Slices

Task 1 leaves the app configurable with no behavior change. Task 2 adds query analysis behind a local interface. Task 3 adds deterministic merge behavior without changing the API. Task 4 creates the first end-to-end decomposed retrieval path. Task 5 improves answer quality. Task 6 makes the new path visible in metrics and traces. Task 7 packages verification and docs.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Query decomposition creates off-topic sub-questions | Medium | Keep original query, cap sub-questions, require self-contained sub-questions, and reject empty/duplicate items. |
| Latency grows too much | High | Cap retrieval queries and sub-questions, retrieve sub-questions concurrently, keep max iterations at 1 in phase 1 unless enabled. |
| Prompt still drops one intent | Medium | Pass explicit intent list and require coverage status per intent in structured output. |
| Context window grows too large | Medium | Merge and trim to `CHAT_COMPLEX_RAG_FINAL_TOP_K`; keep chunk text unchanged but bounded. |
| Metrics cardinality grows | Medium | Use bounded labels only: `strategy`, `outcome`, `query_kind`; no query text, IDs, chunk IDs, or request IDs. |
| No automated tests in this phase | Medium | Use compile/import smoke, JSON validation, compose validation, and manual API traces. |

## Parallelization Opportunities

- Tasks 1 and 2 are mostly sequential because query analysis depends on settings.
- Task 3 can be developed after Task 1 and does not need prompt changes.
- Task 6 can start after Task 4 defines span names and metric attributes.
- Task 7 is last because it depends on final dashboard and docs content.

## Open Questions

- Should complex RAG be enabled by default in `.env.example`, or shipped disabled first?
- Should unsupported sub-questions be shown in the final answer as "not supported by the provided context"?
- Should a later PR expose `sub_questions` and per-intent coverage in the API for debugging?
- Should phase 2 add a cross-encoder reranker service after deterministic coverage scoring is measured?
