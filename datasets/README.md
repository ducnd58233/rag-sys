# Evaluation datasets

Corpus and golden question set for `scripts/evaluation/`. See
[`docs/rag-evaluation/SPEC.md`](../../docs/rag-evaluation/SPEC.md) (in the workspace root, one level above
this repo) for the full evaluation spec, and its "Layout decisions" section for why these files live here
instead of inside an `evaluation/` package.

## Layout

| Path | Committed? | Contents |
|------|------------|----------|
| `manifest.json` | yes | Corpus provenance: arXiv IDs, titles, filenames, source URLs |
| `pdf/` | no (gitignored) | PDFs fetched by `uv run poe eval-download-corpus` |
| `md/`, `docx/` | no (gitignored) | Reserved for fixtures in other formats; empty today |
| `golden/golden-v0.1.jsonl` | yes | The golden question set, v0.1 |
| `golden/golden-v0.1.meta.json` | yes | Version tag + content hash for `golden-v0.1.jsonl` |

## Corpus

Six public arXiv papers in the "Attention Is All You Need" lineage, chosen because they have real
citation and revision relationships (not because they are about this repo's own domain):

| arXiv ID | Title | Relationship |
|----------|-------|--------------|
| 1706.03762 | Attention Is All You Need | root: introduces the Transformer |
| 1810.04805 | BERT | uses the Transformer encoder |
| 1907.11692 | RoBERTa | revises BERT's pretraining recipe |
| 2005.14165 | GPT-3 (Language Models are Few-Shot Learners) | uses the Transformer decoder |
| 2005.11401 | Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | combines a Transformer generator with a dense retriever |
| 2106.09685 | LoRA | parameter-efficient fine-tuning applied to Transformer weight matrices |

These relationships are what the golden dataset's multi-hop and temporal/version-aware cases exercise
(e.g. "what did RoBERTa change relative to BERT" needs both papers; a case cannot be answered from either
one alone).

Fetch the corpus with:

```bash
uv run poe eval-download-corpus
```

The PDFs are not committed (see `docs/rag-evaluation/PLAN.md` ADR-EV-006): licensing for arXiv preprints
is not uniform, and binary PDFs bloat git history. `manifest.json` plus the download script is the
reproducible unit; the bytes are not.

## Indexing the corpus into a local stack

```bash
make docker-up
uv run poe eval-download-corpus
uv run poe eval-index-corpus
```

`eval-index-corpus` uploads and ingests each PDF directly through the app's own composition root
(`build_container()`), the same `IngestDocumentUseCase` the production worker runs, but in-process rather
than through Kafka - there is no HTTP endpoint to poll for "is this document indexed yet", so priming a
local evaluation stack calls the use case directly instead of waiting on the async pipeline. See
`docs/rag-evaluation/PLAN.md` for why this differs from how the eval *runner* talks to the app (over HTTP,
per ADR-EV-001).

## Golden dataset

`golden/golden-v0.1.jsonl` holds 28 cases: single-document factual, multi-hop, exact-identifier,
temporal/version-aware, unanswerable, and adversarial/ambiguous. They were drafted from the assistant's
existing knowledge of these well-known papers, **not yet cross-checked against the indexed corpus text by
a human**. Every case has `reviewed_by: null`. Per `docs/rag-evaluation/SPEC.md` FR-EVAL-1, this dataset
does not yet meet the acceptance bar for being treated as ground truth - it exists so the retrieval
harness has something to run against end to end. Growing it to 250 reviewed cases is tracked as T1-04 in
`docs/rag-evaluation/TASKS.md`.

Each case's `relevant_document_ids` references the source **filename** (e.g.
`1706.03762-attention-is-all-you-need.pdf`), not a numeric document ID - document IDs are assigned fresh
on every ingestion and are not stable across clones (`docs/rag-evaluation/PLAN.md` ADR-EV-005).

Distribution:

| Type | Count |
|------|-------|
| Single-document factual | 9 |
| Multi-document / multi-hop | 6 |
| Exact identifier | 4 |
| Temporal / version-aware | 3 |
| Unanswerable | 4 |
| Ambiguous / adversarial | 2 |

Split: 18 `dev` / 10 `test`.
