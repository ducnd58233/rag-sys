# Evaluation datasets

Corpus and golden question set for `scripts/evaluation/`.

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

The PDFs are not committed: licensing for arXiv preprints is not uniform, and binary PDFs bloat git
history permanently. `manifest.json` plus the download script is the reproducible unit; the bytes are
not.

## Indexing the corpus into a local stack

```bash
make docker-up
uv run poe eval-download-corpus
uv run poe eval-index-corpus
```

`eval-index-corpus` uploads and ingests each PDF directly through the app's own composition root
(`build_container()`) and calls the same `IngestDocumentUseCase` the production worker runs, but
in-process rather than through Kafka - there is no HTTP endpoint to poll for "is this document indexed
yet", so priming a local evaluation stack calls the use case directly instead of waiting on the async
pipeline. The eval *runner* (`scripts/evaluation/run.py`) does not take this shortcut: it always talks to
the running app over HTTP, so a measured run reflects real serialization, middleware, and routing.

**Known limitation on Windows:** indexing a real PDF currently crashes with a native access violation
inside `unstructured`'s `python-magic` dependency (confirmed via `faulthandler`, not specific to this
corpus - it affects the ingestion pipeline generally, not just evaluation). Likely fix is switching to
`python-magic-bin`, which bundles the `libmagic` DLLs for Windows; not yet applied.

## Golden dataset

`golden/golden-v0.1.jsonl` holds 28 cases: single-document factual, multi-hop, exact-identifier,
temporal/version-aware, unanswerable, and adversarial/ambiguous. They were drafted from the assistant's
existing knowledge of these well-known papers, **not yet cross-checked against the indexed corpus text by
a human, and not yet validated against real retrieval output** (see the Windows limitation above - the
corpus has not actually been indexed end to end yet). Every case has `reviewed_by: null`. Treat this
dataset as a structural placeholder that exercises the harness, not as ground truth, until both a human
review pass and a real indexed run have happened.

Each case's `relevant_document_ids` references the source **filename** (e.g.
`1706.03762-attention-is-all-you-need.pdf`), not a numeric document ID. Document IDs are assigned fresh on
every ingestion (a Snowflake ID generated at upload time) and are not stable across clones or reindexes,
so a committed dataset cannot reference them; the filename is already carried on every ingested chunk's
metadata and stays stable, so the runner matches on that instead.

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
