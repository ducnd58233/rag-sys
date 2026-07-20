# Evaluation datasets

Corpus and golden question set for `scripts/evaluation/`.

## Layout

| Path | Committed? | Contents |
|------|------------|----------|
| `manifest.json` | yes | Corpus provenance: arXiv IDs, titles, filenames, source URLs |
| `pdf/` | no (gitignored) | PDFs fetched by `uv run poe eval-download-corpus` |
| `md/`, `docx/` | no (gitignored) | Reserved for fixtures in other formats; empty today |
| `golden/golden-v0.2.jsonl` | yes | The golden question set, v0.2 (current default, used by `run.py`) |
| `golden/golden-v0.1.jsonl` | yes | Superseded first draft, kept for reference / smaller smoke runs |

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

`golden/golden-v0.2.jsonl` holds 58 cases. Every `reference_answer` and `reference_claims` value was
fact-checked against text extracted directly from the downloaded PDFs with `pypdf` (bypassing the app's
own `unstructured`-based ingestion pipeline, since that path is currently blocked - see the Windows
limitation above). This confirms the *facts are correct*; it does **not** confirm retrieval or generation
quality, because the corpus has not actually been indexed and queried end to end yet. `reviewed_by`
records this fact-check pass, not the independent human review FR-EVAL-1-style acceptance still requires.
Growing this further and running an actual human review pass are still open, tracked separately from this
task.

Each case's `relevant_document_ids` references the source **filename** (e.g.
`1706.03762-attention-is-all-you-need.pdf`), not a numeric document ID. Document IDs are assigned fresh on
every ingestion (a Snowflake ID generated at upload time) and are not stable across clones or reindexes,
so a committed dataset cannot reference them; the filename is already carried on every ingested chunk's
metadata and stays stable, so the runner matches on that instead.

### Strategy targeting

Every answerable case carries a `target-*` tag naming the retrieval strategy it is meant to exercise, so
`scripts/evaluation/report.py`'s per-strategy-combination breakdown can be read against intent, not just
against whatever the router happened to pick:

| Tag | Meant to exercise | Count |
|-----|--------------------|-------|
| `target-lexical` | exact term/number recall (favors keyword/BM25 matching) | 16 |
| `target-semantic` | paraphrased/conceptual questions with little literal term overlap (favors vector similarity) | 12 |
| `target-graph` | multi-hop reasoning across 2-3 documents via a real citation or architecture relationship | 17 |
| `target-hybrid` | needs both precise recall and cross-document or conceptual combination at once | 9 |

**Known coverage gap, called out rather than faked:** this dataset has **no** cases genuinely labeled for
the `structured` or `temporal` retrieval strategies, because this corpus cannot honestly exercise them:

- `structured` routes on `RoutingSettings.identifier_patterns` (default `[a-z]+-\d{3,}`, e.g. `deploy-1832`).
  arXiv IDs (`1706.03762`) and this corpus's filenames don't match that shape, so no natural question here
  would actually trigger structured routing - writing cases tagged `target-structured` would just be
  wrong.
- `temporal` filters by document validity windows (`as_of`, superseded versions). Each paper here is
  ingested as a single, never-superseded version, so there is no version history to filter over. The
  existing `difficulty: temporal` cases (BERT vs. RoBERTa chronology) test cross-document reasoning about
  which paper came first, not the system's actual `TemporalStrategy` code path - that would need a
  document deliberately re-ingested as multiple versions, which this corpus doesn't have.

Distribution:

| Type | Count |
|------|-------|
| Single-document factual | 25 |
| Multi-document / multi-hop | 19 |
| Exact identifier | 4 |
| Temporal / version-aware (chronology reasoning, not `TemporalStrategy`) | 3 |
| Unanswerable | 5 |
| Ambiguous / adversarial | 2 |

Split: 37 `dev` / 21 `test`.

`golden/golden-v0.1.jsonl` (28 cases, superseded) is kept for a smaller/faster smoke run; it is not
strategy-tagged.
