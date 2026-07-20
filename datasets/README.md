# Evaluation datasets

Corpus and golden question set for `scripts/evaluation/`. All of it comes from one source:
[`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) on
Hugging Face (license `cc-by-2.5`), itself derived from the official BioASQ Task 11b training set.
Nothing here is self-generated: the questions, reference answers, and relevance judgments are all
taken verbatim from that dataset.

## Layout

| Path | Committed? | Contents |
|------|------------|----------|
| `bioasq/raw/` | no (gitignored) | The two parquet files fetched by `uv run poe eval-download-corpus` |
| `golden/bioasq-v1.jsonl` | no (gitignored, ~6MB) | The golden question set, built from the QA split by `uv run poe eval-build-golden` |
| `golden/bioasq-v1.meta.json` | yes | Content hash, provenance, and known-gap notes for the last `bioasq-v1.jsonl` build |

`bioasq-v1.jsonl` is not committed: at 4,719 cases it's ~6MB, over this repo's 500KB large-file
hook, and - like the corpus parquet - it's 100% mechanically derived from an external source with
nothing hand-curated to lose. `eval-download-corpus && eval-build-golden` reproduces it byte-for-byte
(same deterministic sentence-splitting and split-hashing every run); `bioasq-v1.meta.json` documents
the exact shape (content hash, counts) that build should produce.

## Corpus

`rag-mini-bioasq` ships two configs:

- `question-answer-passages` (4,719 rows): `question`, `answer`, `relevant_passage_ids`, `id`.
- `text-corpus` (40,221 rows): `passage`, `id` - each row is one PubMed abstract, already atomic
  (a paragraph, not a multi-page document needing chunking). `id` is a real PubMed article ID, and
  it's the same ID space referenced by `relevant_passage_ids` above.

Fetch both with:

```bash
uv run poe eval-download-corpus
```

The parquet files are not committed (~26MB combined, gitignored) - `download_bioasq.py` plus the
dataset's own HF URL is the reproducible unit, not the bytes.

## Indexing the corpus into a local stack

```bash
make docker-up
uv run poe eval-download-corpus
uv run poe eval-index-corpus
```

Each passage becomes one small markdown document (`bioasq-passage-<id>.md`), uploaded and ingested
through the app's own composition root (`build_container()`) and the same `IngestDocumentUseCase`
the production worker runs, but in-process rather than through Kafka - there is no HTTP endpoint to
poll for "is this document indexed yet", so priming a local evaluation stack calls the use case
directly instead of waiting on the async pipeline. The eval *runner* (`scripts/evaluation/run.py`)
does not take this shortcut: it always talks to the running app over HTTP, so a measured run
reflects real serialization, middleware, and routing.

**Corpus size is configurable, on purpose.** Best practice for a fair retrieval evaluation is to
index the *full* candidate pool, not a hand-picked subset where the relevant passages are
artificially easy to find - shrinking the corpus shrinks the number of distractors and inflates
recall/precision numbers in a way that doesn't transfer to a realistic setting. `index_corpus.py`
therefore defaults to indexing all 40,221 passages. That's a real, one-time cost (tens of thousands
of upload+ingest calls), so for fast local iteration two flags are available:

- `--corpus-limit N` - index only the first N passages by ID.
- `--corpus-fraction 0.1` - index a deterministic ~10% sample (stable across reruns, hashed by
  passage ID, not `random.sample`).

**A partial-corpus run is not comparable to a full-corpus run** - retrieval difficulty depends on
how many distractors are actually in the index. Use the partial flags for fast local development
only; report metrics from a full-corpus run for anything you'd cite as a real number. The indexer
is also resumable: successfully-ingested passage IDs are appended to
`datasets/bioasq/.index-progress.txt` (gitignored) as it goes, so an interrupted run can pick back
up with `uv run poe eval-index-corpus` instead of restarting from zero (pass `--no-resume` to force
a clean re-index).

**Known limitation on Windows:** ingesting real content through `unstructured` currently hangs or
crashes with a native access violation on this environment - confirmed for PDF content (SIGSEGV
inside `python-magic`, via `faulthandler`) and, separately, for a trivial in-memory markdown string
(indefinite hang, no crash, force-killed after ~10 minutes of flat memory / slowly climbing CPU).
This is not specific to any one file type or to this corpus - it is a pre-existing bug in the
ingestion pipeline's dependency chain on Windows, unrelated to the corpus source. Likely fix is
switching to `python-magic-bin` (bundles the `libmagic` DLLs for Windows) or running the indexer
inside the project's Docker stack instead of the native Windows interpreter; neither has been
applied yet. Until it is, `eval-index-corpus` cannot be verified end to end on this machine.

## Golden dataset

`golden/bioasq-v1.jsonl` holds all 4,719 rows of the `question-answer-passages` split, converted
1:1 by `scripts/evaluation/build_bioasq_golden.py` into this harness's `EvalCase` schema
(`scripts/evaluation/dataset.py`):

| BioASQ field | EvalCase field | Notes |
|---|---|---|
| `question` | `question` | verbatim |
| `answer` | `reference_answer` | verbatim |
| `answer` | `reference_claims` | sentence-split, mechanically (`. `/`!`/`?` boundaries) |
| `relevant_passage_ids` | `relevant_document_ids` | mapped to `bioasq-passage-<id>` to match the filenames `index_corpus.py` uploads |
| (none) | `relevance_grades` | uniform `1` for every relevant id - see the note below |
| (derived) | `difficulty` | `single-hop` if exactly one relevant passage, else `multi-hop` |
| (derived) | `split` | deterministic hash of the case ID, ~80% dev / ~20% test |

No question, answer, or relevance judgment was written or edited by this repo - every row of the
source split is represented exactly once. This is a real difference from the corpus this harness
used before (a hand-authored, self-fact-checked set of questions over downloaded arXiv papers):
that approach required an independent fact-check pass to be trustworthy at all, whereas this one is
already a published, citable benchmark.

**Two honest, disclosed gaps, not glossed over:**

- **No graded relevance.** BioASQ gives binary relevance judgments only (a passage either supports
  the answer or it isn't listed). `relevance_grades` is therefore a uniform `1` for every relevant
  id, which means `ndcg_at_k` on this dataset is a binary-relevance variant of nDCG, not the true
  0-3 graded signal it was designed for.
- **No unanswerable cases.** Every case has `answerable: true` because the source QA split has no
  unanswerable questions. Rather than fabricate fake "no relevant document" cases to fill that gap
  (which the corpus doesn't actually support), this dataset simply cannot exercise abstention
  behavior - that remains untested here.

`difficulty` is a mechanical proxy (relevant-passage count), not a hand-assessed reasoning-type
label: a question with many relevant passages may just have many equally-good supporting sources
for one factoid, not genuine multi-hop reasoning across distinct documents. Treat `multi-hop` here
as "needs more than one supporting passage", not as a claim about the retrieval strategy required.

Distribution (from `golden/bioasq-v1.meta.json`, generated at build time - regenerate with
`uv run poe eval-build-golden` if it ever drifts):

```bash
uv run python -c "import json; print(json.dumps(json.load(open('datasets/golden/bioasq-v1.meta.json'))['difficulty_distribution'], indent=2))"
```
