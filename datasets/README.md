# Evaluation datasets

Corpus and question set for `scripts/evaluation/`. All of it comes from one source:
[`rag-datasets/rag-mini-bioasq`](https://huggingface.co/datasets/rag-datasets/rag-mini-bioasq) on
Hugging Face (license `cc-by-2.5`), itself derived from the official BioASQ Task 11b training set.
Nothing here is self-generated: the questions, reference answers, and relevance judgments are all
taken verbatim from that dataset.

## Two commands

```bash
make docker-up
uv run poe eval-prepare   # download -> process -> ingest -> datasets/manifest.json
uv run poe eval-run       # score retrieval against the dev split, using that manifest
```

`eval-prepare` (`scripts/evaluation/prepare.py`) does everything needed before a run can score
anything: fetch the raw parquet, materialize it into ingestible files, ingest them through the
app's own use cases, and record what happened. `eval-run` (`scripts/evaluation/run.py`) only reads
that record and the running app's HTTP API - it does not touch the dataset source, the filesystem
layout, or the DB directly. Both use the `uv` `evaluation` dependency group (`pyarrow`, `aiohttp`,
`pyyaml`, `tqdm`) rather than the project's main dependencies, so a production install (API/worker
image) never pulls in tooling it doesn't need; `Makefile`'s `eval-prepare`/`eval-run`/`test` targets
already pass `--group evaluation` where it's needed.

`eval-run` defaults to `--split dev` (~80% of cases, chosen so headline numbers aren't tuned
directly against the same cases used to report them). Pass `--split test` for the held-out 20%, or
`--split all` to score every case the manifest's `id_map` can cover regardless of split - useful
when you just want to know "does retrieval work at all against what I ingested" rather than a
methodologically clean dev/test comparison.

## Layout

| Path | Committed? | Written by | Contents |
|------|------------|------------|----------|
| `bioasq/raw/*.parquet` | no (gitignored) | `eval-prepare`'s download step | the two rag-mini-bioasq configs, byte-for-byte as published |
| `bioasq/processed/*.md` | no (gitignored) | `eval-prepare`'s process step | one markdown file per indexed passage - the literal bytes uploaded through the document API |
| `bioasq/.index-progress.jsonl` | no (gitignored) | `eval-prepare`'s ingest step | append-only `{passage_id, document_id}` log, used to resume an interrupted ingest without losing the id mapping already captured |
| `manifest.json` | no (gitignored) | `eval-prepare`, at the end | one entry per dataset: corpus scope, provenance hashes, and `id_map` |

`manifest.json` is not portable across machines or database resets: `id_map` pins bioasq passage
ids to the actual `document_id`s a *specific* local DB assigned when `eval-prepare` ran. Re-run
`eval-prepare` after a fresh `make docker-up` rather than reusing someone else's manifest.

There is no separate "golden dataset" file. `eval-run` reads the raw QA parquet directly through
`scripts/evaluation/sources/bioasq.py`'s `load_qa_cases()`, which is a pure, deterministic function
of that parquet plus the manifest's `id_map` - materializing it to disk would only add a step
without adding independent value, since nothing about the mapping is hand-curated.

## Why a manifest and an `id_map`, not just filenames

Each ingested passage becomes its own document (`bioasq-passage-<passage_id>.md`), and the earlier
version of this harness matched retrieval results back to relevance judgments by comparing that
filename string against `metadata.filename` in the search response. That works, but it is an
indirect proof: `metadata.filename` is best-effort request metadata, not a guaranteed, indexed
identifier, and nothing enforces that it survives every retrieval path unchanged.

`eval-prepare` now captures the real `document_id` the DB assigns at ingest time (the same id
`RetrievedChunkResponse.document_id` always returns, on every result, from every strategy) and
records `passage_id -> document_id` in `manifest.json`'s `id_map`. `eval-run` maps each QA case's
`relevant_passage_ids` through that table before scoring, so recall/precision/MRR/nDCG compare
against the database's own identity for each document, not a filename convention that happens to
agree with it.

One consequence: if a QA case's relevant passages were never ingested (only possible after a
`--corpus-limit`/`--corpus-fraction` prepare run - see below), `load_qa_cases()` drops that case
rather than scoring it against an id that isn't in `id_map`. `eval-run` logs how many cases were
dropped this way. A full-corpus prepare (the default) drops none.

## Corpus size is configurable, on purpose

`rag-mini-bioasq`'s `text-corpus` is not "gold passages plus distractors": every one of its 40,221
passages is the `relevant_passage_ids` target of at least one QA case (the corpus and the union of
every case's relevant ids are the same 40,221 ids - there is no extra distractor pool). That means a
plain id-ordered or hashed slice of the corpus has no reason to land on the *combination* of
passages any one question needs - each is an independently real PubMed id, unrelated to corpus
position - so a naive `--corpus-limit 200` could leave `eval-run` able to score a single case out of
4,719, even though 200 passages were successfully ingested.

`eval-prepare` therefore defaults to ingesting the full corpus (all 40,221 passages, a real one-time
cost of tens of thousands of upload+ingest calls) and only reorders for partial runs:

- `--corpus-limit N` - ingest N passages.
- `--corpus-fraction 0.1` - ingest a deterministic ~10% sample.

Both flags greedily pack passages by QA case, smallest `relevant_passage_ids` set first (mostly
single-hop cases, needing just one specific passage each), so a case is only skipped once its full
relevant set no longer fits the remaining budget - maximizing fully-covered, scoreable cases per
passage spent. Any leftover budget once no more cases fit is filled with other passages in id order,
so `--corpus-limit N` still ingests close to N passages. On the real dataset this turns
`--corpus-limit 200` from ~1 scoreable case into ~208.

**This still is not a substitute for a full-corpus run.** A partial corpus means fewer other
questions' passages are around to compete as retrieval candidates, which is easier than the full
40,221-passage task - report metrics from a full-corpus run for anything you'd cite as a real
number, and use the partial flags for fast local development only. And a case whose relevant
passages didn't fit the budget can never be scored, no matter how the packing is prioritized: to
score against most or all 4,719 QA cases, there's no substitute for a full-corpus `eval-prepare`
(the default, no `--corpus-limit`/`--corpus-fraction`).

**A partial-corpus run is not comparable to a full-corpus run.** Use the partial flags for fast
local development only; report metrics from a full-corpus run for anything you'd cite as a real
number. `eval-prepare` is resumable by default: passage ids already recorded in
`.index-progress.jsonl` are skipped and reused (both their processed file and their `document_id`)
on the next invocation, so an interrupted run can pick back up where it left off. Pass `--no-resume`
to force a clean re-ingest - this creates brand-new documents for every passage rather than reusing
prior ones (the document API has no upsert-by-filename), so only do this against a fresh DB.

Ingestion runs through the same application use cases the HTTP API uses (`CreateDocumentUploadUrlUseCase`,
`CompleteDocumentUploadUseCase`, `IngestDocumentUseCase`), just in-process rather than over
HTTP + Kafka, since there's no endpoint to poll for "is this document indexed yet" and tens of
thousands of round trips through the async pipeline would make this step impractically slow. This
is safe even when the outbox relay and worker are also running against the same stack:
`CompleteDocumentUploadUseCase` still enqueues the normal ingestion outbox event, and
`IngestDocumentUseCase` treats an already-`INDEXED` document version as a no-op, so a second,
worker-driven ingestion of the same document is a harmless skip, not a duplicate.

**Known limitation on Windows:** ingesting real content through `unstructured` currently hangs or
crashes with a native access violation on this environment - confirmed for PDF content (SIGSEGV
inside `python-magic`, via `faulthandler`) and, separately, for a trivial in-memory markdown string
(indefinite hang, no crash, force-killed after ~10 minutes of flat memory / slowly climbing CPU).
This is not specific to any one file type or to this corpus - it is a pre-existing bug in the
ingestion pipeline's dependency chain on Windows. Likely fix is switching to `python-magic-bin`
(bundles the `libmagic` DLLs for Windows) or running `eval-prepare` inside the project's Docker
stack instead of the native Windows interpreter; neither has been applied yet. Until it is,
`eval-prepare`'s ingest step cannot be verified end to end on this machine.

Separately, graph extraction (one step inside ingestion) calls a small local chat model for
structured JSON output; a bad/incomplete response for one chunk is caught and logged, and that
chunk simply contributes no graph facts - it does not fail the document's ingestion.

## Question set

`scripts/evaluation/sources/bioasq.py`'s `load_qa_cases()` reads all 4,719 rows of the
`question-answer-passages` split and maps each one into `EvalCase` (`scripts/evaluation/dataset.py`):

| BioASQ field | EvalCase field | Notes |
|---|---|---|
| `question` | `question` | verbatim |
| `answer` | `reference_answer` | verbatim |
| `answer` | `reference_claims` | sentence-split, mechanically (`. `/`!`/`?` boundaries) |
| `relevant_passage_ids` | `relevant_document_ids` | mapped through `manifest.json`'s `id_map` to the real DB `document_id` for each passage |
| (none) | `relevance_grades` | uniform `1` for every relevant id - see the note below |
| (derived) | `difficulty` | `single-hop` if exactly one relevant passage, else `multi-hop` |
| (derived) | `split` | deterministic hash of the case id, ~80% dev / ~20% test |

No question, answer, or relevance judgment was written or edited by this repo - every row of the
source split is represented exactly once in a full-corpus run, since the mapping is a pure function
with no randomness.

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

Inspect the distribution locally (after `eval-prepare` has produced a manifest):

```bash
uv run --group evaluation python -c "
from collections import Counter
from pathlib import Path
from scripts.evaluation.manifest import load
from scripts.evaluation.sources import bioasq

manifest = load('bioasq')
cases = bioasq.load_qa_cases(Path(manifest.raw_dir), manifest.id_map)
print(Counter(c.difficulty for c in cases))
print(Counter(c.split for c in cases))
"
```

## Adding another dataset

`scripts/evaluation/sources/` is a small registry (`scripts/evaluation/sources/__init__.py`), not a
BioASQ-specific script. To add a dataset, write `scripts/evaluation/sources/<name>.py` exposing the
same module-level surface as `bioasq.py` - `NAME`, `MIME_TYPE`, `download()`, `corpus_path()`,
`qa_path()`, `processed_filename()`, `iter_corpus_passages()`, `load_qa_cases()`, `provenance()` -
and add one line to the `SOURCES` registry. `prepare.py` and `run.py` both take `--dataset <name>`
and never import a source module by name directly, so neither script changes.

## Cleaning up

```bash
uv run poe eval-run -- --cleanup
```

Deletes `datasets/<dataset>/processed/` and the `.index-progress.jsonl` log after writing results -
both are cheap to regenerate from `manifest.json` plus a re-run of `eval-prepare` (which will reuse
the existing `id_map` rather than re-ingesting). It does **not** delete `raw/`, `manifest.json`, or
anything in the database: there is no delete-document use case in this codebase yet, so the only way
to fully reset ingested eval data today is to reset the local stack itself (drop and recreate the
`make docker-up` volumes).
