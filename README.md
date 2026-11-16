# RegLens

**An AI and data-protection compliance assistant built with Retrieval-Augmented Generation (RAG).**

RegLens is designed to answer questions such as *"Is our resume-screening feature a high-risk AI
system?"* or *"How quickly must we report a personal-data breach?"* from real regulatory
documents, and to show the exact passage behind every answer.

It is built from scratch, without LangChain or LlamaIndex, for the *Computing and Society*
assignment "Building and Demystifying RAG Systems". Every stage (ingestion, chunking,
embedding, vector storage, retrieval, augmentation, generation) is plain, readable Python.

> RegLens is a student project. It is not legal advice.

**Author:** Judy Abdeen ([@judyabdeen](https://github.com/judyabdeen)) · Course: *Computing and Society*
(Dr. Sherif M. Tawfik).

**Corpus:** 34 documents from official publishers in the EU, US, UK and international bodies.
The full list, with versions, legal status and verification notes, is in
[`data/sources.csv`](data/sources.csv).

## Scenario

A mid-size software company ships AI features to customers in the EU and the US. Its legal,
compliance and product teams need fast, citable answers about the EU AI Act, GDPR, EDPB and
Commission guidance, NIST frameworks, HIPAA, the CCPA and international principles.

One built-in demonstration: the EU AI Act's high-risk obligations were originally due on
2 August 2026, but Regulation (EU) 2026/1744 moved them (Annex III systems to 2 December 2027,
Annex I systems to 2 August 2028). A model trained earlier tends to give the old date. RegLens
indexes both the original Act and the amending regulation and shows which source says what.

## Status

All stages are built, tested and evaluated.

|Step|What|State|
|-|-|-|
|1|Corpus manifest, downloader, setup checker|done|
|2|Ingestion, chunking, embeddings, vector store, hybrid retrieval, generation|done; 34 of 34 documents, 3,813 chunks|
|3|Web interface (`python scripts/serve.py`)|done|
|4|Evaluation on 27 gold questions|done; results in `eval/results/latest/`|
|5|Technical report, slides, video|done; see [`docs/`](docs/)|
|6|GitHub release|done|

## Results in brief

Measured on 23 answerable and 4 out-of-scope questions, one run, with the language model Gemini
(full tables and the honest caveats are in the [report](docs/RegLens_report.pdf)):

* **Retrieval.** Combining keyword and meaning search finds the right document in the top 8 for 78 % of
questions, against 70 % for either search alone. Re-ranking orders results better but costs about 2 seconds.
* **Answers.** RegLens answered 21 of 23 questions correctly and the model alone 18 of 23. With 23 questions
this three-question gap is a hint, not proof. The clear difference is the three questions about the AI Act
amendment, which is newer than the model's training: the model alone got 0 of 3, RegLens 3 of 3.
* **Where it fails.** Questions that need two documents at once are not retrieved well (0 % on that measure), and
the model alone beat RegLens on two questions.
* **Cost.** Generation dominates time and cost (about 13 s and 3,460 input tokens per answer).

## Documentation

|File|What|
|-|-|
|[`docs/report.pdf`](docs/RegLens_report.pdf)|Technical report (8 pages) following the assignment's sections|
|[`docs/slides.pdf`](docs/RegLens_slides.pdf)|Presentation slides|

## Quick start

Requires Python 3.10 or newer (3.11 or 3.12 is the safest choice) and Git.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate            # macOS / Linux
.venv\\\\\\\\Scripts\\\\\\\\activate               # Windows (PowerShell: .venv\\\\\\\\Scripts\\\\\\\\Activate.ps1)

# 2. Install dependencies (the first install downloads about 1-2 GB: PyTorch is part of sentence-transformers)
pip install -r requirements-dev.txt

# 3. Configure your API key (the .env file is git-ignored)
cp .env.example .env                 # Windows: copy .env.example .env
#    then open .env and paste your free Gemini key from https://aistudio.google.com/apikey

# 4. Check the setup, then fetch the corpus
python scripts/check\\\\\\\_setup.py
python scripts/download\\\\\\\_corpus.py
python scripts/download\\\\\\\_corpus.py --status     # what is present, missing or invalid (no network)

# 5. Read the documents, clean them and cut them into chunks  (writes data/processed/, git-ignored)
python scripts/ingest.py
python scripts/ingest.py --show eu-gdpr        # headings and first chunks of one document
python scripts/inspect\\\\\\\_pdf.py eu-gdpr --pages 12   # font sizes and tags of every line of a page (for debugging)

# 6. Check the machine-learning stack: a real ChromaDB round trip, the embedding and re-ranking models
#    (downloaded once, about 220 MB in total) and one short call to Gemini
python scripts/check\\\\\\\_stack.py

# 7. Embed the chunks and fill the vector database (a few minutes on a laptop CPU)
python scripts/build\\\\\\\_index.py

# 8. Ask questions
python scripts/ask.py "How quickly must a personal data breach be reported?"
python scripts/ask.py                          # interactive chat; follow-up questions work
python scripts/ask.py "When do the AI Act's high-risk obligations apply?" --both   # with and without the documents
python scripts/ask.py --documents              # the indexed documents and their ids
python scripts/ask.py "Is a CV-screening tool high-risk?" --filter jurisdiction=EU --rerank --full

# 9. Or use the web interface (opens http://localhost:8000; stop it with Ctrl+C)
python scripts/serve.py
```

**Keep the project out of OneDrive, Dropbox and similar folders.** Sync clients lock files while they upload
them, which breaks database files, package installs and git. RegLens notices (it moves the index to
`%LOCALAPPDATA%\\\\\\\\RegLens` and says so), but a plain folder such as `C:\\\\\\\\dev\\\\\\\\reglens` is the safe choice.

Three EU legal texts (GDPR, the AI Act and its amending regulation) must be saved by hand
because EUR-Lex blocks scripts. [`data/MANUAL\\\\\\\_DOWNLOADS.md`](data/MANUAL_DOWNLOADS.md) explains how.

Run the tests with `pytest`.

## How the corpus is handled

* The repository does **not** redistribute the documents. It stores a manifest of official
publisher URLs, and the downloader rebuilds the corpus on your machine. Check each
publisher's terms before reusing a document.
* Every download is validated (real PDF, extractable text, not an error or bot-challenge page)
before it is accepted, and a provenance log (`data/download\\\\\\\_report.json`, git-ignored)
records the source URL, time and SHA-256 hash of each file.
* Each document carries a legal status in the manifest (`in\\\\\\\_force`, `final`, `draft`,
`consultation`, `proposed`). Draft and proposed texts are indexed on purpose and must be
labelled as such in answers.

## How the documents are read (ingestion)

`scripts/ingest.py` turns every downloaded file into clean, structured text and then into chunks.

1. **Extract** the text of each PDF page with `pypdf` together with how every line is typeset (font size,
bold) and the bookmarks, if any; raised digits are written out explicitly, so a footnote mark never
fuses with a number (`WP 218` + `6`) and an exponent survives (`10^25`, even when the PDF draws it
digit by digit). pypdf has quirks that would otherwise lose that information - it hands over a whole run
of text as one piece however often the type changes inside it, and it reports text drawn by a form
(headers, footers) twice - so the reader works around them and uses the result only when the pieces add
up to exactly the text pypdf extracts anyway. Web pages are parsed with BeautifulSoup, keeping the main
content and dropping menus, banners and footers.
2. **Clean** the page text: running headers and footers and page numbers are removed, private-use
bullets and ligatures are normalised, words broken over two lines are re-joined, and spacing
artefacts such as `S tate` or `built -in` are repaired. Nothing else is deleted.
3. **Find the structure:** footnotes are separated from the body - a footnote is a block of small print at
the foot of the page that starts with a note number, because text patterns alone cannot tell it from a
numbered list - and headings are recognised from legal patterns (`Article 5`, `CHAPTER III`,
`Annex I`, `§ 7001`), numbering (`2.1`), ALL-CAPS and Title Case lines (these two only when the line is
larger or bolder than the running text, so table cells are not mistaken for headings), or from the PDF
bookmarks when those are detailed enough. "Running text" is the commonest font size - or a larger one
that carries at least 30 % of the text, because a short statement with long footnotes would otherwise
be mistaken for its own footnotes. A heading (numbered or not) that wraps over several lines in the same
bold or larger type is one heading; a numbered heading may also be followed directly by a list item
(`1.4. Robustness, security and safety` / `a) AI systems should ...`), and a line that holds only the broken
half of a link or tag (`\\\\\\\[NIS`) is never a heading. Three more rules matter for EU acts: a labelled line that
merely starts a wrapped sentence (`Annex I. In certain cases ...`) is no heading unless it begins a block or is
set apart in type; the formula `HAVE ADOPTED THIS REGULATION:` ends the recitals and opens the `Enacting terms`
(recitals are not binding, the enacting terms are, and an answer should not confuse them); and articles that an
amending act quotes to insert them into another act (`‘Article 75a ...’`) stay text under the amending
article, so a passage is never filed under an article number that belongs to a different act.
4. **Chunk** by that structure ("semantic" in the sense of meaning units: an article, a section):
a section becomes one chunk if it fits in 320 words, small neighbours are merged, long sections are
cut at sentence ends with a 40-word overlap, and every chunk remembers its heading path and pages.
The text that is embedded starts with `document title | chapter > article`, so a bare passage still
says where it comes from. Footnotes become separate chunks, gathered only from neighbouring pages so that
the page range of a chunk stays true. A fixed-size baseline (`--strategy fixed`) exists for the evaluation.
5. **Audit:** every document gets warnings (`NO\\\\\\\_HEADINGS`, `MANY\\\\\\\_HEADINGS`, `EMPTY\\\\\\\_PAGES` with the page
numbers, `GARBLED\\\\\\\_CHARS` with the glyph codes, `RUN\\\\\\\_TOGETHER\\\\\\\_WORDS`, `TITLE\\\\\\\_MISMATCH`, ...) so that a
badly extracted document is noticed, not silently indexed.

Outputs go to `data/processed/`: `blocks/<id>.json` (cleaned documents, cached by file hash),
`chunks.jsonl` (what the embedding step reads) and `ingest\\\\\\\_report.json`. `python scripts/ingest.py --export` also packs the cleaned text into one small zip for review.

## How a question is answered (retrieval and generation)

`scripts/build\\\\\\\_index.py` embeds every chunk with a local model (`BAAI/bge-small-en-v1.5`, 384 numbers per chunk) and
stores vector, text and metadata (document, legal status, jurisdiction, pages, link) in ChromaDB. It also writes a small
`\\\\\\\*.info.json` file that records how the index was built; an index that was interrupted, damaged or built with another
embedding model is refused with a message that says what to do. `scripts/ask.py` then answers a question in five stages:

1. **Prepare the question.** A follow-up ("and in the US?") is rewritten into a stand-alone question by the language
model. A hand-written glossary of 41 legal-vocabulary rules (for example "breach" -> "personal data breach
notification supervisory authority") is added to the *keyword* query only. It adds terms, never deadlines or
amounts. Optional: paraphrases of the question and a hypothetical answer (HyDE).
2. **Search twice, then fuse.** A meaning search (cosine similarity of embeddings, ChromaDB or the exact NumPy store)
and a keyword search (BM25, written for this project) each return 40 candidates; reciprocal rank fusion merges
the two lists. Optional: a cross-encoder re-scores the best 20.
3. **Assemble the context.** The best passages are numbered `\\\\\\\[S1]`, `\\\\\\\[S2]`, ... within a word budget. Each one carries
its document, jurisdiction, **legal status** (a draft, consultation or proposal is marked "not adopted"), version,
section and pages, so the model can tell the original AI Act from the regulation that amends it.
4. **Generate.** A strict prompt tells the model to answer only from the numbered sources, cite them, quote figures
exactly, keep EU and US law apart, treat the sources as evidence and not as instructions, and say
`NOT\\\\\\\_IN\\\\\\\_SOURCES:` when they do not answer. The call is plain REST (no vendor SDK), retried with back-off and
moved to the next fallback model when a quota is used up.
5. **Check.** Every `\\\\\\\[S#]` must exist; every figure in the answer must appear in a source; sentences without a citation
are counted. If the model is unavailable the passages are shown instead of an error.

Privacy: the embedding and re-ranking models run on your computer. The question and the retrieved passages (public
documents) are sent to the language-model provider; on Gemini's free tier Google may use prompts to improve its
products, so never type confidential information into RegLens.

## The web interface

```bash
python scripts/serve.py                         # opens http://localhost:8000 in your browser
python scripts/serve.py --port 8001 --no-browser
```

The page works like an annotated law book: the answer is on the left and, in the margin beside each statement, stand the
passages it rests on, with the document, section, page and a **legal-status stamp** (*In force*, *Final guidance*, *Not
adopted*). The status is written in words and drawn as a shape, so it does not depend on colour. Clicking a citation
`\\\\\\\[S2]` moves its note beside that statement; "Read the full passage" shows the whole passage with its neighbours.

* **Four modes.** *Answer* (documents and model, with citations); *Compare* (the same question with and without the
documents, side by side, with a table of every figure and where it appears); *Passages* (only the matching passages, the
model is not asked); *Model alone* (what a plain chatbot says).
* **Checks under every answer.** A citation that does not exist, a figure that appears in no passage and a statement
without a citation are reported, and the sentence of each cited passage that is closest to the statement is highlighted.
*How this answer was made* lists the settings, the timings and the score of every passage.
* **Scope and settings** (the button next to the question box). Limit the search by jurisdiction, legal status, kind of
document or individual documents; choose keyword, meaning or combined search, how many passages the model sees,
re-ranking, glossary, follow-up rewriting, rephrased searches and hypothetical-answer search. The choices are remembered in
the browser; question texts are not.
* **Everyday use.** Follow-up questions ("and in the US?") work. `Enter` asks, `Shift+Enter` starts a new line, `/` jumps to
the question box. A running question can be stopped. *Save as Markdown* exports the conversation with its sources. Light
and dark themes follow the system; the page also works on a phone and with the keyboard alone.
* **When something is missing the page says what, and how to fix it:** no index yet (the page opens by itself when
`build\\\\\\\_index.py` finishes), no API key (passages are shown), a used-up quota or a provider outage (the passages are shown
with the reason), a stopped server.

**Security.** The server runs on your computer next to your API key, so it listens on `127.0.0.1` only, refuses requests whose
`Host` is not a local name (DNS rebinding) and writes that come from another web site (`Origin` check), caps request bodies at
64 KiB, and sends a strict Content-Security-Policy: no inline scripts or styles and nothing loaded from other sites (there are
no external fonts, scripts or trackers). Answers and passages come from a model and from documents, so the page always inserts
them as text, never as markup. The question is not written to the server log, and error messages never contain the key.

The routes, for anyone who wants to build another front end: `GET /api/status`, `GET /api/documents`,
`GET /api/passages/{id}` and `POST /api/ask` (JSON: `question`, `mode`, `history`, `options`, `filters`). Errors always look like
`{"error": {"kind": ..., "message": ..., "fix": ...}}`.

**Tests.** `pytest` runs everything. The page's logic (`tests/js`) is tested with Node.js when it is installed, and
`tests/test\\\\\\\_web\\\\\\\_browser.py` drives the real page in Chromium against a pretend model; both are skipped when Node or
Playwright is missing (`pip install playwright \\\\\\\&\\\\\\\& playwright install chromium`). `tests/test\\\\\\\_web\\\\\\\_static.py` checks, without a
browser, that the page obeys its own security policy and agrees with the server about modes, search methods and filters.

## Evaluation

`python scripts/evaluate.py` measures the system on 27 gold questions (`eval/gold.jsonl`): 23 the documents can answer
(numbers, rules, definitions, a question whose answer changed with the AI Act amendment, comparisons across documents) and
4 they cannot (the right behaviour is to say so). Nothing is estimated: every gold question names the documents that hold
the answer, a regular expression for the passage that contains it, and the facts a correct answer must state, and
`--validate` proves that all of them exist in the real corpus before anything is measured.

|What|Command|Needs the language model?|
|-|-|-|
|Check the gold set against the corpus|`python scripts/evaluate.py --validate`|no|
|Retrieval: keywords vs meaning vs hybrid, re-ranking, glossary, footnotes|`python scripts/evaluate.py`|no|
|Also the neural embedding model against the hashing baseline|`--embedding-baseline`|no|
|Also structure-aware against fixed-size chunks|`--fixed-chunks data/processed\\\\\\\_fixed/chunks.jsonl` (made by `python scripts/ingest.py --strategy fixed --out-dir data/processed\\\\\\\_fixed`)|no|
|Also Chroma against the exact NumPy search|`--stores` (after `python scripts/build\\\\\\\_index.py --backend both`)|no|
|Answers: the model alone against RegLens (correct, grounded, cited, abstains, time, tokens)|`--generate`|yes (about 50 requests)|
|Also a 1-5 judge for relevance, faithfulness and correctness|`--generate --judge`|yes|
|Score the saved answers again after a gold pattern was corrected (no model, no index; the report lists every verdict that changed)|`--rescore`|no|
|Also answers with re-ranking and multi-query, and multi-query and HyDE retrieval|`--generate --extra`|yes|

Results go to `eval/results/latest/` (`report.md`, `results.json`, `generation.jsonl`). Answers are saved as they
arrive, so a run that hits the free quota can be continued by running the same command again. Use `--min-interval 8` to
go slower and `--limit 5` for a quick trial. The metrics are defined in `src/reglens/evaluation/metrics.py`.

## Project layout

```
data/
  sources.csv            manifest: one row per document (URL, version, status, verification)
  MANUAL\\\\\\\_DOWNLOADS.md    how to save the files that scripts cannot fetch
  raw/                   downloaded documents (git-ignored)
  processed/             cleaned text, chunks and audit report (git-ignored)
eval/
  gold.jsonl             the gold questions: documents, evidence and facts for each
  results/               the numbers of the runs that are reported (committed)
docs/                    technical report, slides and video script (PDF)
scripts/
  check\\\\\\\_setup.py         environment and API-key check (never prints the key)
  download\\\\\\\_corpus.py     downloads and validates the corpus
  ingest.py              reads, cleans and chunks the corpus
  inspect\\\\\\\_pdf.py         shows font sizes, bold and footnote detection for one PDF (debugging aid)
  check\\\\\\\_stack.py         ChromaDB, embedding model, re-ranker and language model on this computer
  build\\\\\\\_index.py         embeds the chunks and fills the vector database
  ask.py                 asks a question (one shot or chat)
  serve.py               starts the web interface
  evaluate.py            measures retrieval and answer quality on the gold questions
src/reglens/
  config.py              settings from environment variables and .env
  diagnostics.py         the checks behind check\\\\\\\_setup.py
  pdfutil.py             quiet pypdf reading and readable PDF error messages
  corpus/manifest.py     loads and validates sources.csv
  corpus/download.py     downloader and file validation
  ingest/pdfread.py      reads PDF pages with the type size and weight of every line
  ingest/textclean.py    character, spacing, hyphenation and footnote-marker repairs
  ingest/layout.py       page headers, page numbers and footnotes
  ingest/inspector.py    the tool behind inspect\\\\\\\_pdf.py
  ingest/structure.py    headings and paragraphs
  ingest/webpage.py      HTML pages
  ingest/chunking.py     structure-aware chunks and the fixed-size baseline
  ingest/pipeline.py     the whole ingestion run, cache, audit and export
  embeddings/            text -> unit-length vectors: local sentence-transformers model, dependency-free hashing baseline
  vectorstore/           one interface, two stores: ChromaDB (approximate) and NumPy (exact); metadata filters
  retrieval/             BM25, rank fusion, glossary, query rewriting, cross-encoder re-ranking, context assembly
  generation/            prompts, citation and figure checks, the Answer object
  llm/                   Gemini, OpenAI and Anthropic over plain REST; retries and model fallback
  indexing.py            builds the index, records how it was built, refuses a mismatching one
  pipeline.py            question -> retrieval -> prompt -> model -> checks -> answer
  runtime.py             settings -> a ready pipeline (used by the command line, later by the web app)
  stackcheck.py          the checks behind check\\\\\\\_stack.py
  evaluation/            gold questions, retrieval and answer metrics, the optional judge, the report
  cli/                   the commands behind build\\\\\\\_index.py, ask.py, serve.py and evaluate.py
  web/service.py         what the page may ask for, as plain Python: requests, modes, the comparison with the model alone
  web/app.py             the HTTP layer (Starlette): routes, host and origin checks, security headers, static files
  web/static/            the page: index.html, app.css and ES modules (lib = page logic, render = answers, app = behaviour)
tests/                   unit tests, including a local test server for the downloader and
                         synthetic documents that imitate the layout of the real ones;
                         tests/js (page logic, Node.js) and tests/test\\\\\\\_web\\\\\\\_browser.py (real browser)
```

## Security

Never commit `.env` or paste an API key into a chat, issue or screenshot. `.gitignore` already
excludes `.env`. If a key leaks, create a new key in Google AI Studio and delete the old one.

