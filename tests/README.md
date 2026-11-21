# VectorDB-Plugin Test Suite

These tests check the program's own code. Anything they write goes into temporary folders, so running them won't change your settings or databases.

## Running the tests

From the program's folder, with its virtual environment activated (`.\Scripts\activate`), install pytest once:

```
pip install pytest
```

Then run:

```
python -m pytest tests
```

A few tests skip themselves when they can't run on your computer: one needs a CUDA GPU, and the symlink tests need permission to create symlinks on Windows.

## Test files

| File | What it covers |
|------|----------------|
| `conftest.py` | Shared fixtures: a sample config, sample citation metadata and the project root |
| `test_backends_data.py` | Chat Backend Settings dialog: API keys, model choices and the reasoning efforts each model offers, the LM Studio address and port, and the LM Studio and ChatGPT connection checks |
| `test_chat_base.py` | `build_augmented_query`, maximum lengths, maximum new tokens, generation settings, `make_bnb_settings` and the Mistral prompt |
| `test_constants.py` | Required keys in the chat, embedding, vision and Whisper model lists, document loaders, thinking tags, pipeline presets and supported file types |
| `test_create_state.py` | Create Database tab: adding and removing files, the embedding model list, settings, build progress read from the build log, recording the new database with the settings it was built with, the checks before a build starts, including waiting for a backup or restore, reporting a failed backup of the new database, and holding off transcription from the PDF check until the build ends |
| `test_credentials.py` | Config file load and save round trip, and where each credential is stored |
| `test_database_info.py` | Each database's creation date and version in `metadata.db`: recorded when it is built, kept from the file's date when an older database is upgraded, raised only by committed changes, and read without writing |
| `test_database_additions.py` | Adding files to an existing database: versioned SQLite commit, duplicate skipping before embedding, original source paths with symlink or copy staging, supported formats and rollback after append |
| `test_database_interactions.py` | Model family detection, query prompts, text normalization, `metadata.db` creation, audio document loading, searches skipping removed chunks, searches scoring added chunks alongside the index, the file type filter reaching added chunks, and the prepare step leaving out documents before they are split or embedded |
| `test_document_ids.py` | Document ids in `metadata.db`: new databases, older mapping formats, the one-time upgrade of older databases (including a failed upgrade changing nothing) and batched lookups |
| `test_document_processor.py` | `FixedSizeTextSplitter` and PDF chunking: size, overlap with no repeated tail chunk, metadata and edge cases; the email, spreadsheet, text, CSV and Word loaders |
| `test_extract_metadata.py` | Content and file hashing, metadata extraction and PDF page numbers |
| `test_gui_validation.py` | Database name rules, port extraction, the file type filter mapping and name preconditions (no GUI imports) |
| `test_index_changes.py` | The record in `metadata.db` of documents added or removed since the index was built: present in new and upgraded databases, written only when a change commits, read without writing and cleared |
| `test_index_rebuild.py` | Rebuild index: a real index and chunk store rebuilt without removed chunks, a failed rebuild changing nothing, an empty database refused, and interrupted rebuilds finished or undone; new chunks appended by document addition and folded in by a rebuild, appends that were never committed dropped, and vectors of another size refused |
| `test_initialize.py` | Startup: recreating a missing `config.yaml`, reporting an unreadable one, and creating folders in the program folder |
| `test_jeeves_helpers.py` | Ask Jeeves: the poems file, source rows from the user guide and text prepared for speech |
| `test_kokoro_text.py` | Speech text: phoneme clauses rejoined with their punctuation, and sentence splitting around legal abbreviations |
| `test_local_model.py` | Local chat model process: answers from the passages it is given, returns their citations, and is reported if it dies while answering |
| `test_manage_data.py` | Manage Databases tab: the database list, database details (chunk counts and creation dates from `metadata.db`), file rows, read-only connections, safely deleting a database, backing up and restoring, backups that are out of date, changes since the index was built, starting a rebuild of the index, and the stats worker noting when it was interrupted |
| `test_markdown_js.py` | Answer rendering in `markdown.js`, run through Node.js (skipped without it): headings, code blocks under list items, nested lists, links and double-underscore names |
| `test_minimax.py` | MiniMax constants, model list and temperature clamping |
| `test_models_catalog.py` | Models tab: CPU-only filtering, download status, benchmark summaries, precision compared with the embedding code, theme colors, and what a failed download leaves behind |
| `test_openai_chat.py` | ChatGPT models, the reasoning efforts each one offers, older saved choices replaced, and the request sent to OpenAI |
| `test_process_manager.py` | Process cleanup at exit: stops a running process and tolerates one that is already closed |
| `test_query_data.py` | Query Database tab: citations, token counts, answer text, readiness messages, the database list, local model flags, chunk rows, a voice recording with no speech, empty answers, a question not being sent after its turn has ended, and the no-chunks message naming a search term |
| `test_scraper.py` | HTML processing, the scraper registry and selectors, URL checks, file name cleanup, the Mintlify scraper's OpenAPI pages, alerts and redirects, the Transformers scraper's handling of Hugging Face's rate limit, a whole crawl that saves every linked page without queueing a link twice, re-fetching an empty page left by an interrupted save, URLs with brackets, and keeping failed links when a resume is interrupted |
| `test_search_process.py` | Database search process: the same database stays loaded, a new database or changed settings end the old process before loading, releasing, errors, crashes and timeouts |
| `test_settings_state.py` | Settings tab: reading, checking and saving settings, GPU-only choices, half precision, TTS backends with their device and precision choices and model cards, vision model cards and unreadable config files |
| `test_symlinks.py` | Symlink creation |
| `test_thinking_tags.py` | Thinking tag structure and filtering, and LM Studio hiding a model's thinking without losing the answer |
| `test_tools.py` | File hashing, markdown chunk extraction and chunk analysis |
| `test_tools_state.py` | Tools tab: Whisper models and precisions, transcription checks and results (including audio with no speech), scraping status including failed links and scrapes that saved nothing, vision, OCR and the backup summary, including databases changed since the backup backups that wait while another database task runs, and running tools keeping the window open |
| `test_utilities.py` | `normalize_chat_text`, `format_citations`, `prepare_long_path`, `normalize_text`, embedding batch sizes, package and installation checks, platform and Python version info, the device and precision each TTS backend runs in, and the database name checks before a build |
