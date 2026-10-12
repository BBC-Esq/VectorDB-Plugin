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
| `test_backends_data.py` | Chat Backend Settings dialog: API keys, model choices, the LM Studio address and port, and the LM Studio and ChatGPT connection checks |
| `test_chat_base.py` | `build_augmented_query`, maximum lengths, maximum new tokens, generation settings and `make_bnb_settings` |
| `test_constants.py` | Required keys in the chat, embedding, vision and Whisper model lists, document loaders, thinking tags, pipeline presets and supported file types |
| `test_create_state.py` | Create Database tab: adding and removing files, the embedding model list, settings, build progress read from the build log, recording the new database and the checks before a build starts, including waiting for a backup or restore |
| `test_credentials.py` | Config file load and save round trip, and where each credential is stored |
| `test_database_info.py` | Each database's creation date and version in `metadata.db`: recorded when it is built, kept from the file's date when an older database is upgraded, raised only by committed changes, and read without writing |
| `test_database_interactions.py` | Model family detection, query prompts, text normalization, `metadata.db` creation and audio document loading |
| `test_document_ids.py` | Document ids in `metadata.db`: new databases, older mapping formats, the one-time upgrade of older databases (including a failed upgrade changing nothing) and batched lookups |
| `test_document_processor.py` | `FixedSizeTextSplitter` chunking: size, overlap, metadata and edge cases |
| `test_extract_metadata.py` | Content and file hashing, metadata extraction and PDF page numbers |
| `test_gui_validation.py` | Database name rules, port extraction, the file type filter mapping and name preconditions (no GUI imports) |
| `test_jeeves_helpers.py` | Ask Jeeves: the poems file, source rows from the user guide and text prepared for speech |
| `test_local_model.py` | Local chat model process: answers from the passages it is given and returns their citations |
| `test_manage_data.py` | Manage Databases tab: the database list, database details (chunk counts and creation dates from `metadata.db`), file rows, read-only connections, safely deleting a database, backing up and restoring, and backups that are out of date |
| `test_minimax.py` | MiniMax constants, model list and temperature clamping |
| `test_models_catalog.py` | Models tab: CPU-only filtering, download status, benchmark summaries, precision compared with the embedding code, and theme colors |
| `test_process_manager.py` | Process cleanup at exit: stops a running process and tolerates one that is already closed |
| `test_query_data.py` | Query Database tab: citations, token counts, answer text, readiness messages, the database list, local model flags and chunk rows |
| `test_scraper.py` | HTML processing, the scraper registry and selectors, URL checks and file name cleanup |
| `test_search_process.py` | Database search process: the same database stays loaded, a new database or changed settings end the old process before loading, releasing, errors, crashes and timeouts |
| `test_settings_state.py` | Settings tab: reading, checking and saving settings, GPU-only choices, half precision, TTS backends with their device and precision choices and model cards, vision model cards and unreadable config files |
| `test_symlinks.py` | Symlink creation |
| `test_thinking_tags.py` | Thinking tag structure and filtering |
| `test_tools.py` | File hashing, markdown chunk extraction and chunk analysis |
| `test_tools_state.py` | Tools tab: Whisper models and precisions, transcription checks, scraping status, vision, OCR and the backup summary, including databases changed since the backup and backups that wait while another database task runs |
| `test_utilities.py` | `normalize_chat_text`, `format_citations`, `prepare_long_path`, `normalize_text`, embedding batch sizes, package and installation checks, platform and Python version info, and the device and precision each TTS backend runs in |
