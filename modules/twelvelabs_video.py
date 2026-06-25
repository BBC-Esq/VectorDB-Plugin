"""TwelveLabs video ingestion.

Adds *video* files to the vector database, mirroring how ``modules/transcribe.py``
adds audio.  Whisper turns speech into text; here TwelveLabs Pegasus turns the
*whole* video (visuals + audio + on-screen text) into a description, and that
description is written to ``Docs_for_DB`` as a ``{page_content, metadata}`` JSON
- the exact shape ``CreateVectorDB.load_audio_documents`` already consumes, so
the rest of the pipeline (chunk -> embed -> store -> search) needs no changes.

The optional ``marengo_video_embedding`` helper returns TwelveLabs Marengo's
512-dim multimodal embedding for a video segment, for callers that want to embed
video directly rather than going through a text description.

This is an opt-in cloud backend: it requires a TwelveLabs API key and changes no
default behavior.  Get a free key at https://twelvelabs.io.
"""
import json
import os
from pathlib import Path

from twelvelabs import TwelveLabs
from twelvelabs.types.video_context import VideoContext_Url

from core.config import get_config
from core.constants import PROJECT_ROOT

# Pegasus = video understanding/analysis; Marengo = multimodal embeddings (512-dim).
PEGASUS_MODEL = "pegasus1.5"
MARENGO_MODEL = "marengo3.0"
MARENGO_DIM = 512

DEFAULT_PROMPT = (
    "Describe this video in detail. Cover what is shown visually, what is said "
    "or heard, any on-screen text, and the overall topic, so the description can "
    "answer questions about the video."
)
DEFAULT_MAX_TOKENS = 2048

DOCS_DIR = PROJECT_ROOT / "Docs_for_DB"


def _resolve_api_key(api_key=None):
    """Config ``twelvelabs.api_key`` first, then the TWELVELABS_API_KEY env var."""
    if api_key:
        return api_key
    try:
        cfg = getattr(get_config(), "twelvelabs", None)
    except Exception:
        cfg = None
    if cfg is not None:
        # AppConfig sub-sections are pydantic models; fall back to dict access too.
        key = getattr(cfg, "api_key", None) or (cfg.get("api_key") if isinstance(cfg, dict) else None)
        if key:
            return key
    return os.environ.get("TWELVELABS_API_KEY")


class TwelveLabsVideoProcessor:
    """Analyze a video with Pegasus and write a vector-DB document.

    Parameters mirror ``WhisperTranscriber``: construct, then call
    ``start_transcription_process(video)`` with a public video URL.
    Unlike Whisper, TwelveLabs runs server-side, so ``video`` is a URL the
    TwelveLabs API can fetch (not a local path).
    """

    def __init__(self, api_key=None, model=PEGASUS_MODEL, prompt=DEFAULT_PROMPT,
                 max_tokens=DEFAULT_MAX_TOKENS):
        key = _resolve_api_key(api_key)
        if not key:
            raise ValueError(
                "TwelveLabs API key not found. Set it in config.yaml under "
                "'twelvelabs: api_key:' or via the TWELVELABS_API_KEY environment "
                "variable. Get a free key at https://twelvelabs.io."
            )
        self.client = TwelveLabs(api_key=key)
        self.model = model
        self.prompt = prompt
        self.max_tokens = max_tokens

    def analyze(self, video_url):
        """Return Pegasus's text analysis of the video at ``video_url``."""
        response = self.client.analyze(
            model_name=self.model,
            video=VideoContext_Url(url=video_url),
            prompt=self.prompt,
            max_tokens=self.max_tokens,
        )
        return (response.data or "").strip()

    def start_transcription_process(self, video_url, name=None):
        """Analyze ``video_url`` and write ``Docs_for_DB/<name>.json``.

        ``name`` defaults to the URL's filename stem; returns the JSON path.
        """
        text = self.analyze(video_url)
        if not text:
            raise RuntimeError("TwelveLabs returned an empty analysis for the video.")
        return self._create_document_object(text, video_url, name)

    def _create_document_object(self, text, video_url, name=None):
        DOCS_DIR.mkdir(parents=True, exist_ok=True)

        if name is None:
            name = Path(video_url.split("?", 1)[0]).stem or "video"

        # No local file to stat (the video lives at a remote URL), so build
        # metadata directly rather than via extract_typed_metadata(path, ...).
        metadata = {
            "file_path": video_url,
            "file_name": Path(video_url.split("?", 1)[0]).name or name,
            "file_type": Path(video_url.split("?", 1)[0]).suffix or ".mp4",
            "document_type": "video",
            "source": "twelvelabs",
            "model": self.model,
        }

        doc_dict = {"page_content": text, "metadata": metadata}
        json_path = DOCS_DIR / f"{name}.json"
        json_path.write_text(json.dumps(doc_dict, indent=4), encoding="utf-8")
        return json_path


def marengo_video_embedding(video_url, api_key=None, model=MARENGO_MODEL):
    """Return Marengo's 512-dim embedding for the video at ``video_url``.

    The video segments are embedded; the first segment's vector is returned
    (use the SDK directly for per-segment vectors).
    """
    key = _resolve_api_key(api_key)
    if not key:
        raise ValueError(
            "TwelveLabs API key not found. Set it in config.yaml under "
            "'twelvelabs: api_key:' or via the TWELVELABS_API_KEY environment variable."
        )
    client = TwelveLabs(api_key=key)
    task = client.embed.tasks.create(
        model_name=model,
        video_url=video_url,
    )
    task = client.embed.tasks.wait_for_done(task_id=task.id)
    result = client.embed.tasks.retrieve(task_id=task.id)
    segments = result.video_embedding.segments
    return segments[0].float_


def marengo_text_embedding(text, api_key=None, model=MARENGO_MODEL):
    """Return Marengo's 512-dim embedding for a text query.

    Useful for searching a Marengo-embedded video index with a text query in
    the same 512-dim multimodal space.
    """
    key = _resolve_api_key(api_key)
    if not key:
        raise ValueError(
            "TwelveLabs API key not found. Set it in config.yaml under "
            "'twelvelabs: api_key:' or via the TWELVELABS_API_KEY environment variable."
        )
    client = TwelveLabs(api_key=key)
    response = client.embed.create(model_name=model, text=text)
    return response.text_embedding.segments[0].float_
