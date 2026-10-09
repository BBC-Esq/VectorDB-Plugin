import sys
import os
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def sample_config():
    return {
        'Compute_Device': {
            'available': ['cpu', 'cuda'],
            'database_creation': 'cuda',
            'database_query': 'cpu',
            'gpu_brand': 'NVIDIA',
        },
        'EMBEDDING_MODEL_DIMENSIONS': 384,
        'EMBEDDING_MODEL_NAME': 'Models/vector/BAAI--bge-small-en-v1.5',
        'database': {
            'chunk_overlap': 250,
            'chunk_size': 700,
            'contexts': '5',
            'database_to_search': '',
            'document_types': '',
            'half': False,
            'search_term': '',
            'similarity': 0.8,
        },
        'created_databases': {},
        'hf_access_token': None,
        'minimax': {
            'api_key': None,
            'model': 'MiniMax-M2.7',
        },
        'openai': {
            'api_key': None,
            'model': 'gpt-4o-mini',
            'reasoning_effort': 'medium',
        },
        'server': {
            'api_key': '',
            'connection_str': 'http://localhost:1234/v1',
            'show_thinking': False,
        },
    }


@pytest.fixture
def config_file(tmp_path, sample_config):
    config_path = tmp_path / "config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(sample_config, f)
    return config_path


@pytest.fixture
def sample_metadata_list():
    return [
        {
            'file_path': '/docs/report.pdf',
            'file_name': 'report.pdf',
            'file_type': '.pdf',
            'similarity_score': 0.9234,
            'page_number': 1,
        },
        {
            'file_path': '/docs/report.pdf',
            'file_name': 'report.pdf',
            'file_type': '.pdf',
            'similarity_score': 0.8567,
            'page_number': 2,
        },
        {
            'file_path': '/docs/report.pdf',
            'file_name': 'report.pdf',
            'file_type': '.pdf',
            'similarity_score': 0.8901,
            'page_number': 3,
        },
        {
            'file_path': '/docs/notes.txt',
            'file_name': 'notes.txt',
            'file_type': '.txt',
            'similarity_score': 0.7654,
        },
    ]
