import hashlib

import pytest


class TestHashFile:

    def _hash_file(self, filepath):
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def test_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_bytes(b"consistent content")
        assert self._hash_file(f) == self._hash_file(f)

    def test_different_content(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content A")
        f2.write_bytes(b"content B")
        assert self._hash_file(f1) != self._hash_file(f2)

    def test_matches_hashlib(self, tmp_path):
        content = b"test data for hashing"
        f = tmp_path / "test.txt"
        f.write_bytes(content)
        expected = hashlib.sha256(content).hexdigest()
        assert self._hash_file(f) == expected

    def test_empty_file(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_bytes(b"")
        expected = hashlib.sha256(b"").hexdigest()
        assert self._hash_file(f) == expected


class TestExtractChunks:

    def _extract_chunks(self, content):
        chunks = []
        current_chunk = []
        current_header = None

        for line in content.split('\n'):
            if line.startswith('### '):
                if current_header is not None:
                    chunks.append({
                        'header': current_header,
                        'content': '\n'.join(current_chunk).strip()
                    })
                current_header = line.strip()
                current_chunk = []
            else:
                current_chunk.append(line)

        if current_header is not None:
            chunks.append({
                'header': current_header,
                'content': '\n'.join(current_chunk).strip()
            })

        return chunks

    def test_basic_extraction(self):
        content = "### Header 1\nBody text here.\n### Header 2\nMore text."
        chunks = self._extract_chunks(content)
        assert len(chunks) == 2

    def test_multiple_sections(self):
        content = "### A\nText A\n### B\nText B\n### C\nText C"
        chunks = self._extract_chunks(content)
        assert len(chunks) == 3

    def test_preserves_header(self):
        content = "### My Section\nSome body text."
        chunks = self._extract_chunks(content)
        assert chunks[0]['header'] == '### My Section'

    def test_preserves_content(self):
        content = "### My Section\nLine 1\nLine 2"
        chunks = self._extract_chunks(content)
        assert 'Line 1' in chunks[0]['content']
        assert 'Line 2' in chunks[0]['content']

    def test_no_headers_returns_empty(self):
        content = "Just some text without any headers."
        chunks = self._extract_chunks(content)
        assert chunks == []

    def test_empty_content(self):
        chunks = self._extract_chunks("")
        assert chunks == []


class TestAnalyzeChunks:

    def _analyze(self, chunks):
        if not chunks:
            return {}
        lengths = [len(c['content']) for c in chunks]
        return {
            'count': len(chunks),
            'longest': max(lengths),
            'shortest': min(lengths),
            'average': sum(lengths) / len(lengths),
        }

    def test_finds_longest(self):
        chunks = [
            {'header': '### A', 'content': 'short'},
            {'header': '### B', 'content': 'this is much longer text'},
        ]
        analysis = self._analyze(chunks)
        assert analysis['longest'] == len('this is much longer text')

    def test_finds_shortest(self):
        chunks = [
            {'header': '### A', 'content': 'short'},
            {'header': '### B', 'content': 'this is much longer text'},
        ]
        analysis = self._analyze(chunks)
        assert analysis['shortest'] == len('short')

    def test_single_chunk(self):
        chunks = [{'header': '### A', 'content': 'only one'}]
        analysis = self._analyze(chunks)
        assert analysis['count'] == 1
        assert analysis['longest'] == analysis['shortest']

    def test_empty_returns_empty(self):
        analysis = self._analyze([])
        assert analysis == {}
