import os
import sys

import pytest

from db.create_symlinks import _create_single_symlink, create_symlinks_parallel


def _can_create_symlinks():
    if sys.platform != 'win32':
        return True
    import tempfile
    try:
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "src.txt")
            lnk = os.path.join(d, "lnk.txt")
            with open(src, 'w') as f:
                f.write("test")
            os.symlink(src, lnk)
            return True
    except (OSError, NotImplementedError):
        return False


CAN_SYMLINK = _can_create_symlinks()
skip_no_symlink = pytest.mark.skipif(not CAN_SYMLINK, reason="Symlink privilege not available")


class TestCreateSingleSymlink:

    @skip_no_symlink
    def test_creates_symlink(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("hello")
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        success, error = _create_single_symlink((str(src), str(target_dir)))
        assert success is True
        assert error is None
        assert (target_dir / "source.txt").is_symlink()

    @skip_no_symlink
    def test_skips_existing(self, tmp_path):
        src = tmp_path / "source.txt"
        src.write_text("hello")
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        _create_single_symlink((str(src), str(target_dir)))
        success, error = _create_single_symlink((str(src), str(target_dir)))
        assert success is False
        assert error is None


class TestCreateSymlinksParallel:

    def test_nonexistent_target(self, tmp_path):
        count, errors = create_symlinks_parallel(str(tmp_path / "src"), str(tmp_path / "nonexistent"))
        assert count == 0
        assert errors == []

    @skip_no_symlink
    def test_from_directory(self, tmp_path):
        src_dir = tmp_path / "source"
        src_dir.mkdir()
        (src_dir / "a.txt").write_text("aaa")
        (src_dir / "b.txt").write_text("bbb")

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        count, errors = create_symlinks_parallel(str(src_dir), str(target_dir))
        assert count == 2
        assert errors == []

    @skip_no_symlink
    def test_from_list(self, tmp_path):
        f1 = tmp_path / "file1.txt"
        f2 = tmp_path / "file2.txt"
        f1.write_text("one")
        f2.write_text("two")

        target_dir = tmp_path / "target"
        target_dir.mkdir()

        count, errors = create_symlinks_parallel([str(f1), str(f2)], str(target_dir))
        assert count == 2
        assert errors == []

    @skip_no_symlink
    def test_empty_source(self, tmp_path):
        src_dir = tmp_path / "empty_source"
        src_dir.mkdir()
        target_dir = tmp_path / "target"
        target_dir.mkdir()

        count, errors = create_symlinks_parallel(str(src_dir), str(target_dir))
        assert count == 0
        assert errors == []
