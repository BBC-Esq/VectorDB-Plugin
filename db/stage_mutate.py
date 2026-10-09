"""Run an existing-database mutation in a clean subprocess."""

import ctypes
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _setup_tiledb_dlls():
    import tiledb  # noqa: F401

    venv_root = os.path.dirname(os.path.dirname(sys.executable))
    site_packages = os.path.join(venv_root, "Lib", "site-packages")
    candidates = (
        os.path.join(site_packages, "tiledb.libs"),
        os.path.join(site_packages, "tiledb", "vector_search", "lib"),
    )
    for directory in candidates:
        if os.path.isdir(directory):
            try:
                os.add_dll_directory(directory)
            except OSError:
                pass
    for directory in candidates:
        if not os.path.isdir(directory):
            continue
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".dll"):
                try:
                    ctypes.CDLL(os.path.join(directory, filename))
                except Exception:
                    pass


def main():
    if len(sys.argv) != 4 or sys.argv[1] != "remove":
        print(
            f"Usage: {sys.argv[0]} remove <database_directory> <document_hash>",
            file=sys.stderr,
        )
        return 2

    _setup_tiledb_dlls()
    from db.database_mutations import remove_document

    result = remove_document(sys.argv[2], sys.argv[3])
    print("VECTORDB_MUTATION_RESULT " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
