import hashlib
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Union, List, Tuple


def _points_to(link_path: Path, source_path) -> bool:
    try:
        return link_path.is_symlink() and link_path.resolve() == Path(source_path).resolve()
    except Exception:
        return False


def _create_single_symlink(args):
    source_path, target_dir = args
    try:
        source = Path(source_path)
        target = Path(target_dir)
        link_path = target / source.name
        if not link_path.exists():
            link_path.symlink_to(source_path)
            return True, None
        if _points_to(link_path, source_path):
            return False, None
        suffix_hash = hashlib.md5(str(source).encode("utf-8")).hexdigest()[:8]
        disambiguated = target / f"{source.stem}_{suffix_hash}{source.suffix}"
        if not disambiguated.exists():
            disambiguated.symlink_to(source_path)
            return True, None
        if _points_to(disambiguated, source_path):
            return False, None
        return False, f"Symlink collision could not be resolved for {source.name}"
    except Exception as e:
        return False, f"Error creating symlink for {Path(source_path).name}: {str(e)}"

def create_symlinks_parallel(source: Union[str, Path, List[str], List[Path]], 
                           target_dir: Union[str, Path] = "Docs_for_DB") -> Tuple[int, list]:
    target_dir = Path(target_dir)
    if not target_dir.exists():
        print(f"Target directory does not exist: {target_dir}")
        return 0, []

    try:
        if isinstance(source, (str, Path)) and not isinstance(source, list):
            source_dir = Path(source)
            if not source_dir.exists():
                raise ValueError(f"Source directory does not exist: {source_dir}")
            files = [(str(p), str(target_dir)) for p in source_dir.iterdir() if p.is_file()]

        elif isinstance(source, list):
            files = [(str(Path(p)), str(target_dir)) for p in source]

        else:
            raise ValueError("Source must be either a directory path or a list of file paths")

        file_count = len(files)
        if file_count <= 1000:
            results = [_create_single_symlink(file) for file in files]
        else:
            if file_count <= 10000:
                processes = 1
            else:
                processes = min((file_count // 10000) + 1, cpu_count())

            print(f"Processing {file_count} files using {processes} processes")

            with Pool(processes=processes) as pool:
                results = pool.map(_create_single_symlink, files)

        count = sum(1 for success, _ in results if success)
        errors = [error for _, error in results if error is not None]
        
        print(f"\nComplete! Created {count} symbolic links")
        if errors:
            print("\nErrors occurred:")
            for error in errors:
                print(error)

        return count, errors

    except Exception as e:
        raise RuntimeError(f"An error occurred: {str(e)}")
