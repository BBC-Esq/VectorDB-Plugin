from pathlib import Path
from multiprocessing import Pool, cpu_count

from PySide6.QtCore import QElapsedTimer, QThread, Signal

from db.create_symlinks import _create_single_symlink

ALLOWED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".eml",
    ".msg",
    ".csv",
    ".xls",
    ".xlsx",
    ".xlsm",
    ".rtf",
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".html",
    ".htm",
    ".md",
}


class SymlinkWorker(QThread):
    progress = Signal(int)
    finished = Signal(int, list)

    def __init__(self, source, target_dir, parent=None):
        super().__init__(parent)
        self.source = source
        self.target_dir = Path(target_dir)

    def run(self):
        if isinstance(self.source, (str, Path)):
            dir_path = Path(self.source)
            try:
                import os
                filenames = os.listdir(str(dir_path))
                files = [
                    str(dir_path / filename)
                    for filename in filenames
                    if (dir_path / filename).is_file() 
                    and (dir_path / filename).suffix.lower() in ALLOWED_EXTENSIONS
                ]
            except OSError:
                files = []
                print(f"Error accessing directory {dir_path}")
        else:
            files = list(self.source)

        total = len(files)
        made = 0
        errors = []
        last_pct = -1
        timer = QElapsedTimer()
        timer.start()
        step = max(1, total // 100) if total else 1

        if total > 1000:
            processes = min((total // 10000) + 1, cpu_count())
            file_args = [(f, str(self.target_dir)) for f in files]
            with Pool(processes=processes) as pool:
                for i, (ok, err) in enumerate(
                    pool.imap_unordered(_create_single_symlink, file_args), 1
                ):
                    if self.isInterruptionRequested():
                        pool.terminate()
                        break
                    if ok:
                        made += 1
                    if err:
                        errors.append(err)
                    if i % step == 0 or i == total:
                        pct = int(i * 100 / total) if total else 100
                        if pct != last_pct and timer.elapsed() > 500:
                            self.progress.emit(pct)
                            last_pct = pct
                            timer.restart()
        else:
            for f in files:
                if self.isInterruptionRequested():
                    break

                ok, err = _create_single_symlink((f, str(self.target_dir)))
                if ok:
                    made += 1
                if err:
                    errors.append(err)
                if made % step == 0 or made == total:
                    pct = int(made * 100 / total) if total else 100
                    if pct != last_pct and timer.elapsed() > 500:
                        self.progress.emit(pct)
                        last_pct = pct
                        timer.restart()

        self.finished.emit(made, errors)
