import logging
import subprocess
import threading

logger = logging.getLogger(__name__)


def drain_subprocess(process, timeout, on_line=None):
    output_lines = []

    def _reader():
        for line in process.stdout:
            line = line.rstrip("\n")
            if line.strip():
                if on_line is not None:
                    on_line(line)
                output_lines.append(line)

    reader = threading.Thread(target=_reader, daemon=True)
    reader.start()
    reader.join(timeout)

    if reader.is_alive():
        logger.error(f"Subprocess exceeded {timeout}s without finishing; terminating it.")
        process.kill()
        reader.join(10)

    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        logger.error("Subprocess closed its output but did not exit; terminating it.")
        process.kill()
        process.wait()

    return output_lines
