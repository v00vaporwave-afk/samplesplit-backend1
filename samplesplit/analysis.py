from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from .processing import ProcessingError


def _analysis_python() -> str:
    return os.getenv("SAMPLESPLIT_ANALYSIS_PYTHON", os.getenv("SAMPLESPLIT_DEMUCS_PYTHON", sys.executable))


def _run(mode: str, paths: list[Path], timeout: int) -> dict:
    command = [_analysis_python(), "-m", "samplesplit.analysis_worker", mode, *(str(path) for path in paths)]
    environment = os.environ.copy()
    environment.setdefault("HF_HOME", str(Path(tempfile.gettempdir()) / "samplesplit-huggingface"))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProcessingError(f"Audio analysis could not run: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        raise ProcessingError(f"Audio analysis failed: {detail[-1] if detail else 'unknown error'}")
    try:
        return json.loads(result.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise ProcessingError("Audio analysis returned an invalid result.") from exc


def classify_stems(stem_paths: dict[str, Path]) -> dict:
    ordered = list(stem_paths.items())
    result = _run("classify", [path for _, path in ordered], timeout=15 * 60)
    by_filename = result["instruments"]
    return {
        stem: {**by_filename[path.stem], "source_stem": stem}
        for stem, path in ordered
    }


def detect_music_metadata(source: Path) -> dict:
    return _run("music", [source], timeout=5 * 60)
