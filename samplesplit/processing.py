from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
import os
import logging
import threading
import time
from pathlib import Path

STEMS = ("vocals", "drums", "bass", "other")
DEMUCS_TIMEOUT_SECONDS = 30 * 60
DEMUCS_MODEL = os.environ.get("SAMPLESPLIT_DEMUCS_MODEL", "mdx_extra_q")
DEMUCS_JOB_LOCK = threading.Lock()
logger = logging.getLogger("samplesplit.processing")


class ProcessingError(RuntimeError):
    pass


def separate_with_demucs(source: Path, work_dir: Path) -> dict[str, Path]:
    output_dir = work_dir / "separated"
    demucs_source = source
    if source.suffix.lower() == ".mp3":
        ffmpeg = os.environ.get("SAMPLESPLIT_FFMPEG") or shutil.which("ffmpeg")
        if not ffmpeg:
            raise ProcessingError("MP3 decoding requires ffmpeg, but ffmpeg is not available.")
        demucs_source = work_dir / "demucs_input.wav"
        conversion = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(source), "-c:a", "pcm_s16le", str(demucs_source)],
            capture_output=True,
            text=True,
        )
        if conversion.returncode != 0 or not demucs_source.is_file():
            raise ProcessingError("MP3 conversion failed before stem separation.")
    demucs_python = os.environ.get("SAMPLESPLIT_DEMUCS_PYTHON", sys.executable)
    command = [
        demucs_python,
        "-m",
        "demucs",
        "--name",
        DEMUCS_MODEL,
        "--device",
        "cpu",
        "--jobs",
        "1",
        "--shifts",
        "0",
        "--overlap",
        "0.1",
        "--segment",
        "10",
        "--out",
        str(output_dir),
        str(demucs_source),
    ]
    environment = os.environ.copy()
    environment.setdefault("TORCH_HOME", str(Path.cwd() / ".model-cache"))
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    configured_ffmpeg = os.environ.get("SAMPLESPLIT_FFMPEG")
    if configured_ffmpeg:
        tool_dir = work_dir / "tools"
        tool_dir.mkdir(exist_ok=True)
        ffmpeg_link = tool_dir / "ffmpeg"
        if not ffmpeg_link.exists():
            ffmpeg_link.symlink_to(Path(configured_ffmpeg).resolve())
        environment["PATH"] = f"{tool_dir}{os.pathsep}{environment.get('PATH', '')}"
    logger.info("demucs_waiting model=%s device=cpu source=%s", DEMUCS_MODEL, demucs_source.name)
    with DEMUCS_JOB_LOCK:
        started_at = time.monotonic()
        logger.info("demucs_start model=%s device=cpu jobs=1 source=%s", DEMUCS_MODEL, demucs_source.name)
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                env=environment,
                timeout=DEMUCS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            logger.error("demucs_failed model=%s reason=timeout seconds=%s", DEMUCS_MODEL, DEMUCS_TIMEOUT_SECONDS)
            raise ProcessingError("Processing timed out during stem separation.") from exc
        except OSError as exc:
            logger.exception("demucs_failed model=%s reason=subprocess_launch_error detail=%s", DEMUCS_MODEL, exc)
            raise ProcessingError(f"Stem separation could not start: {exc}") from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        logger.error(
            "demucs_failed model=%s exit_code=%s reason=%s\nstdout:\n%s\nstderr:\n%s",
            DEMUCS_MODEL,
            result.returncode,
            detail or "No Demucs error detail was returned.",
            result.stdout,
            result.stderr,
        )
        final_detail = detail.splitlines()[-1] if detail else "No Demucs error detail was returned."
        raise ProcessingError(f"Stem separation failed. {final_detail}")

    logger.info(
        "demucs_complete model=%s device=cpu seconds=%.1f source=%s",
        DEMUCS_MODEL,
        time.monotonic() - started_at,
        demucs_source.name,
    )
    track_dir = output_dir / DEMUCS_MODEL / demucs_source.stem
    paths = {stem: track_dir / f"{stem}.wav" for stem in STEMS}
    missing = [stem for stem, path in paths.items() if not path.exists()]
    if missing:
        logger.error("demucs_failed model=%s reason=missing_outputs stems=%s", DEMUCS_MODEL, ",".join(missing))
        raise ProcessingError(f"Demucs did not create: {', '.join(missing)}")
    return paths


def detect_beats(source: Path):
    import librosa
    import numpy as np
    try:
        audio, sample_rate = librosa.load(source, sr=None, mono=True)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sample_rate, units="time")
    except Exception as exc:
        raise ProcessingError("Beat detection could not read this audio file.") from exc

    bpm = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = np.asarray(beats, dtype=float)
    if not np.isfinite(bpm) or bpm <= 0 or len(beat_times) < 17:
        raise ProcessingError("Not enough clear beats were found to make a complete 4-bar loop.")
    return bpm, beat_times


def make_loops(stem_paths: dict[str, Path], beat_times, pack_dir: Path) -> int:
    import soundfile as sf
    boundaries = list(range(0, len(beat_times) - 16, 16))
    if not boundaries:
        raise ProcessingError("The track is too short for a complete 4-bar loop.")

    loop_count = 0
    for stem in STEMS:
        audio, sample_rate = sf.read(stem_paths[stem], always_2d=True)
        stem_dir = pack_dir / stem
        stem_dir.mkdir(parents=True, exist_ok=True)
        for loop_index, beat_index in enumerate(boundaries, start=1):
            start = int(round(beat_times[beat_index] * sample_rate))
            end = int(round(beat_times[beat_index + 16] * sample_rate))
            if start < 0 or end > len(audio) or end <= start:
                continue
            sf.write(stem_dir / f"{stem}_loop_{loop_index:02d}.wav", audio[start:end], sample_rate, subtype="PCM_16")
            if stem == STEMS[0]:
                loop_count += 1
    if loop_count == 0:
        raise ProcessingError("No complete 4-bar loops fit inside the separated audio.")
    return loop_count


def create_zip(pack_dir: Path, bpm: float, output_zip: Path) -> Path:
    (pack_dir / "info.txt").write_text(
        f"Detected BPM: {bpm:.1f}\nLoop length: 4 bars\n", encoding="utf-8"
    )
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(pack_dir.rglob("*")):
            if path.is_file():
                archive.write(path, Path("sample_pack") / path.relative_to(pack_dir))
    return output_zip


def process_track(source: Path, work_dir: Path, separator=separate_with_demucs) -> tuple[Path, float, int]:
    stem_paths = separator(source, work_dir)
    bpm, beat_times = detect_beats(source)
    pack_dir = work_dir / "sample_pack"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    loop_count = make_loops(stem_paths, beat_times, pack_dir)
    output_zip = create_zip(pack_dir, bpm, work_dir / "sample_pack.zip")
    return output_zip, bpm, loop_count
