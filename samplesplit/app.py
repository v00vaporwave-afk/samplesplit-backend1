from __future__ import annotations

import re
import shutil
import tempfile
import threading
import uuid
import logging
import importlib.util
import json
import os
import subprocess
import time
import wave
import zipfile
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .processing import ProcessingError, STEMS, process_track, separate_with_demucs
from .analysis import classify_stems, detect_music_metadata

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_AUDIO_SECONDS = 60
ALLOWED_EXTENSIONS = {".mp3", ".wav"}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
JOB_RETENTION_SECONDS = 2 * 60 * 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("samplesplit")

app = FastAPI(title="SampleSplit")
allowed_origins = [origin.strip() for origin in os.getenv("SAMPLESPLIT_ALLOWED_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
if allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


@app.exception_handler(HTTPException)
async def http_error(_request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": str(exc.detail)})


@app.exception_handler(RequestValidationError)
async def validation_error(_request: Request, _exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": "The upload request is missing a required field."})


@app.exception_handler(Exception)
async def unexpected_error(_request: Request, exc: Exception):
    logger.exception("unhandled_api_error", exc_info=exc)
    return JSONResponse(status_code=500, content={"error": "The audio processing backend encountered an unexpected error."})


def safe_name(filename: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(filename).stem).strip("_") or "track"
    return stem[:60] + Path(filename).suffix.lower()


def safe_track_stem(filename: str) -> str:
    return Path(safe_name(filename)).stem


def wav_duration(path: Path) -> float | None:
    try:
        with wave.open(str(path), "rb") as audio:
            return audio.getnframes() / audio.getframerate()
    except (wave.Error, OSError, ZeroDivisionError):
        return None


def source_audio_duration(path: Path) -> float | None:
    if path.suffix.lower() == ".wav":
        duration = wav_duration(path)
        if duration is not None:
            return duration
    configured_ffprobe = os.environ.get("SAMPLESPLIT_FFPROBE")
    if not configured_ffprobe:
        configured_ffmpeg = os.environ.get("SAMPLESPLIT_FFMPEG")
        sibling = Path(configured_ffmpeg).with_name("ffprobe") if configured_ffmpeg else None
        configured_ffprobe = str(sibling) if sibling and sibling.is_file() else shutil.which("ffprobe")
    if not configured_ffprobe:
        logger.error("audio_duration_probe_failed reason=ffprobe_unavailable")
        return None
    try:
        result = subprocess.run(
            [
                configured_ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error("audio_duration_probe_failed exit_code=%s reason=%s", result.returncode, result.stderr.strip())
            return None
        duration = float(result.stdout.strip())
        return duration if duration > 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        logger.error("audio_duration_probe_failed reason=%s", exc)
        return None


async def save_upload(upload: UploadFile, destination: Path) -> None:
    size = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="That file is larger than the 100 MB limit.")
            output.write(chunk)
    if size == 0:
        raise HTTPException(status_code=400, detail="That file is empty. Please choose another track.")


def validate_audio_header(path: Path, extension: str) -> None:
    header = path.read_bytes()[:12]
    is_wav = len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WAVE"
    is_mp3 = header[:3] == b"ID3" or (len(header) >= 2 and header[0] == 0xFF and header[1] & 0xE0 == 0xE0)
    if (extension == ".wav" and not is_wav) or (extension == ".mp3" and not is_mp3):
        raise HTTPException(status_code=415, detail="Unsupported file type. Please choose a valid MP3 or WAV file.")


def dependency_status() -> dict[str, bool]:
    return {
        "demucs": importlib.util.find_spec("demucs") is not None,
        "librosa": importlib.util.find_spec("librosa") is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


def run_job(job_id: str, source: Path, work_dir: Path) -> None:
    try:
        logger.info("job=%s processing_started type=%s", job_id, source.suffix.lower())
        with JOBS_LOCK:
            JOBS[job_id].update(status="processing", message="Separating stems and finding beats…")
        output_zip, bpm, loop_count = process_track(source, work_dir)
        with JOBS_LOCK:
            JOBS[job_id].update(
                status="complete", message=f"Created {loop_count} loops per stem at {bpm:.1f} BPM.",
                zip_path=str(output_zip), bpm=bpm, loop_count=loop_count,
            )
        logger.info("job=%s processing_complete bpm=%.1f loops_per_stem=%d", job_id, bpm, loop_count)
        threading.Timer(JOB_RETENTION_SECONDS, cleanup_job, args=(job_id,)).start()
    except ProcessingError as exc:
        logger.warning("job=%s processing_failed reason=%s", job_id, exc)
        shutil.rmtree(work_dir, ignore_errors=True)
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", message=str(exc))
    except Exception:
        logger.exception("job=%s unexpected_processing_error", job_id)
        shutil.rmtree(work_dir, ignore_errors=True)
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", message="Processing failed. Check the Terminal window for details.")


def run_separation_job(job_id: str, source: Path, work_dir: Path) -> None:
    try:
        logger.info("job=%s separation_started type=%s", job_id, source.suffix.lower())
        with JOBS_LOCK:
            JOBS[job_id].update(status="processing", stage="separating_stems", progress=20, message="Separating vocals, drums, bass, and other…")
        stem_paths = separate_with_demucs(source, work_dir)
        with JOBS_LOCK:
            JOBS[job_id].update(stage="analyzing_instruments", progress=68, message="Analyzing instruments with CLAP…")
        instrument_analysis = classify_stems(stem_paths)
        used_names: dict[str, int] = {}
        for stem in STEMS:
            base_name = instrument_analysis[stem]["name"]
            used_names[base_name] = used_names.get(base_name, 0) + 1
            instrument_analysis[stem]["export_name"] = base_name if used_names[base_name] == 1 else f"{base_name} {used_names[base_name]}"
        with JOBS_LOCK:
            JOBS[job_id].update(stage="detecting_music", progress=84, message="Detecting BPM and musical key…")
        music = detect_music_metadata(source)
        with JOBS_LOCK:
            JOBS[job_id].update(stage="creating_pack", progress=94, message="Creating your organized sample pack…")
        completed_at = time.time()
        durations = {stem: wav_duration(stem_paths[stem]) for stem in STEMS}
        with JOBS_LOCK:
            started_at = JOBS[job_id]["started_at"]
            JOBS[job_id].update(
                status="complete",
                message="Four stems are ready.",
                stem_paths={stem: str(stem_paths[stem]) for stem in STEMS},
                stems=list(STEMS),
                downloaded=[],
                processing_seconds=round(completed_at - started_at, 1),
                durations=durations,
                duration=next((value for value in durations.values() if value), None),
                analysis=instrument_analysis,
                bpm=music.get("bpm"),
                key=music.get("key"),
                key_confidence=music.get("key_confidence"),
                stage="complete",
                progress=100,
            )
        logger.info("job=%s separation_complete stems=%s", job_id, ",".join(STEMS))
        timer = threading.Timer(JOB_RETENTION_SECONDS, cleanup_job, args=(job_id,))
        timer.daemon = True
        timer.start()
    except ProcessingError as exc:
        logger.warning("job=%s separation_failed reason=%s", job_id, exc)
        shutil.rmtree(work_dir, ignore_errors=True)
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", message=str(exc))
    except Exception:
        logger.exception("job=%s unexpected_separation_error", job_id)
        shutil.rmtree(work_dir, ignore_errors=True)
        with JOBS_LOCK:
            JOBS[job_id].update(status="error", message="Stem separation failed unexpectedly.")


async def receive_audio(file: UploadFile, rights_confirmed: bool) -> tuple[Path, Path, str]:
    if not rights_confirmed:
        raise HTTPException(status_code=400, detail="You must confirm that you have permission to process this audio.")
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Please choose an MP3 or WAV file.")
    content_type = (file.content_type or "").lower()
    if content_type and not (content_type.startswith("audio/") or content_type == "application/octet-stream"):
        raise HTTPException(status_code=415, detail="That does not look like an audio file.")
    work_dir = Path(tempfile.mkdtemp(prefix="samplesplit_"))
    source = work_dir / safe_name(file.filename or f"track{extension}")
    try:
        await save_upload(file, source)
        validate_audio_header(source, extension)
        duration = source_audio_duration(source)
        if duration is None:
            raise HTTPException(status_code=422, detail="The audio duration could not be determined. Please choose another MP3 or WAV file.")
        if duration > MAX_AUDIO_SECONDS:
            raise HTTPException(status_code=413, detail="For this beta, tracks must be 60 seconds or shorter.")
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        await file.close()
    return source, work_dir, extension


@app.post("/api/process", status_code=202)
async def start_processing(
    file: UploadFile = File(...),
    rights_confirmed: bool = Form(...),
):
    source, work_dir, extension = await receive_audio(file, rights_confirmed)

    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "queued", "message": "Preparing your track…", "work_dir": str(work_dir)}
    threading.Thread(target=run_job, args=(job_id, source, work_dir), daemon=True).start()
    logger.info("job=%s upload_accepted type=%s", job_id, extension)
    return {"job_id": job_id}


@app.post("/api/separate", status_code=202)
async def start_separation(
    file: UploadFile = File(...),
    rights_confirmed: bool = Form(...),
):
    original_name = Path(file.filename or "track").name
    source, work_dir, extension = await receive_audio(file, rights_confirmed)
    job_id = uuid.uuid4().hex
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "queued",
            "stage": "uploading",
            "progress": 8,
            "message": "Preparing your track…",
            "work_dir": str(work_dir),
            "original_name": original_name,
            "track_stem": safe_track_stem(original_name),
            "started_at": time.time(),
        }
    threading.Thread(target=run_separation_job, args=(job_id, source, work_dir), daemon=True).start()
    logger.info("job=%s separation_upload_accepted type=%s", job_id, extension)
    return {"job_id": job_id}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/test-upload")
async def test_upload(file: UploadFile = File(...)):
    extension = Path(file.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported file type.")
    size = 0
    try:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="This file is larger than 100 MB.")
    finally:
        await file.close()
    if size == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    logger.info("temporary_upload_test_received type=%s size=%d", extension, size)
    return {"status": "received", "filename": Path(file.filename or "track").name, "size": size}


@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="This processing job was not found.")
        return {
            key: value
            for key, value in job.items()
            if key not in {"zip_path", "work_dir", "stem_paths", "started_at", "track_stem"}
        }


def cleanup_job(job_id: str) -> None:
    with JOBS_LOCK:
        job = JOBS.pop(job_id, None)
    if job:
        shutil.rmtree(job.get("work_dir", ""), ignore_errors=True)
        logger.info("job=%s temporary_files_deleted", job_id)


@app.get("/api/stems/{job_id}/download-all")
def download_all_stems(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("status") != "complete":
            raise HTTPException(status_code=404, detail="The separated stems are not ready.")
        stem_paths = {stem: Path(job.get("stem_paths", {}).get(stem, "")) for stem in STEMS}
        track_stem = job.get("track_stem", "Track")
        analysis = job.get("analysis", {})
        work_dir = Path(job["work_dir"])
    if any(not path.is_file() for path in stem_paths.values()):
        raise HTTPException(status_code=404, detail="One or more separated stem files are missing.")
    zip_path = work_dir / f"{track_stem}_SamplePack.zip"
    if not zip_path.is_file():
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for folder in ("Samples", "Stems", "Metadata", "Preview"):
                archive.writestr(f"SamplePack/{folder}/", "")
            for stem, path in stem_paths.items():
                detected_name = safe_track_stem(analysis.get(stem, {}).get("export_name", stem.title()))
                archive.write(path, f"SamplePack/Stems/{track_stem}_{detected_name}.wav")
            metadata = {
                "track": job.get("original_name", track_stem),
                "bpm": job.get("bpm"),
                "key": job.get("key"),
                "duration_seconds": job.get("duration"),
                "generated_samples": 0,
                "generated_stems": len(STEMS),
                "stems": [
                    {
                        "source_stem": stem,
                        "detected_name": analysis.get(stem, {}).get("name", stem.title()),
                        "category": analysis.get(stem, {}).get("group", "Other"),
                        "confidence": analysis.get(stem, {}).get("confidence"),
                    }
                    for stem in STEMS
                ],
            }
            archive.writestr("SamplePack/Metadata/pack_metadata.json", json.dumps(metadata, indent=2, ensure_ascii=False))
            archive.writestr(
                "SamplePack/Metadata/pack_info.txt",
                "\n".join([
                    f"Track: {metadata['track']}",
                    f"Detected BPM: {metadata['bpm'] if metadata['bpm'] is not None else 'Not available'}",
                    f"Estimated key: {metadata['key'] or 'Not available'}",
                    f"Duration: {metadata['duration_seconds'] if metadata['duration_seconds'] is not None else 'Not available'} seconds",
                    "Generated samples: 0",
                    f"Generated stems: {len(STEMS)}",
                    "Instrument names are AI-generated estimates.",
                ]),
            )
            archive.writestr("SamplePack/Samples/README.txt", "Loop chopping is not included in this version. No sample loops were generated.\n")
            archive.writestr("SamplePack/Preview/README.txt", "Use the SampleSplit results dashboard to preview every generated stem.\n")
    return FileResponse(zip_path, media_type="application/zip", filename=zip_path.name)


@app.get("/api/stems/{job_id}/{stem}")
def download_stem(job_id: str, stem: str):
    if stem not in STEMS:
        raise HTTPException(status_code=404, detail="That stem does not exist.")
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("status") != "complete":
            raise HTTPException(status_code=404, detail="The separated stems are not ready.")
        path = Path(job.get("stem_paths", {}).get(stem, ""))
        track_stem = job.get("track_stem", "Track")
        detected_name = job.get("analysis", {}).get(stem, {}).get("export_name", stem.title())
    if not path.is_file():
        raise HTTPException(status_code=404, detail="The separated stem file is missing.")
    filename = f"{track_stem}_{safe_track_stem(detected_name)}.wav"
    return FileResponse(path, media_type="audio/wav", filename=filename)


@app.get("/api/download/{job_id}")
def download(job_id: str, background_tasks: BackgroundTasks):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or job.get("status") != "complete":
            raise HTTPException(status_code=404, detail="Your sample pack is not ready.")
        zip_path = Path(job["zip_path"])
    background_tasks.add_task(cleanup_job, job_id)
    return FileResponse(zip_path, media_type="application/zip", filename="SampleSplit_sample_pack.zip", background=background_tasks)


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
