# SampleSplit API on Railway

## Build

Upload this folder to a GitHub repository, connect that repository to Railway, and deploy it as a service. Railway will detect the `Dockerfile` at the repository root and build the container.

The container starts the API with:

```sh
uvicorn samplesplit.app:app --host 0.0.0.0 --port ${PORT:-8000}
```

Set Railway's health-check path to `/api/health`. It should return `{"status":"ok"}`.

## Required environment variables

Set the exact public address of the SampleSplit frontend:

```text
SAMPLESPLIT_ALLOWED_ORIGINS=https://your-frontend.example.com
```

Railway provides `PORT` automatically. No API key is required for the current Demucs, librosa, or CLAP processing.

## Storage

Uploads and generated files use temporary storage and are deleted by the application when practical. Demucs and CLAP model weights download on first use.

For faster restarts, attach a persistent Railway volume at `/var/cache/samplesplit`. Allow several gigabytes for model caches. Because Railway volumes are mounted as root, set `RAILWAY_RUN_UID=0` when using this volume. Do not mount a volume over `/app`.

## Beta limitations

- Run one replica. Job status and result locations currently live in one process and its local filesystem.
- Deployments and restarts interrupt active processing jobs.
- The first job may be slow while model files download.
- CPU processing can be slow for long tracks; Demucs and PyTorch need substantial memory and CPU.
- The API accepts MP3 and WAV uploads up to 100 MB, subject to the service's available resources.
- Configure the frontend to call the generated Railway API address instead of localhost.
