FROM python:3.10-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    TORCH_HOME=/var/cache/samplesplit/torch \
    HF_HOME=/var/cache/samplesplit/huggingface \
    SAMPLESPLIT_DEMUCS_PYTHON=/usr/local/bin/python \
    SAMPLESPLIT_ANALYSIS_PYTHON=/usr/local/bin/python \
    SAMPLESPLIT_FFMPEG=/usr/bin/ffmpeg

RUN apt-get update \
    && apt-get install --no-install-recommends --yes \
        ffmpeg \
        libgomp1 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN groupadd --system samplesplit \
    && useradd --system --gid samplesplit --create-home samplesplit \
    && mkdir -p /var/cache/samplesplit \
    && chown -R samplesplit:samplesplit /var/cache/samplesplit /app

COPY --chown=samplesplit:samplesplit samplesplit ./samplesplit
COPY --chown=samplesplit:samplesplit static ./static

USER samplesplit

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8000') + '/api/health', timeout=4)"

CMD ["sh", "-c", "uvicorn samplesplit.app:app --host 0.0.0.0 --port ${PORT:-8000}"]
