# Model service (ml.service): POST /analyze on :8001.
# Build from the repository root:  docker build -f ml/Dockerfile -t ktru-model .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only torch first (the default wheel pulls CUDA, several GB), then the rest
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
RUN pip install -r requirements.txt

# embedding model baked into the image (~470 MB): the container starts without network access
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

COPY ml ./ml
COPY data/processed/ktru.parquet ./data/processed/ktru.parquet
COPY data/synonyms.json data/value_aliases.json ./data/

# KTRU embeddings are computed once at build time (data/cache) so startup only loads them
RUN python -c "from ml.pipeline import Pipeline; Pipeline()" \
    && useradd --system --uid 10001 --no-create-home app \
    && chown -R app /app/data/cache

# torch takes every visible core by default; in a container that oversubscribes the CPU
# (measured: p50 ~106 ms with the default vs ~12 ms with 4 threads). Override with -e OMP_NUM_THREADS=...
ENV OMP_NUM_THREADS=4 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HOST=0.0.0.0 \
    PORT=8001 \
    LOG_LEVEL=INFO

USER app
EXPOSE 8001
# ready = pipeline loaded (model + KTRU index); /health only says the process is alive
HEALTHCHECK --interval=10s --timeout=3s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8001/ready', timeout=2).status != 200)"

CMD ["python", "-m", "ml.service"]
