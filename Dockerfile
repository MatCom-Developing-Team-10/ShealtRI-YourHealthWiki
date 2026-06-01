FROM python:3.11-slim

WORKDIR /app

# build-essential is required to compile chromadb's hnswlib wheel on slim.
# It is removed after pip install to keep the runtime image small.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first to maximise Docker layer caching.
COPY requirements.txt requirements-ui.txt ./
RUN pip install --no-cache-dir -r requirements-ui.txt

# Pre-fetch heavy NLP assets at build time so the container never hits
# the network at startup or on the first query.
RUN python -m spacy download es_core_news_md \
    && python -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True)"

# Drop the compiler toolchain — no longer needed at runtime.
RUN apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/* /root/.cache/pip

# Copy the application code (data/, models/, logs/ are excluded via .dockerignore
# and arrive through compose volumes at runtime).
COPY . .

# Persistent paths are bind-mounted by docker-compose; create them so the app
# does not crash if it ever runs without compose.
RUN mkdir -p /app/data /app/models /app/logs

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SRI_ENV=production

EXPOSE 8501

CMD ["uvicorn", "ui.app:app", "--host", "0.0.0.0", "--port", "8501"]
