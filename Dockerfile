# Multi-stage build for optimized image size
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
# build-essential: for compiling Python packages
# git: for version control inside container
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (leverage Docker layer caching)
# Each file is a separate layer — only rebuilds when that file changes
COPY requirements.txt requirements-ui.txt ./

# Install runtime + UI deps — extended timeout for slow connections
RUN pip install --no-cache-dir --timeout=120 --retries=5 -r requirements-ui.txt

# Download spaCy Spanish model after dependencies are installed
RUN python -m spacy download es_core_news_md

# Copy entire project
COPY . .

# Create necessary directories for persistent data
RUN mkdir -p /app/data /app/models /app/logs

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SRI_ENV=production

# Expose Streamlit port (8501 default)
EXPOSE 8501

# Health check (optional but recommended)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8501')" || exit 1

# Default command: run Streamlit UI
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
