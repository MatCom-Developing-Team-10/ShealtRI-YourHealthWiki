---
name: docker-setup
description: Configuración de Docker y docker-compose para el proyecto SRI. Usa este skill cuando necesites crear o modificar el Dockerfile, docker-compose.yml, agregar dependencias al contenedor, resolver problemas de build, o asegurar que el sistema sea reproducible. El Dockerfile debe existir desde el Corte 1 — no dejarlo para después.
---

# Docker para el SRI

## Principio fundamental

Es un **monolito modular** = un solo servicio en Docker. No hay múltiples contenedores para cada módulo. Un Dockerfile, un servicio en docker-compose (más ChromaDB si se separa).

## Dockerfile base

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dependencias del sistema (para scraping, procesamiento, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias Python primero (cacheo de capas Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Puerto para la UI (Streamlit=8501, Gradio=7860)
EXPOSE 8501

# Comando por defecto
CMD ["streamlit", "run", "ui/app.py", "--server.address", "0.0.0.0"]
```

## docker-compose.yml

```yaml
version: "3.8"

services:
  sri:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data          # Datos del corpus (persistente)
      - ./models:/app/models      # Modelos entrenados (persistente)
    environment:
      - PYTHONUNBUFFERED=1
      - SRI_ENV=production
```

Si ChromaDB se usa como servidor separado (en vez de embebido):

```yaml
services:
  sri:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    depends_on:
      - chromadb
    environment:
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma

volumes:
  chroma_data:
```

## requirements.txt mínimo (Corte 1)

```
scikit-learn>=1.3
numpy>=1.24
requests>=2.31
beautifulsoup4>=4.12
lxml>=4.9
chromadb>=0.4
streamlit>=1.28
joblib>=1.3
```

Se agregan en cortes posteriores:
- Corte 2: `langchain`, `sentence-transformers`, `openai` (o similar para RAG)
- Corte 3: dependencias adicionales de UI, métricas de evaluación

## Reglas Docker del proyecto

1. **El Dockerfile existe desde el Corte 1.** No posponerlo.
2. **Todo debe funcionar con `docker-compose up --build`.** Sin pasos manuales adicionales.
3. **No dependencias en la nube.** Todo local y reproducible.
4. **Volúmenes para datos persistentes.** El corpus y los modelos entrenados sobreviven a rebuilds.
5. **`.dockerignore`** para evitar copiar cosas innecesarias:

```
.git
__pycache__
*.pyc
.env
.venv
node_modules
.claude/
docs/
tests/
*.md
```

## Comandos de desarrollo

```bash
# Build y arrancar
docker-compose up --build

# Solo build (sin arrancar)
docker-compose build

# Arrancar en background
docker-compose up -d

# Ver logs
docker-compose logs -f sri

# Ejecutar comando dentro del contenedor
docker-compose exec sri python -m modules.crawler.service

# Reconstruir índice LSI dentro del contenedor
docker-compose exec sri python -m modules.indexer.service --rebuild
```
