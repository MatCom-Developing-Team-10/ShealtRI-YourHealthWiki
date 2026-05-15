# ShealtRI Console Interface Guide

## Setup

### Local Development (without Docker)

**Prerequisites:**
- Python 3.11+
- Virtual environment activated
- Dependencies installed

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install dependencies
pip install --timeout=300 --retries=20 -r requirements.txt

# Download spaCy Spanish model (one-time)
python -m spacy download es_core_news_md
```

### Corpus Setup

Place your documents in `data/raw/`:

```
data/raw/
├── libro-diabetes.pdf
├── libro-cardiologia.pdf
├── documentos.jsonl    # from crawler (optional)
└── ...
```

Supported formats: PDF, TXT, JSON, CSV, Markdown, JSONL

---

## Usage: Local

### Interactive Mode

```bash
python cli.py
```

You'll see:
```
[ShealtRI] Loading pipeline...
  loading NLP model (spaCy)... done
  reading documents from data/raw/... done
  source  : data/raw/ (N docs)
  indexing... done
  fitting LSI (n_components=100)... done

╔══════════════════════════════════════════════════╗
║         ShealtRI — Medical Information SRI       ║
║   Type a query, 'stats', 'help', or 'quit'       ║
╚══════════════════════════════════════════════════╝

Query> 
```

**Commands:**
- `<query>` — search for medical terms, e.g., `diabetes síntomas`
- `stats` — show corpus statistics (documents, terms, tokens)
- `help` — show available commands
- `quit` — exit

**Example session:**
```
Query> hipertensión arterial
  Results (3 found):
  
  1. [0.892] Hypertension Overview
       https://medlineplus.gov/hypertension
       La hipertensión arterial es una enfermedad cardiovascular...

  2. [0.756] Blood Pressure Management
       https://mayoclinic.org/blood-pressure
       El control de la presión arterial es fundamental...
       
Query> stats

  n_documents              : 42
  n_terms                  : 1258
  total_tokens             : 2894
  avg_tokens_per_doc       : 68.90
  avg_postings_per_term    : 2.30

Query> quit
Bye.
```

### One-shot Query

```bash
python cli.py --query "síntomas de diabetes"
```

Returns results and exits immediately.

### Statistics Only

```bash
python cli.py --stats
```

Shows corpus statistics without the interactive prompt.

### Custom Result Count

```bash
python cli.py --query "asma" --top-k 3
```

Returns top 3 results (default: 5).

---

## Usage: Docker

### Prerequisites

- Docker and Docker Compose installed
- PDFs placed in `data/raw/` (local directory)

### Build and Start

```bash
# First time (builds the image)
docker-compose up --build -d

# Subsequent starts
docker-compose up -d
```

The container will:
1. Install all dependencies
2. Download the spaCy model
3. Load your corpus from `data/raw/`
4. Build the LSI index
5. Start the CLI in interactive mode

### Interactive Mode (in Container)

```bash
docker-compose exec sri python cli.py
```

Same interface as local version. Exit with `quit` or `Ctrl+C`.

### One-shot Query (in Container)

```bash
docker-compose exec sri python cli.py --query "presión alta"
```

### Statistics (in Container)

```bash
docker-compose exec sri python cli.py --stats
```

### Run Tests (in Container)

```bash
# Smoke tests (stdlib only)
docker-compose exec sri python tests/smoke_test.py

# Integration tests (full pipeline)
docker-compose exec sri python -m pytest tests/integration/test_pipeline.py -v -s
```

### View Logs

```bash
docker-compose logs -f sri
```

### Stop Container

```bash
docker-compose down
```

---

## Performance Notes

- **First run:** ~30-60 seconds (spaCy model load, indexing, LSI fitting)
- **Subsequent runs:** <5 seconds (cached index and model)
- **PDF corpus:** Each page becomes one document; 300-page book = 300 docs
- **Query latency:** <100ms (LSI cosine similarity)

## Persistent Data

With Docker, these directories are mounted from your local filesystem:
- `./data/` — corpus documents, indices, ChromaDB
- `./models/` — serialized LSI models
- `./logs/` — application logs (if generated)

Changes inside the container persist after `docker-compose down`.

---

## Troubleshooting

### "spaCy model 'es_core_news_md' not found"
```bash
python -m spacy download es_core_news_md  # local
docker-compose exec sri python -m spacy download es_core_news_md  # docker
```

### "No results found"
- Corpus may be empty (check `data/raw/`)
- Query may be too specific or use unexpected terms
- Try `stats` to verify documents were loaded

### Docker container crashes on start
```bash
docker-compose logs sri
```

Check logs for errors. Rebuild if deps changed:
```bash
docker-compose up --build -d
```

### PDF not recognized
Ensure PDF is valid and readable:
```bash
file data/raw/your-file.pdf  # should show: PDF document, version 1.x
```

### Memory issues with large corpus
Limit memory in `docker-compose.yml`:
```yaml
mem_limit: 2g
cpus: '1.5'
```

---

## Next Steps

- **Corte 2:** Add RAG (Retrieval-Augmented Generation) with LLM responses
- **Corte 3:** Add Streamlit web UI (switches CMD to `streamlit run ui/app.py`)
