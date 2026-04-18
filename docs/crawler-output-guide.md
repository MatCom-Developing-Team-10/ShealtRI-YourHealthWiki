# Crawler Output — Guide for the Indexer

This document explains the format and location of the raw data produced by the
crawler module, and how the indexer should consume it.

## File location

After running the crawler, one JSONL file is created per source inside `data/raw/`:

```
data/raw/
    mayo_clinic.jsonl
    medlineplus.jsonl
    nhs.jsonl
```

Each file is appended to on every crawl run. Call `RawDocumentStorage.clear(source_name)`
before a full re-crawl if you need to start fresh.

## File format

Each file follows the **JSONL** format — one JSON object per line, one line per document.

```json
{
  "doc_id": "5e8ff578-d973-52e2-807d-1f71b80a5c84",
  "text": "A congenital heart defect is a problem with the structure of the heart...",
  "url": "https://www.mayoclinic.org/diseases-conditions/congenital-heart-defects/...",
  "metadata": {
    "title": "Congenital heart defects in children",
    "source": "mayo_clinic",
    "language": "en",
    "date": "2024-01-15T00:00:00",
    "category": "disease"
  }
}
```

### Field reference

| Field | Type | Description |
|-------|------|-------------|
| `doc_id` | `str` | UUID5 derived from the URL. Stable across re-crawls — the same URL always produces the same ID. |
| `text` | `str` | Clean article text with no HTML tags. Paragraphs separated by `\n\n`. This is the field to index. |
| `url` | `str` | Canonical URL of the source page. |
| `metadata.title` | `str` | Article title as it appears on the site. |
| `metadata.source` | `str` | Source identifier: `"mayo_clinic"`, `"medlineplus"`, or `"nhs"`. |
| `metadata.language` | `str` | ISO 639-1 language code: `"en"` or `"es"`. |
| `metadata.date` | `str` | Last-modified or published date. Format varies by source (ISO 8601 when available). Empty string if unavailable. |
| `metadata.category` | `str` | Broad topic tag. Values: `"disease"`, `"symptom"`, `"procedure"`, `"drug"`, `"wellness"`, `"mental-health"`, `"health-topic"`, `"reference"`. |

## Reading the files in Python

There is already a `DocumentLoader` at `modules/document_loader/service.py` —
check it before writing your own loading code.

If you need to read the files directly:

```python
import json
from pathlib import Path
from core.models import Document

documents: list[Document] = []

for path in Path("data/raw").glob("*.jsonl"):
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            documents.append(Document(
                doc_id=record["doc_id"],
                text=record["text"],
                url=record["url"],
                metadata=record["metadata"],
            ))
```

## What the indexer needs

The indexer only needs two fields to build the corpus:

- **`text`** — the raw medical content to tokenize and vectorize.
- **`doc_id`** — must be preserved as-is. It is the stable identifier that
  connects a document across the entire pipeline (crawler → indexer → retriever → UI).

The `metadata` fields are not used during indexing itself, but they must be
stored alongside the vectors in ChromaDB so the retriever can return them with
the search results.

## Running the crawler

```python
from modules.crawler import CrawlerService, CrawlConfig
from modules.crawler.scrapers.mayo_clinic import MayoClinicScraper
from modules.crawler.scrapers.medlineplus import MedlinePlusScraper
from modules.crawler.scrapers.nhs import NHSScraper

result = CrawlerService(
    scrapers=[MayoClinicScraper(), MedlinePlusScraper(), NHSScraper()],
    config=CrawlConfig(delay_seconds=2.0),  # max_pages=None runs the full crawl
).run()

print(result)
# CrawlResult(saved=..., visited=..., ok=..., failed=..., rate=...%, duration=...s)
```

> **Note:** The full crawl across all three sources will download several thousand
> pages and take multiple hours. Use `max_pages=N` for testing.
