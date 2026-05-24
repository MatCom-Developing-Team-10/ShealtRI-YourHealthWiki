# WebSearch Module

## Overview

The `web_search` module provides a **keyword-based fallback retrieval strategy** for the SRI pipeline. It acts as a secondary retriever when the primary LSI retriever does not return sufficient results.

## Key Features

- **Keyword Extraction**: Normalizes and filters query text (stopword removal, lowercasing)
- **TF-IDF Scoring**: Ranks documents by term frequency and keyword density
- **Title Boosting**: Gives higher scores when keywords appear in document titles
- **Graceful Fallback**: Returns empty list rather than crashing on edge cases
- **Fully Testable**: Decoupled from storage layer via `DocumentStore` interface

## Architecture

### Relationship to Scrappers

The web_search module **does not directly invoke scrappers**. Instead:

1. **Crawler** (via scrappers) downloads and extracts content → raw JSONL files
2. **Indexer** processes documents → builds indices
3. **LSI Retriever** queries using latent semantic analysis
4. **Web Search** (fallback) queries using keyword matching on the indexed corpus

The corpus is the same pool of documents that came from the scrappers.

### Integration Point

```
Query
  ↓
LSI Retriever (primary strategy)
  ↓
  [< min_results?]
  ↓
Web Search (fallback strategy) ← You are here
  ↓
RetrievedDocuments
```

## Usage

### Standalone Usage

```python
from modules.web_search import WebSearchRetriever
from infra.chroma_repository import ChromaRepository
from modules.document_loader import DocumentStore

# Initialize with a document store
doc_store = MyDocumentStore()  # Must implement DocumentStore interface
retriever = WebSearchRetriever(document_store=doc_store)

# Retrieve documents
from core.models import Query
query = Query(text="symptoms of hypertension")
results = retriever.retrieve(query, top_k=10)

for result in results:
    print(f"{result.document.metadata.get('title')}: {result.score:.2f}")
```

### Pipeline Integration

```python
from core.pipeline import RetrievalContext
from modules.web_search import WebSearchRetriever

# Create retrieval context with web_search as fallback
fallback_retriever = WebSearchRetriever(
    document_store=doc_store,
    min_results=5,
)

context = RetrievalContext(strategy=fallback_retriever)
results = context.execute_search(query, top_k=10)
```

### Conditional Fallback

```python
# Use LSI first, then fall back to web_search if needed
lsi_results = lsi_retriever.retrieve(query, top_k=10)

if len(lsi_results) < 5:  # min_results threshold
    fallback_results = web_search_retriever.retrieve(query, top_k=5)
    lsi_results.extend(fallback_results)
```

## Implementation Details

### Keyword Extraction

The `_extract_keywords()` method:

1. Tokenizes text into words (using regex `\b[a-záéíóúña-z0-9-]+\b`)
2. Converts to lowercase
3. Filters out stopwords (English + medical common words)
4. Filters out words with < 3 characters

Example:
```python
"the symptoms of high blood pressure"
→ ["symptoms", "high", "blood", "pressure"]  # "the", "of" removed
```

### Document Scoring

The `_compute_score()` method uses a simple TF-IDF approach:

```
score = Σ(keyword_frequency / normalized_doc_length) × title_boost
```

Where:
- **keyword_frequency**: Number of times the keyword appears in the document
- **normalized_doc_length**: Document length normalized to avoid bias toward long docs
- **title_boost**: 1.5x multiplier if keyword appears in document title

Example:
```
Query: "hypertension blood pressure"
Doc A (text contains "hypertension" 2x, "blood" 1x, title="Understanding Hypertension"):
  score = (2 + 1) / norm_length × 1.5 = 3.5

Doc B (text contains "hypertension" 1x, no title match):
  score = 1 / norm_length = 0.5

Result: Doc A ranked higher
```

### Normalization

All scores are clamped to [0.0, 1.0] for consistency with other retrievers:

```python
normalized_score = min(raw_score, 1.0)
```

## DocumentStore Interface

The module requires a `DocumentStore` that implements:

```python
class DocumentStore(ABC):
    def get_by_id(self, doc_id: str) -> Document | None:
        """Retrieve a single document."""

    def list_all_ids(self) -> list[str]:
        """List all document IDs (required for web_search)."""
```

If the store doesn't implement `list_all_ids()`, the retriever logs an error and returns an empty list.

## Performance Considerations

### Current Limitations

- **Naive iteration**: Loops through all documents in the store
- **No caching**: Scores are recomputed on every query
- **Memory**: Full document text is loaded for scoring

### Optimizations for Production

1. **Inverted Index**: Cache keyword → [doc_ids] mapping
2. **Lazy Scoring**: Only score documents in the candidate set
3. **Disk-based Iteration**: Stream documents from storage instead of loading all into memory
4. **TF-IDF Cache**: Pre-compute document term frequencies

## Testing

Run tests with:

```bash
python -m pytest modules/web_search/test_service.py -v
```

Test coverage includes:
- Keyword extraction (stopwords, normalization, empty queries)
- Document scoring (keyword matches, title boosts, empty docs)
- Full retrieval (ranking, top_k limit, error handling)
- Integration (missing docs, unsupported store interfaces)

## Configuration

The module currently has no configuration file. To customize:

```python
# Adjust stopwords
retriever.STOPWORDS.add("custom_term")

# Adjust title boost factor (currently hardcoded as 1.5)
# → Would require modification to _compute_score()
```

## Future Enhancements

- [ ] Support for query expansion (synonyms, related terms)
- [ ] Integration with spell-correction module (Trie-based)
- [ ] Ranking by document metadata (recency, source authority)
- [ ] Hybrid scoring (combine LSI + keyword scores)
- [ ] BM25 ranking instead of TF-IDF
- [ ] Language-aware tokenization (handle medical terminology better)

## References

- **Core Interfaces**: `core/interfaces.py`
- **Document Models**: `core/models.py`
- **Tests**: `modules/web_search/test_service.py`
- **CLAUDE.md**: Project architecture and conventions
