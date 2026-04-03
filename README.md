# ShealtRI-YourHealthWiki

Sistema de Recuperación de Información (SRI) para el dominio de Salud y Medicina.

## Descripción

Sistema que permite realizar consultas en lenguaje natural sobre temas médicos y recupera documentos relevantes usando **LSI (Latent Semantic Indexing)**, con generación de respuestas enriquecidas vía RAG.

**Características principales:**
- ✅ Corrección ortográfica automática en queries (Trie-based spell checker)
- ✅ Preprocesamiento con spaCy (lemmatización, eliminación de stopwords)
- ✅ Modelo LSI con TF-IDF + TruncatedSVD
- ✅ Almacenamiento de dos niveles (Vector DB + Document Store)
- ✅ Arquitectura modular y extensible

## Arquitectura

**Pipeline + Microkernel** empaquetado como **Monolito Modular**.

### Flujo de Documentos (Indexación)
```
Raw Documents
    ↓
[TextProcessor - is_query=False]
    (normalize → tokenize → remove stopwords → lemmatize → filter)
    ↓ Tokens added to Spell Checker Vocabulary
    ↓
[Indexer]
    (build inverted_index, vocabulary)
    ↓
[TfidfProcessor.fit()]
    (learn TF-IDF matrix & weights)
    ↓
[LSIModel.fit()]
    (SVD, generate embeddings)
    ↓
[Storage: ChromaDB + DocumentStore]
```

### Flujo de Queries (Búsqueda)
```
User Query
    ↓
[TextProcessor - is_query=True]
    (normalize → tokenize → remove stopwords → lemmatize → filter)
    ↓ Tokens corrected using Spell Checker Vocabulary
    ↓
[Indexer]
    (build query IndexedCorpus)
    ↓
[TfidfProcessor.transform()]
    (vectorize with learned vocabulary)
    ↓
[LSIModel.project_query()]
    (project to latent space)
    ↓
[ChromaDB.search_similar()]
    (cosine similarity search)
    ↓
[Return top-k documents]
```

## Estructura del proyecto

```
├── core/                     # Núcleo: interfaces ABC, modelos, config
├── modules/                  # Módulos del pipeline
│   ├── document_loader/      # Carga de documentos desde archivos
│   ├── text_processor/       # Preprocesamiento (spaCy) + Spell Checker (Trie)
│   ├── indexer/              # Construcción de IndexedCorpus
│   └── retriever/            # Modelo LSI (TF-IDF + TruncatedSVD)
├── infra/                    # ChromaDB, almacenamiento
├── docs/                     # Documentación LNCS
├── Dockerfile               # Reproducibilidad desde Corte 1
└── requirements.txt         # Dependencias
```

## Instalación

```bash
# Clonar repositorio
git clone <repo-url>
cd ShealtRI-YourHealthWiki

# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt

# Descargar modelo spaCy
python -m spacy download es_core_news_md
```

## Uso rápido

### Indexación (construye vocabulario y modelos)
```python
from core.models import Document
from modules.document_loader import DocumentLoader
from modules.text_processor import TextProcessor
from modules.indexer import IndexerService
from modules.retriever import LSIRetriever

# 1. Cargar documentos
loader = DocumentLoader()
documents = loader.load_from_directory("data/raw/")

# 2. Preprocesar (tokens se añaden al spell checker)
processor = TextProcessor()
indexer = IndexerService(text_processor=processor)
corpus = indexer.build(documents)

# 3. Entrenar retriever
retriever = LSIRetriever(repository=..., document_store=...)
retriever.fit(corpus)
```

### Búsqueda (corrige queries automáticamente)
```python
from core.models import Query

# Query con error ortográfico
query_text = "hipertensoin arterail"
query_corpus = indexer.build_query(query_text)

result = Query(text=query_text, indexed_corpus=query_corpus)
results = retriever.retrieve(result)

# Devuelve documentos sobre "hipertensión arterial"
for doc in results:
    print(f"{doc.document.text} (score: {doc.score:.3f})")
```

## Spell Checker

El corrector ortográfico está **integrado en el TextProcessor** y funciona automáticamente:

- **Para documentos** (`is_query=False`): Los tokens se añaden al vocabulario del Trie
- **Para queries** (`is_query=True`): Los tokens se corrigen usando el vocabulario conocido

Ver [SPELL_CHECKER_USAGE.md](SPELL_CHECKER_USAGE.md) para más detalles.

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.11+ |
| NLP | spaCy + NLTK (lemmatización, stopwords español) |
| Modelo LSI | scikit-learn (TruncatedSVD, TfidfVectorizer) |
| Spell Checker | Trie-based (Levenshtein distance) |
| Base vectorial | ChromaDB |
| RAG | LangChain + LLM API |
| UI | Streamlit |
| Contenedores | Docker + docker-compose |

## Documentación

- [Uso del Spell Checker](SPELL_CHECKER_USAGE.md)
- [Guía del Proyecto](CLAUDE.md)
- [Actualizaciones de Arquitectura](ARCHITECTURE_UPDATES.md)
- [Arquitectura del proyecto](docs/arch/arquitectura-proyecto-sri.md)
- [Almacenamiento de dos niveles](docs/arch/almacenamiento-dos-niveles.md)

## Testing

El proyecto incluye tests unitarios e integración para validar cada módulo:

```bash
# Ejecutar todos los tests
python -m pytest tests/ -v

# Con cobertura de código
python -m pytest tests/ --cov=modules --cov=core --cov-report=term-missing

# Tests en paralelo
python -m pytest tests/ -n auto
```

Ver [CLAUDE.md](CLAUDE.md) para más detalles sobre testing y patrón de mocks con interfaces ABC.

## Cortes de evaluación

- **Corte 1:** Crawler, indexer, retriever LSI, spell checker
- **Corte 2:** RAG, web search, plugins opcionales
- **Corte 3:** UI completa, ranking avanzado, evaluación

## Licencia

MIT

