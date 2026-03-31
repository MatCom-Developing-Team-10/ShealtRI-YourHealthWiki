# ShealtRI-YourHealthWiki

Sistema de Recuperación de Información (SRI) para el dominio de Salud y Medicina.

## Descripción

Sistema que permite realizar consultas en lenguaje natural sobre temas médicos y recupera documentos relevantes usando **LSI (Latent Semantic Indexing)**, con generación de respuestas enriquecidas vía RAG.

## Arquitectura

**Pipeline + Microkernel** empaquetado como **Monolito Modular**.

```
Query → DocumentLoader → TextProcessor → Indexer → TF-IDF → LSI → ChromaDB → Results
```

## Estructura del proyecto

```
├── core/                     # Núcleo: interfaces ABC, modelos, config
├── modules/                  # Módulos del pipeline
│   ├── document_loader/      # Carga de documentos desde archivos
│   ├── text_processor/       # Preprocesamiento de texto (NLTK)
│   ├── indexer/              # Construcción de IndexedCorpus
│   └── retriever/            # Modelo LSI (TF-IDF + TruncatedSVD)
├── infra/                    # ChromaDB, almacenamiento
├── docs/                     # Documentación LNCS
└── tests/                    # Tests unitarios y de integración
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
```

## Uso rápido

```python
from core.models import Document
from modules.document_loader import DocumentLoader
from modules.text_processor import TextProcessor
from modules.indexer import IndexerService
from modules.retriever import LSIRetriever

# 1. Cargar documentos
loader = DocumentLoader()
documents = loader.load_from_directory("data/raw/")

# 2. Preprocesar y construir corpus
processor = TextProcessor()
indexer = IndexerService(text_processor=processor)
corpus = indexer.build(documents)

# 3. Entrenar retriever
retriever = LSIRetriever(repository=..., document_store=...)
retriever.fit(corpus)

# 4. Buscar
from core.models import Query
results = retriever.retrieve(Query(text="síntomas de hipertensión"))
```

## Stack tecnológico

| Componente | Tecnología |
|------------|------------|
| Lenguaje | Python 3.11+ |
| Modelo LSI | scikit-learn (TruncatedSVD, TfidfVectorizer) |
| Base vectorial | ChromaDB |
| NLP | NLTK (tokenización, stemming español) |
| RAG | LangChain + LLM API |
| UI | Streamlit |

## Documentación

- [Arquitectura del proyecto](docs/arch/arquitectura-proyecto-sri.md)
- [Almacenamiento de dos niveles](docs/arch/almacenamiento-dos-niveles.md)
- [Análisis del modelo](docs/Analisis%20del%20Modelo.md)

## Cortes de evaluación

- **Corte 1:** Crawler, indexer, retriever básico
- **Corte 2:** RAG, web search, plugins opcionales
- **Corte 3:** UI completa, ranking avanzado, evaluación

## Licencia

MIT
