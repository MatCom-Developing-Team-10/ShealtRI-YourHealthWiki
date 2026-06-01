# ShealtRI — YourHealthWiki

Sistema de Recuperación de Información (SRI) para el dominio de **Salud y Medicina**.

Permite realizar consultas en lenguaje natural sobre temas médicos y recupera documentos
relevantes usando **LSI (Latent Semantic Indexing)**, con re-ranking híbrido, generación de
respuestas adaptadas al perfil del usuario vía **RAG**, búsqueda web de respaldo y
recomendación de lectura relacionada.

Proyecto integrador del curso **SRI 2025-2026**. Equipo de 3 personas, 3 cortes de evaluación.

---

## Características principales

- ✅ **Modelo LSI**: TF-IDF + `TruncatedSVD`, con proyección de queries al espacio latente y similitud coseno.
- ✅ **Corrección ortográfica** automática de queries (spell checker basado en Trie + distancia de Levenshtein).
- ✅ **Preprocesamiento NLP** con spaCy y NLTK (normalización, tokenización, stopwords en español, lematización).
- ✅ **Re-ranking híbrido** combinando score LSI con BM25 (`rank-bm25`).
- ✅ **RAG con perfiles de usuario** (paciente / estudiante / profesional) generado vía API de **Groq**.
- ✅ **Búsqueda web de respaldo** (DuckDuckGo + fetch/parse) cuando el corpus local no da resultados suficientes.
- ✅ **Recomendador** de documentos relacionados (content-based con diversificación MMR).
- ✅ **Retroalimentación de relevancia** (Rocchio) y **expansión de consultas** (tesauro de sinónimos médicos) como plugins opcionales.
- ✅ **Evaluación** con métricas P, R, F1 y NDCG.
- ✅ **Crawler** de fuentes médicas (MedlinePlus, Mayo Clinic, NHS) respetando `robots.txt`.
- ✅ **Almacenamiento de dos niveles**: Vector DB (ChromaDB) + Document Store en sistema de archivos.
- ✅ **Interfaz web** (FastAPI + SPA HTML/CSS/JS) y **CLI** interactiva.
- ✅ Reproducible con **Docker + docker-compose**.

---

## Arquitectura

**Pipeline + Microkernel**, empaquetado como **Monolito Modular** (un solo proceso Python, sin microservicios).

- El flujo de una consulta es un **pipeline**: entrada → corrección/parseo → recuperación → ranking → RAG → respuesta.
- Los módulos opcionales se conectan como **plugins** vía hooks: `pre_retrieval`, `post_retrieval`, `post_ranking`.
- Si un plugin no se registra, su hook es un *no-op*: el pipeline se comporta como si no existiera.

### Flujo de indexación (documentos)

```text
Raw Documents (data/raw/)
    ↓
[TextProcessor  is_query=False]
    normalize → tokenize → remove stopwords → lemmatize → filter
    ↓  (los tokens alimentan el vocabulario del Spell Checker)
[Indexer (+ Chunker)]
    índice invertido, vocabulario, IndexedCorpus
    ↓
[TfidfProcessor.fit()]   →   [LSIModel.fit()]  (SVD, embeddings)
    ↓
[Storage: ChromaDB + FileSystemDocumentStore]
```

### Flujo de consulta (búsqueda)

```text
User Query
    ↓
[TextProcessor  is_query=True]   →   tokens corregidos con el Spell Checker
    ↓
[Plugin hook: pre_retrieval]     →   expansión de sinónimos (opcional)
    ↓
[TfidfProcessor.transform()]     →   [LSIModel.project_query()]  (espacio latente)
    ↓
[ChromaRepository.search_similar()]  (similitud coseno, top-k)
    ↓  (si hay pocos resultados → FallbackRetriever → Web Search)
[Plugin hook: post_retrieval]
    ↓
[HybridRanker]  (LSI + BM25)     →   [Plugin hook: post_ranking]  (feedback Rocchio)
    ↓
[RAGService]  respuesta adaptada al perfil (Groq)  +  [Recommender]  lecturas relacionadas
    ↓
Respuesta + documentos
```

---

## Estructura del proyecto

```text
ShealtRI-YourHealthWiki/
├── core/                     # Pipeline, interfaces ABC, modelos, microkernel de plugins
│   ├── interfaces.py         #   contratos ABC (BaseRetriever, Plugin, ...)
│   ├── models.py             #   Document, Query, PipelineContext, UserProfile, ...
│   ├── pipeline.py           #   RetrievalContext
│   └── plugin_pipeline.py    #   orquestador de hooks (microkernel)
├── modules/                  # Módulos obligatorios del pipeline
│   ├── crawler/              #   scraping de fuentes médicas (MedlinePlus, Mayo, NHS) + robots.txt
│   ├── document_loader/      #   carga de PDF/TXT/JSON/CSV/MD/JSONL
│   ├── text_processor/       #   NLP (spaCy/NLTK) + Spell Checker (Trie)
│   ├── indexer/              #   IndexedCorpus, chunker, document_store, index_store
│   ├── retriever/            #   LSI (tfidf_processor, lsi_model) + FallbackRetriever
│   ├── ranker/               #   HybridRanker (LSI + BM25)
│   ├── rag/                  #   RAG (Groq), perfiles, prompt templates, evaluador
│   ├── web_search/           #   fallback DuckDuckGo + fetcher
│   ├── recommender/          #   recomendador content-based (MMR)
│   └── evaluation/           #   métricas P, R, F1, NDCG + dataset
├── plugins/                  # Módulos opcionales (microkernel)
│   ├── expansion/            #   expansión de consultas (tesauro de sinónimos)
│   └── feedback/             #   retroalimentación de relevancia (Rocchio)
├── infra/                    # ChromaRepository, almacenamiento
├── ui/                       # Interfaz web FastAPI (app.py + static/)
├── tests/                    # unit / integration / regression + smoke_test
├── docs/                     # Documentación técnica + informe LNCS
├── data/                     # raw/, documents/, chroma/, evaluation/
├── models/                   # Artefactos LSI serializados
├── cli.py                    # CLI real (pipeline completo)
├── cli_demo.py               # CLI demo (sin dependencias, scorer de keywords)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt          # dependencias de runtime
├── requirements-ui.txt       # + FastAPI/uvicorn (Corte 3)
└── requirements-dev.txt      # + pytest y herramientas de test
```

---

## Instalación

### Requisitos previos

- Python 3.11+
- (Opcional) Docker y Docker Compose

### Local

```bash
# Clonar el repositorio
git clone <repo-url>
cd ShealtRI-YourHealthWiki

# Crear y activar entorno virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt        # runtime (CLI / pipeline)
pip install -r requirements-ui.txt     # + interfaz web (FastAPI)
pip install -r requirements-dev.txt    # + tests

# Descargar el modelo de spaCy en español (una sola vez)
python -m spacy download es_core_news_md
```

> En Linux/macOS existe además el script `setup_local.sh` que automatiza estos pasos.

### Variables de entorno

Copia `.env.example` a `.env` y completa la clave de Groq para habilitar el RAG:

```bash
cp .env.example .env
# Edita .env y define:
# GROQ_API_KEY=tu_clave   (gratis en https://console.groq.com/keys)
```

Sin `GROQ_API_KEY` el sistema sigue funcionando, pero el RAG cae en una respuesta de plantilla
menos útil. El resto de la recuperación (LSI, ranking, búsqueda) no requiere la clave.

---

## Uso

### Corpus

Coloca tus documentos en `data/raw/`. Formatos soportados: **PDF, TXT, JSON, CSV, Markdown, JSONL**.
Si no hay archivos JSONL, la CLI carga automáticamente **20 documentos médicos sintéticos** de prueba.

### Interfaz web (FastAPI)

```bash
uvicorn ui.app:app --host 0.0.0.0 --port 8501
# Abre http://localhost:8501
```

### CLI interactiva

```bash
python cli.py                                   # REPL interactivo
python cli.py --query "síntomas de hipertensión"
python cli.py --query "diabetes tipo 2" --profile estudiante
python cli.py --query "asma" --top-k 3
python cli.py --stats                           # estadísticas del corpus
```

Comandos del REPL: `<query>`, `stats`, `help`, `quit`. Perfiles disponibles: `paciente` (por defecto),
`estudiante`, `profesional`. Guía completa en [CLI_GUIDE.md](CLI_GUIDE.md).

### CLI demo (sin dependencias)

Para verificar la interfaz cuando no hay dependencias pesadas instaladas ni Docker. Usa un corpus
sintético y un scorer por solapamiento de keywords en lugar de LSI:

```bash
python cli_demo.py
python cli_demo.py --query "diabetes tipo 2"
python cli_demo.py --stats
```

### Docker

```bash
# Primera vez (construye la imagen)
docker-compose up --build -d

# La app web queda en http://localhost:8501

# Ejecutar la CLI dentro del contenedor
docker-compose exec sri python cli.py

# Logs y parada
docker-compose logs -f sri
docker-compose down
```

Los directorios `./data`, `./models` y `./logs` se montan como volúmenes y persisten entre ejecuciones.

---

## Spell Checker

El corrector ortográfico está **integrado en el `TextProcessor`** y funciona automáticamente:

- **Documentos** (`is_query=False`): los tokens se añaden al vocabulario del Trie.
- **Queries** (`is_query=True`): los tokens se corrigen contra el vocabulario conocido
  usando distancia de Levenshtein.

Así, una query como `"hipertensoin arterail"` se corrige a `"hipertensión arterial"` antes de
vectorizarse. En la interfaz web el usuario puede decidir si usar la query corregida o la suya original.

---

## Stack tecnológico

| Componente            | Tecnología                                               |
|-----------------------|----------------------------------------------------------|
| Lenguaje              | Python 3.11+                                             |
| NLP                   | spaCy (`es_core_news_md`) + NLTK (stopwords, tokenización)|
| Modelo LSI            | scikit-learn (`TfidfVectorizer`, `TruncatedSVD`)         |
| Spell Checker         | Trie + distancia de Levenshtein                          |
| Ranking               | `rank-bm25` (BM25) + score LSI (HybridRanker)            |
| Base vectorial        | ChromaDB                                                 |
| Document Store        | Sistema de archivos (`FileSystemDocumentStore`)          |
| RAG                   | API de Groq (`groq`) con prompts por perfil de usuario   |
| Búsqueda web          | DuckDuckGo (`ddgs`) + requests + BeautifulSoup/lxml      |
| Interfaz web          | FastAPI + uvicorn + SPA HTML/CSS/JS                       |
| Persistencia modelos  | joblib                                                    |
| Contenedores          | Docker + docker-compose                                  |

---

## Testing

```bash
# Todos los tests
python -m pytest tests/ -v

# Con cobertura
python -m pytest tests/ --cov=modules --cov=core --cov-report=term-missing

# En paralelo
python -m pytest tests/ -n auto

# Smoke test (solo stdlib, sin dependencias pesadas)
python tests/smoke_test.py
```

La suite incluye tests **unitarios** (`tests/unit/`), de **integración** (`tests/integration/`)
y de **regresión** (`tests/regression/`). Cada módulo se testea de forma aislada mediante mocks
de las interfaces ABC. Configuración en [pytest.ini](pytest.ini).

---

## Cortes de evaluación

- **Corte 1** — Adquisición e indexación: crawler con `robots.txt`, indexer con índice invertido,
  retriever LSI básico, spell checker, base vectorial, Dockerfile.
- **Corte 2** — Integración avanzada: mejoras al LSI, RAG completo, búsqueda web de respaldo,
  plugins (expansión, feedback).
- **Corte 3** — Sistema completo: interfaz web, ranking híbrido, recomendador, evaluación
  (P, R, F1, NDCG), integración end-to-end.

---

## Documentación

- [Guía del proyecto (CLAUDE.md)](CLAUDE.md) — convenciones, arquitectura y reglas
- [Guía de la CLI (CLI_GUIDE.md)](CLI_GUIDE.md)
- [Manual de uso](docs/MANUAL_USO.md)
- [Guía de RAG](docs/RAG_GUIDE.md)
- [Documentación del modelo LSI](docs/modelo-lsi-documentacion.md)
- [Guía del indexer](docs/indexer-guide.md)
- [Guía de salida del crawler](docs/crawler-output-guide.md)
- [Arquitectura del proyecto](docs/arch/arquitectura-proyecto-sri.md)
- [Almacenamiento de dos niveles](docs/arch/almacenamiento-dos-niveles.md)
- [Informe LNCS](docs/INFORME%20LNCS/) — documentación formal en LaTeX/PDF

---

## Licencia

Ver [LICENSE](LICENSE).
