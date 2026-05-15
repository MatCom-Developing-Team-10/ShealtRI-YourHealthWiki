# ShealtRI Manual de Uso

## Índice

1. [Instalación Local](#instalación-local)
2. [Instalación con Docker](#instalación-con-docker)
3. [Gestión del Corpus](#gestión-del-corpus)
4. [Uso de la Interfaz de Consola](#uso-de-la-interfaz-de-consola)
5. [Ejemplos Prácticos](#ejemplos-prácticos)
6. [Solución de Problemas](#solución-de-problemas)
7. [Arquitectura del Proyecto](#arquitectura-del-proyecto)

---

## Instalación Local

### Requisitos Previos

- **Python 3.11 o superior**
- **pip** (gestor de paquetes de Python)
- **Git**
- Conexión a Internet (para descargar dependencias)

### Pasos de Instalación

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/MatCom-Developing-Team-10/ShealtRI-YourHealthWiki.git
cd ShealtRI-YourHealthWiki
```

#### 2. Crear Entorno Virtual

```bash
# En macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# En Windows
python -m venv .venv
.venv\Scripts\activate
```

#### 3. Instalar Dependencias

```bash
# Instalar dependencias de runtime (recomendado para Cortes 1 y 2)
pip install --timeout=300 --retries=20 -r requirements.txt

# O instalar con dependencias de desarrollo (si quieres correr tests)
pip install --timeout=300 --retries=20 -r requirements-dev.txt

# O instalar todo incluyendo UI (solo para Corte 3)
pip install --timeout=300 --retries=20 -r requirements-ui.txt
```

**Nota sobre el timeout:** Si tu conexión es lenta, los flags `--timeout=300 --retries=20` hacen que pip espere más tiempo y reinente las descargas.

#### 4. Descargar Modelo de spaCy

```bash
python -m spacy download es_core_news_md
```

Este modelo (~150 MB) es necesario para procesar texto en español. Solo se descarga una sola vez.

#### 5. Crear Carpeta de Datos (Opcional pero Recomendado)

```bash
mkdir -p data/raw
mkdir -p models
mkdir -p logs
```

---

## Instalación con Docker

### Requisitos Previos

- **Docker** (descarga desde https://www.docker.com/)
- **Docker Compose** (incluido con Docker Desktop en macOS y Windows)
- **Git**

### Pasos de Instalación

#### 1. Clonar el Repositorio

```bash
git clone https://github.com/MatCom-Developing-Team-10/ShealtRI-YourHealthWiki.git
cd ShealtRI-YourHealthWiki
```

#### 2. Construir e Iniciar el Contenedor

```bash
# Primera vez (construye la imagen)
docker-compose up --build -d

# Posteriores (usa la imagen construida)
docker-compose up -d
```

El flag `-d` significa "detached" (ejecuta en segundo plano).

#### 3. Verificar que el Contenedor Está Corriendo

```bash
docker-compose ps
```

Deberías ver algo como:
```
NAME        STATUS
sri-app     Up 2 minutes
```

#### 4. Detener el Contenedor

```bash
docker-compose down
```

---

## Gestión del Corpus

### Agregar Documentos Digitales

ShealtRI soporta múltiples formatos de documentos. Coloca tus archivos en la carpeta `data/raw/`:

```
data/raw/
├── libro1.pdf              # Libros en PDF (1 documento por página)
├── articulo.txt            # Archivos de texto
├── datos.csv               # Datos en CSV
├── documento.json          # JSON individual
├── documentos.jsonl        # JSONL (línea por documento, del crawler)
└── README.md               # Markdown
```

### Formatos Soportados

| Formato | Descripción | Comando |
|---------|-------------|---------|
| **PDF** | Libros, artículos médicos | `cp libro.pdf data/raw/` |
| **TXT** | Texto plano | `cp documento.txt data/raw/` |
| **JSON** | Documento individual o lista | `cp corpus.json data/raw/` |
| **JSONL** | Salida del crawler (1 doc/línea) | Copiado automático por crawler |
| **CSV** | Datos tabulares (médicos) | `cp datos.csv data/raw/` |
| **Markdown** | Documentación | `cp guia.md data/raw/` |

### Estructura de un Documento JSON

Si quieres crear documentos JSON manualmente:

```json
{
  "doc_id": "diabetes-001",
  "text": "La diabetes tipo 2 es una enfermedad metabólica crónica...",
  "url": "https://example.com/diabetes",
  "metadata": {
    "title": "Diabetes Tipo 2",
    "source": "MedlineNot",
    "date": "2024-01-15"
  }
}
```

O para múltiples documentos (lista):

```json
[
  {
    "doc_id": "doc1",
    "text": "...",
    "url": "...",
    "metadata": {}
  },
  {
    "doc_id": "doc2",
    "text": "...",
    "url": "...",
    "metadata": {}
  }
]
```

### Estructura de un Documento JSONL (Crawler Output)

Cada línea es un documento JSON completo:

```jsonl
{"doc_id": "doc1", "text": "contenido...", "url": "...", "metadata": {...}}
{"doc_id": "doc2", "text": "contenido...", "url": "...", "metadata": {...}}
{"doc_id": "doc3", "text": "contenido...", "url": "...", "metadata": {...}}
```

### Cargar Documentos

Una vez que tienes archivos en `data/raw/`, el sistema los carga automáticamente la próxima vez que ejecutes la interfaz de consola.

```bash
# Local
python cli.py

# Docker
docker-compose exec sri python cli.py
```

El sistema detectará el formato por la extensión del archivo y los cargará automáticamente.

---

## Uso de la Interfaz de Consola

### Modo Interactivo (Local)

```bash
source .venv/bin/activate
python cli.py
```

### Modo Interactivo (Docker)

```bash
docker-compose exec sri python cli.py
```

### Pantalla Inicial

```
[ShealtRI] Loading pipeline...
  loading NLP model (spaCy)... done
  reading documents from data/raw/... done
  source  : data/raw/ (42 docs)
  indexing... done
  fitting LSI (n_components=100)... done

╔══════════════════════════════════════════════════╗
║         ShealtRI — Medical Information SRI       ║
║   Type a query, 'stats', 'help', or 'quit'       ║
╚══════════════════════════════════════════════════╝

Query> 
```

### Comandos Disponibles

#### Búsqueda (Query)

Escribe términos médicos en español:

```
Query> síntomas de diabetes
```

**Resultado esperado:**
```
Results (3 found):

1. [0.892] Diabetes Tipo 2
    https://medlineplus.gov/diabetes
    La diabetes tipo 2 es una enfermedad metabólica crónica...

2. [0.756] Insulina y Glucosa
    https://mayoclinic.org/insulin
    La insulina es una hormona producida por el páncreas...

Query> 
```

Los números entre `[ ]` son scores de similitud (0.0 a 1.0). Mayor score = mayor relevancia.

#### Estadísticas del Corpus

```
Query> stats
```

**Resultado esperado:**
```
  n_documents              : 42
  n_terms                  : 1258
  total_tokens             : 2894
  avg_tokens_per_doc       : 68.90
  avg_postings_per_term    : 2.30
```

Explicación:
- **n_documents**: Número total de documentos/páginas en el corpus
- **n_terms**: Número de términos únicos después de procesamiento
- **total_tokens**: Número total de palabras (tokens)
- **avg_tokens_per_doc**: Promedio de palabras por documento
- **avg_postings_per_term**: En cuántos documentos aparece en promedio cada término

#### Ayuda

```
Query> help
```

Muestra todos los comandos disponibles.

#### Salir

```
Query> quit
```

O presiona `Ctrl+C` para interrumpir.

### One-Shot Queries (sin modo interactivo)

#### Local

```bash
python cli.py --query "hipertensión arterial"
python cli.py --query "asma bronquial" --top-k 3
python cli.py --stats
```

#### Docker

```bash
docker-compose exec sri python cli.py --query "síntomas de presión alta"
docker-compose exec sri python cli.py --top-k 10 --query "diabetes"
docker-compose exec sri python cli.py --stats
```

### Flags Opcionales

```bash
--query "texto"        # Query única, sin modo interactivo
--stats                # Solo mostrar estadísticas
--top-k N              # Número de resultados (defecto: 5)
```

---

## Ejemplos Prácticos

### Ejemplo 1: Instalación Completa Local

```bash
# 1. Clonar
git clone https://github.com/MatCom-Developing-Team-10/ShealtRI-YourHealthWiki.git
cd ShealtRI-YourHealthWiki

# 2. Entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar deps
pip install --timeout=300 --retries=20 -r requirements.txt
python -m spacy download es_core_news_md

# 4. Agregar PDFs
mkdir -p data/raw
cp ~/Downloads/libros_medicos/*.pdf data/raw/

# 5. Correr
python cli.py
```

### Ejemplo 2: Uso con Docker

```bash
# 1. Clonar
git clone https://github.com/MatCom-Developing-Team-10/ShealtRI-YourHealthWiki.git
cd ShealtRI-YourHealthWiki

# 2. Agregar PDFs a la carpeta local
mkdir -p data/raw
cp ~/Downloads/diabetes.pdf data/raw/
cp ~/Downloads/cardiologia.pdf data/raw/

# 3. Construir e iniciar
docker-compose up --build -d

# 4. Esperar a que compile (1-2 minutos)
docker-compose logs -f sri

# 5. Una vez listo, correr CLI
docker-compose exec sri python cli.py
```

### Ejemplo 3: Query Rápida Local

```bash
python cli.py --query "¿cuáles son los síntomas de la diabetes?" --top-k 5
```

### Ejemplo 4: Query Rápida en Docker

```bash
docker-compose exec sri python cli.py --query "tratamiento de la hipertensión"
```

### Ejemplo 5: Correr Tests

```bash
# Local - smoke tests (stdlib, sin spaCy)
python tests/smoke_test.py

# Local - tests de integración (pipeline real)
python -m pytest tests/integration/test_pipeline.py -v -s

# Docker - integration tests
docker-compose exec sri python -m pytest tests/integration/test_pipeline.py -v -s
```

---

## Solución de Problemas

### Problema: "ModuleNotFoundError: No module named 'spacy'"

**Causa:** Las dependencias no están instaladas o no activaste el venv.

**Solución:**

```bash
# Verifica que estés en el venv
which python  # debe mostrar .../ShealtRI-YourHealthWiki/.venv/bin/python

# Si no estás en el venv
source .venv/bin/activate

# Reinstala
pip install -r requirements.txt
```

### Problema: "spaCy model 'es_core_news_md' not found"

**Causa:** No descargaste el modelo de spaCy.

**Solución:**

```bash
# Local
python -m spacy download es_core_news_md

# Docker
docker-compose exec sri python -m spacy download es_core_news_md
```

### Problema: "No results found for: 'query'"

**Causa:** 
- El corpus está vacío (sin PDFs en `data/raw/`)
- La query no coincide con ningún documento
- Los documentos no se cargaron correctamente

**Solución:**

```bash
# Verifica que hay archivos
ls -la data/raw/

# Comprueba estadísticas
python cli.py --stats

# Intenta una query simple
python cli.py --query "diabetes"
```

### Problema: Docker "connection refused" o "exit code 137"

**Causa:** 
- El contenedor se quedó sin memoria
- El build no finalizó correctamente

**Solución:**

```bash
# Detén y limpia
docker-compose down

# Reconstruye sin caché
docker-compose up --build -d

# Verifica logs
docker-compose logs -f sri
```

### Problema: "pip._vendor.urllib3.exceptions.ReadTimeoutError"

**Causa:** Conexión lenta al descargar dependencias.

**Solución:**

```bash
# Aumenta el timeout aún más
pip install --timeout=600 --retries=20 -r requirements.txt

# O instala paquetes individuales
pip install numpy scipy scikit-learn
```

### Problema: PDF no se carga correctamente

**Causa:** El PDF está corrupto o tiene formato no estándar.

**Solución:**

```bash
# Verifica que el PDF sea válido
file data/raw/libro.pdf  # debe mostrar "PDF document"

# Intenta extraer texto manualmente
pdftotext data/raw/libro.pdf -  | head -20

# Si no funciona, convierte a TXT primero
```

---

## Arquitectura del Proyecto

### Flujo de Datos

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT: Query                         │
│                   "diabetes síntomas"                   │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  TextProcessor       │ (spaCy)
        │  - Lematización      │
        │  - Stopwords         │
        │  - Spelling check    │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  IndexerService      │
        │  - TF-IDF            │
        │  - Build Query Corpus│
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  LSIRetriever        │
        │  - TruncatedSVD      │
        │  - Cosine Similarity │
        │  - Top-K Ranking     │
        └──────────┬───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  OUTPUT: Results     │
        │  [Doc1: 0.89]        │
        │  [Doc2: 0.72]        │
        └──────────────────────┘
```

### Componentes Principales

```
src/
├── core/                           # Núcleo del sistema
│   ├── interfaces.py               # ABCs (BaseRetriever, DocumentStore, etc.)
│   ├── models.py                   # Modelos de datos (Document, Query, etc.)
│   └── pipeline.py                 # Orquestador (RetrievalContext)
│
├── modules/                        # Módulos obligatorios del pipeline
│   ├── text_processor/             # Procesa texto (spaCy, stopwords, Trie)
│   ├── indexer/                    # Construye índices (TF-IDF, vocabulario)
│   ├── retriever/                  # Modelo LSI (TruncatedSVD, cosine)
│   ├── document_loader/            # Carga múltiples formatos
│   └── ranker/                     # Ranking y ordenamiento (Corte 2)
│
├── infra/                          # Infraestructura de datos
│   ├── chroma_repository.py        # ChromaDB vector store
│   └── storage.py                  # Almacenamiento persistente
│
└── ui/                             # Interfaz de usuario
    ├── cli.py                      # Consola (Cortes 1-2)
    └── app.py                      # Streamlit web (Corte 3)
```

### Almacenamiento Persistente

```
data/
├── raw/                 # Corpus de entrada (PDFs, TXTs, JSONs)
├── chroma/              # ChromaDB (embeddings y búsqueda vectorial)
└── documents/           # FileSystemDocumentStore (metadata)

models/
└── lsi/                 # Modelos serializados (joblib)
    ├── tfidf.joblib
    ├── svd_model.joblib
    └── vocabulary.joblib
```

Estos directorios son **persistentes**: se guardan en disco y se reutilizan en ejecuciones posteriores. Si borras alguno, se reconstruye automáticamente la próxima vez que ejecutes.

---

## Configuración Avanzada

### Aumentar Precisión del Modelo LSI

En `cli.py`, modifica `n_components`:

```python
self.retriever = LSIRetriever(
    ...
    n_components=150,  # Aumenta de 100 a 150 para corpus grandes
    ...
)
```

Más componentes = más precisión pero más lento.

### Cambiar Threshold de Similitud

En `cli.py`:

```python
self.retriever = LSIRetriever(
    ...
    similarity_threshold=0.5,  # Solo resultados con score >= 0.5
    ...
)
```

### Limitar Recursos en Docker

En `docker-compose.yml`:

```yaml
services:
  sri:
    ...
    mem_limit: 2g         # Máximo 2 GB de RAM
    cpus: '1.5'           # Máximo 1.5 CPUs
```

---

## Próximas Versiones

- **Corte 2:** RAG (Retrieval-Augmented Generation) con respuestas LLM
- **Corte 3:** Interfaz web Streamlit con visualizaciones
- Plugins opcionales: expansión de queries, multimodalidad

---

**Última actualización:** Mayo 2026
**Versión:** Cortes 1-2
**Soporte:** revelianny10@gmail.com
