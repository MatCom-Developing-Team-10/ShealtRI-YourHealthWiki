# Arquitectura de Almacenamiento de Dos Niveles

## Visión general

El sistema de recuperación LSI implementa una arquitectura de almacenamiento separada en dos capas para optimizar rendimiento, escalabilidad y mantenibilidad.

```
┌─────────────────────────────────────────────────────────────────┐
│                         RETRIEVER                                │
│                                                                  │
│   Query ──► Vector Store ──► IDs + Scores ──► Document Store    │
│                   │                                │             │
│                   ▼                                ▼             │
│            ChromaDB/FAISS                   FileSystem/JSON      │
│         (embeddings + IDs)              (contenido completo)     │
└─────────────────────────────────────────────────────────────────┘
```

## Nivel 1: Vector Store (ChromaDB)

Almacena únicamente la información necesaria para la búsqueda vectorial.

### Contenido almacenado

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `doc_id` | string | Identificador único del documento |
| `embedding` | float[] | Vector LSI de dimensión k |
| `url` | string | URL de origen (metadata mínima) |

### Características

- **Optimizado para búsqueda**: Índices especializados para similitud coseno
- **Ligero**: No almacena texto completo, solo vectores
- **Rápido**: Búsquedas en milisegundos incluso con miles de documentos

### Ejemplo de uso

```python
# Almacenar documento en el vector store
collection.add(
    ids=["doc_001"],
    embeddings=[doc_vector.tolist()],  # Vector LSI de dimensión k
    metadatas=[{"url": "https://example.com/articulo"}]
)

# Buscar documentos similares
results = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=10
)
# Retorna: ids=["doc_042", "doc_108", ...], distances=[0.12, 0.18, ...]
```

## Nivel 2: Document Store (FileSystem/JSON)

Almacena el contenido completo y metadata rica de cada documento.

### Contenido almacenado

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `doc_id` | string | Identificador único (mismo que Vector Store) |
| `title` | string | Título del documento |
| `content` | string | Texto completo del documento |
| `url` | string | URL de origen |
| `source` | string | Fuente del documento (ej: "medlineplus", "who") |
| `crawled_at` | datetime | Fecha de crawling |
| `metadata` | dict | Metadata adicional específica del dominio |

### Estructura de archivos

```
data/
└── documents/
    ├── index.json           # Índice maestro de documentos
    ├── doc_001.json
    ├── doc_002.json
    └── ...
```

### Formato de documento

```json
{
    "doc_id": "doc_001",
    "title": "Hipertensión arterial: síntomas y tratamiento",
    "content": "La hipertensión arterial es una condición...",
    "url": "https://medlineplus.gov/spanish/highbloodpressure.html",
    "source": "medlineplus",
    "crawled_at": "2024-03-15T10:30:00Z",
    "metadata": {
        "category": "cardiovascular",
        "language": "es",
        "word_count": 1250
    }
}
```

## Flujo de recuperación en dos fases

### Fase 1: Búsqueda vectorial

```python
def search_vectors(query_vector: np.ndarray, top_n: int = 10) -> list[tuple[str, float]]:
    """Busca en el Vector Store y retorna IDs con scores."""
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=top_n
    )
    return list(zip(results["ids"][0], results["distances"][0]))
```

### Fase 2: Fetch de documentos

```python
def fetch_documents(doc_ids: list[str]) -> list[Document]:
    """Recupera documentos completos del Document Store."""
    documents = []
    for doc_id in doc_ids:
        doc_path = f"data/documents/{doc_id}.json"
        with open(doc_path) as f:
            doc_data = json.load(f)
            documents.append(Document(**doc_data))
    return documents
```

### Coordinación en el Retriever

```python
class LSIRetriever(BaseRetriever):
    def __init__(self, vector_store: VectorStore, document_store: DocumentStore):
        self.vector_store = vector_store
        self.document_store = document_store

    def retrieve(self, query: Query) -> list[Document]:
        # Fase 1: Búsqueda vectorial
        query_vector = self._project_query(query.text)
        results = self.vector_store.search(query_vector, top_n=10)

        # Fase 2: Fetch de contenido
        doc_ids = [doc_id for doc_id, score in results]
        documents = self.document_store.fetch(doc_ids)

        # Asignar scores a documentos
        scores = {doc_id: score for doc_id, score in results}
        for doc in documents:
            doc.score = scores[doc.doc_id]

        return sorted(documents, key=lambda d: d.score, reverse=True)
```

## Beneficios de la separación

### 1. Escalabilidad

ChromaDB no se infla con texto completo. Un vector de 100 dimensiones ocupa ~400 bytes, mientras que un documento médico promedio puede ocupar 10-50 KB.

| Documentos | Solo vectores (k=100) | Con texto completo |
|------------|----------------------|-------------------|
| 1,000 | ~400 KB | ~25 MB |
| 10,000 | ~4 MB | ~250 MB |
| 100,000 | ~40 MB | ~2.5 GB |

### 2. Flexibilidad

- Actualizar el contenido de un documento sin recalcular su vector
- Agregar metadata sin afectar el índice vectorial
- Migrar el Document Store sin tocar ChromaDB

### 3. Rendimiento

- Vector Store: optimizado para operaciones de similitud
- Document Store: optimizado para lectura secuencial de JSON

### 4. Mantenibilidad

- Cada store tiene una responsabilidad clara
- Fácil debugging: verificar vectores y documentos por separado
- Testing aislado de cada capa

## Interfaces en el código

```python
# infra/storage.py

class VectorStore(ABC):
    """Interfaz para el almacenamiento de vectores."""

    @abstractmethod
    def add(self, doc_id: str, embedding: np.ndarray, metadata: dict) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_n: int) -> list[tuple[str, float]]: ...

    @abstractmethod
    def delete(self, doc_id: str) -> None: ...


class DocumentStore(ABC):
    """Interfaz para el almacenamiento de documentos completos."""

    @abstractmethod
    def save(self, document: Document) -> None: ...

    @abstractmethod
    def fetch(self, doc_ids: list[str]) -> list[Document]: ...

    @abstractmethod
    def delete(self, doc_id: str) -> None: ...
```

## Sincronización entre stores

Es crítico mantener consistencia entre ambos stores:

```python
class StorageCoordinator:
    """Coordina operaciones entre Vector Store y Document Store."""

    def __init__(self, vector_store: VectorStore, document_store: DocumentStore):
        self.vector_store = vector_store
        self.document_store = document_store

    def index_document(self, document: Document, embedding: np.ndarray) -> None:
        """Indexa un documento en ambos stores de forma atómica."""
        try:
            self.document_store.save(document)
            self.vector_store.add(
                doc_id=document.doc_id,
                embedding=embedding,
                metadata={"url": document.url}
            )
        except Exception as e:
            # Rollback: eliminar de ambos stores si falla
            self.document_store.delete(document.doc_id)
            self.vector_store.delete(document.doc_id)
            raise e

    def delete_document(self, doc_id: str) -> None:
        """Elimina un documento de ambos stores."""
        self.vector_store.delete(doc_id)
        self.document_store.delete(doc_id)
```
