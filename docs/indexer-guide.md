# Guía técnica del módulo Indexer

> Documento vivo. Describe el estado, las decisiones de diseño y la integración del módulo `modules/indexer/` con el resto del sistema SRI.

## 1. Contexto y alcance

El **indexer** es el componente que transforma una colección de documentos (producida por el crawler y materializada en disco por el `DocumentLoader`) en una estructura optimizada para búsqueda: el `IndexedCorpus`. Esa estructura es el **adapter** que desacopla la zona de texto (`TextProcessor`) de la zona algebraica (`TfidfProcessor` + `LSIModel`).

Antes de esta iteración el módulo tenía:

- `FileSystemDocumentStore` — ya completo, persiste el **texto completo** de los documentos.
- `IndexerService.build()` — un stub que lanzaba `NotImplementedError`.
- Sin `build_query()`, sin `update()`, sin `remove()`.
- Sin capa de persistencia para los resultados del indexer.
- Conflicto de merge sin resolver en `__init__.py`.

Tras esta iteración el módulo ofrece:

| Capacidad | Ubicación |
|---|---|
| Construcción de corpus desde documentos | `IndexerService.build()` |
| Construcción de corpus de query (con corrección ortográfica) | `IndexerService.build_query()` |
| Actualización incremental idempotente | `IndexerService.update()` |
| Borrado con remapeo de `doc_idx` | `IndexerService.remove()` |
| Estadísticas diagnósticas | `IndexerService.stats()` |
| Persistencia en disco del `IndexedCorpus` | `IndexStore.save()` / `IndexStore.load()` |
| Persistencia del vocabulario del spell checker | `IndexStore.save_spell_vocabulary()` / `load_spell_vocabulary()` |
| Gestión (manifest, doc_ids, clear) | `IndexStore.manifest()`, `indexed_doc_ids()`, `clear()` |

## 2. Cambios implementados, archivo por archivo

### 2.1 `modules/indexer/__init__.py`

**Antes**: contenía marcadores literales `<<<<<<< HEAD`, `=======`, `>>>>>>>` que impedían siquiera importar el paquete.

**Después**: resuelto conservando la versión más completa (exporta `IndexerService`, `IndexerConfig`, `FileSystemDocumentStore` y sus excepciones). Se añadió también la API nueva de persistencia: `IndexStore` y `IndexStoreError`.

### 2.2 `modules/indexer/service.py` — implementación de `IndexerService`

Método `build(documents) -> IndexedCorpus`:

1. Procesa cada doc con `TextProcessor.process(text, is_query=False)`. Efecto lateral clave: los tokens resultantes se añaden al Trie del spell checker.
2. Tokeniza por `str.split()` (la salida del `TextProcessor` es ya una cadena espacio-separada de tokens procesados).
3. Descarta documentos con menos de `min_document_length` tokens (log a nivel DEBUG).
4. Computa `Counter` por documento con frecuencias de término.
5. Agrega en un `defaultdict(list)` la estructura `inverted_index[term] = [(doc_idx, tf), ...]`.
6. Aplica `min_term_frequency` sobre la frecuencia total del corpus (no sobre la `document frequency`) para filtrar ruido y typos.
7. Devuelve un `IndexedCorpus` con vocabulario **ordenado** (determinismo de columnas en TF-IDF).
8. Emite log INFO cada `log_progress_every` documentos.

Método `build_query(query_text) -> IndexedCorpus`:

1. Procesa con `is_query=True` → corrige tokens contra el vocabulario ya acumulado.
2. Cuenta tf del único "doc" (la query) con `Counter`.
3. Construye `inverted_index[term] = [(0, tf)]`, que es exactamente lo que `TfidfProcessor.transform` espera al leer `postings[0][1]`.
4. Empaqueta en un `IndexedCorpus` con un `Document` sintético (`doc_id="__query__"`) para satisfacer el invariante `len(documents) == len(processed_texts)`.

Método `update(existing, new_documents) -> IndexedCorpus`:

1. Calcula `existing_ids = {doc.doc_id for doc in existing.documents}` para idempotencia.
2. Copia las postings existentes como listas mutables.
3. Para cada nuevo documento:
   - Si su `doc_id` ya existe, se salta (duplicate skip).
   - Si tras procesar queda por debajo de `min_document_length`, se salta (short skip).
   - En otro caso se asigna `new_doc_idx = len(kept_documents)` y se añaden postings.
4. Devuelve un nuevo `IndexedCorpus`. **No muta el argumento.**
5. Log final con contadores (`+added, duplicates, short, total, vocab`).

Método `remove(existing, doc_ids) -> IndexedCorpus`:

1. Construye el set de IDs a quitar.
2. Recorre `existing.documents` una sola vez generando un mapa `old_idx → new_idx` para los supervivientes; así los índices quedan **contiguos** tras el borrado.
3. Reescribe `inverted_index` reemplazando los `doc_idx` viejos por los nuevos.
4. Los términos cuyas postings quedan vacías desaparecen del índice y del vocabulario.

Método estático `stats(corpus) -> dict`:

Devuelve un diccionario con `n_documents`, `n_terms`, `total_tokens`, `avg_tokens_per_doc`, `avg_postings_per_term`. Cheap — seguro llamar en cada update para logging/monitoring.

### 2.3 `modules/indexer/index_store.py` (nuevo) — capa de persistencia

Clase `IndexStore` con layout en disco:

```
data/indexer/
├── corpus.joblib     ← IndexedCorpus completo
├── spell_vocab.txt   ← un token por línea (reconstruye el Trie)
├── doc_ids.txt       ← doc_ids indexados (lookup rápido)
└── manifest.json     ← schema_version, created_at, updated_at, counts
```

API pública:

| Método | Rol |
|---|---|
| `save(corpus)` | Escribe corpus + doc_ids + manifest (atómico). |
| `load() -> IndexedCorpus` | Lee el último snapshot. |
| `save_spell_vocabulary(checker)` | Persiste el Trie (como lista plana ordenada). |
| `load_spell_vocabulary(checker) -> int` | Reinyecta palabras en un `TrieSpellChecker`. |
| `manifest() -> dict` | Metadatos; `{}` si no existe. |
| `indexed_doc_ids() -> set[str]` | Consulta rápida sin cargar joblib. |
| `exists() -> bool` | ¿Hay corpus guardado? |
| `clear()` | Borra todos los artefactos. |

Escritura atómica: todo pasa por `.tmp` + `os.replace()` para evitar archivos corruptos si el proceso muere a mitad de escritura.

### 2.4 `modules/text_processor/spell_checker.py` — `words()` público

Se añadió un método público `words() -> list[str]` y su helper `_collect_words()`. Hace un DFS del Trie y devuelve todas las palabras completas. Es la forma que `IndexStore` usa para serializar el vocabulario sin conocer la estructura interna del Trie.

## 3. Decisiones de diseño y justificación

### 3.1 `IndexerService` permanece *stateless*

**Decisión**: El servicio no tiene `self._corpus`, no guarda estado entre llamadas. Todos los métodos que modifican el corpus (`build`, `update`, `remove`) devuelven un `IndexedCorpus` nuevo.

**Justificación**:
- **Testeabilidad**: un método puro con entrada → salida se prueba sin fixtures elaborados ni mocks.
- **Previsibilidad**: no hay side-effects ocultos. El único efecto lateral legítimo es el que hace `TextProcessor` sobre el spell checker, y es transparente (declarado en la interfaz `is_query`).
- **Separación de responsabilidades**: el estado en disco vive en `IndexStore`. El estado in-memory vive en quien lo invoca. El servicio orquesta, no posee.

**Alternativa descartada**: un `IndexerService` stateful que mantuviera `self._corpus` y expusiera `save()`/`load()` propios. Habría mezclado computación con I/O y habría exigido saber el `storage_dir` en el constructor, complicando el testing.

### 3.2 Persistencia en un módulo separado (`IndexStore`)

**Decisión**: crear `IndexStore` en `modules/indexer/index_store.py` en lugar de añadir métodos `save`/`load` a `IndexerService` o a `IndexedCorpus`.

**Justificación**:
- `IndexedCorpus` vive en `core/interfaces.py` y es un **contrato de dominio**. No debe saber de discos, rutas ni serialización.
- Replica el patrón ya establecido del proyecto: `DocumentStore` (abstracto) + `FileSystemDocumentStore` (implementación), o `BaseRepository` + `ChromaRepository`. Consistencia arquitectónica.
- Habilita múltiples backends a futuro (p. ej. un `S3IndexStore` o `SQLiteIndexStore`) sin tocar el servicio.

### 3.3 Escritura atómica por todas partes

**Decisión**: cada `save` escribe a `archivo.ext.tmp` y hace `Path.replace()` al final.

**Justificación**:
- `os.replace` (que `Path.replace` usa internamente) es atómico en el mismo sistema de archivos en POSIX y en Windows.
- Evita el caso en el que el proceso muere dejando `corpus.joblib` a medio escribir y el siguiente arranque no puede leerlo.
- Limpia el `.tmp` si algo falla, previniendo basura huérfana.

### 3.4 `spell_vocab.txt` y `doc_ids.txt` como texto plano

**Decisión**: dos artefactos texto, uno por línea, en lugar de joblib/pickle.

**Justificación**:
- **Inspección**: `head spell_vocab.txt` es suficiente para auditar qué aprendió el Trie. Con pickle haría falta un script.
- **`git diff` útil**: si se versiona el índice para reproducibilidad académica, los diffs son legibles.
- **Sencillez**: no dependen de compatibilidad entre versiones de Python ni de pickling. `.txt` es estable.
- **Tamaño**: ~40 bytes por doc_id × N. Un corpus de 100k docs son 4 MB — despreciable.

### 3.5 `doc_ids.txt` como sidecar

**Decisión**: duplicar los doc_ids en un archivo aparte a pesar de que ya están dentro de `corpus.joblib`.

**Justificación**:
- `indexed_doc_ids()` se usa antes de llamar a `update()` para filtrar duplicados. Llamar `load()` solo para preguntar "¿ya indexé esto?" es caro (descomprimir joblib, deserializar dataclasses completos).
- El sidecar es minúsculo (texto) y se reescribe siempre junto al corpus, así que nunca se desincroniza en un flujo normal.
- Hay fallback: si el sidecar no existe (corpus guardado por una versión antigua), `indexed_doc_ids()` carga el corpus como degradación amable.

### 3.6 `update()` no reaplica `min_term_frequency`

**Decisión**: el filtro de frecuencia mínima se aplica **solo** en `build()`, nunca en `update()`.

**Justificación**:
- Un término con `total_freq=3` que sobrevivió a `build()` con threshold `2` no debería desaparecer al añadir más documentos que lo mencionen (o que no).
- Si se reaplicase, la composición del vocabulario dependería del **orden** en que llegan los docs, no solo del contenido final. Rompería la reproducibilidad.
- Si se requiere compactar agresivamente, la opción explícita es llamar `indexer.build(list(corpus.documents) + new_documents)` — reprocesa todo.

### 3.7 `remove()` renumera `doc_idx` de forma contigua

**Decisión**: tras borrar docs, los supervivientes se renumeran `0..n-1` sin huecos.

**Justificación**:
- `TfidfProcessor.fit(corpus)` usa `(doc_idx, tf)` para construir la `csr_matrix` con shape `(n_docs, n_terms)`. Un hueco (p. ej. `doc_idx=5` cuando ya solo hay 4 docs) produciría una fila fantasma vacía o un out-of-bounds.
- Reasignar es barato (un pase por los documentos + un pase por las postings) y mantiene el invariante que TF-IDF espera.

### 3.8 `IndexedCorpus` sigue intacto

**Decisión**: no se modifica el contrato en `core/interfaces.py`. Ni se le añaden métodos `save/load`, ni campos nuevos, ni versionado.

**Justificación**:
- Minimiza el blast radius: retriever, text_processor y tests no tienen que cambiar.
- El `schema_version` vive en `manifest.json` (donde debe estar el versionado de un formato de persistencia), no en el objeto de dominio.

### 3.9 `stats()` como método estático

**Decisión**: `IndexerService.stats(corpus)` no requiere `self`.

**Justificación**:
- No usa `text_processor` ni `config`. Recibe el corpus y lo describe.
- Permite `IndexerService.stats(store.load())` sin instanciar un servicio.
- Expresa que la función es un utility, no una operación propia del servicio.

### 3.10 `TF-IDF` y `SVD` no se actualizan incrementalmente

**Decisión**: tras `update()`/`remove()` hay que llamar `LSIRetriever.fit(corpus)` otra vez, regenerando la matriz TF-IDF y el SVD desde cero.

**Justificación**:
- `sklearn.decomposition.TruncatedSVD` no soporta `partial_fit`. Hacer SVD incremental implicaría cambiar de algoritmo (`IncrementalPCA`, o `scipy.sparse.linalg.svds` con restarts) y alterar la semántica del espacio latente (no es estable entre refits incrementales).
- Para un corpus académico (decenas de miles de docs) el refit completo toma segundos. No justifica la complejidad.
- Decisión consciente de **dejar explícito** el refit — el usuario sabe cuándo paga el costo.

## 4. Integración con el resto del pipeline

### 4.1 Con el crawler y los scrapers (`modules/crawler/`)

- **No hay import cruzado.** El único acople es semántico: el indexer espera `Document` (de `core.models`), que es lo que el crawler produce.
- El crawler escribe en `data/raw/<source>.jsonl` vía `RawDocumentStorage`. El indexer no lee esos archivos directamente — lo hace `DocumentLoader`, que es su "adapter de entrada".
- La estabilidad del `doc_id` (UUID5 derivado de la URL en `GenericCrawler._generate_doc_id`) es la que hace viable el `update()` idempotente: mismo URL → mismo ID → el indexer lo reconoce como duplicado.

### 4.2 Con el `DocumentLoader` (`modules/document_loader/`)

- `DocumentLoader.load_from_directory("data/raw/")` produce la `list[Document]` que alimenta `build()` o `update()`.
- Validación mínima (`doc_id`, `text`, `url` requeridos) se hace ahí. El indexer confía.

### 4.3 Con el `TextProcessor` (`modules/text_processor/`)

- Dependencia inyectada en el constructor de `IndexerService`. El servicio no crea un `TextProcessor` propio — respeta la inversión de dependencias.
- Contrato dual:
  - `process(text, is_query=False)` en `build()`/`update()` → puebla el Trie.
  - `process(text, is_query=True)` en `build_query()` → corrige contra el Trie.
- La **misma instancia** de `TextProcessor` debe usarse en todo el ciclo de vida para que el Trie esté disponible cuando llegue la query. Si se reinicia el proceso, `IndexStore.load_spell_vocabulary()` restaura el Trie en una nueva instancia.

### 4.4 Con el retriever LSI (`modules/retriever/`)

- `LSIRetriever.fit(corpus)` consume `IndexedCorpus` tal cual sale del indexer (`build` o `update`).
- `LSIRetriever.retrieve(Query(..., indexed_corpus=q_corpus))` consume el `IndexedCorpus` de una query producido por `build_query()`.
- `TfidfProcessor.transform(query_corpus)` lee solo `query_corpus.inverted_index` y descarta términos OOV; por eso `build_query` no necesita filtrar el vocabulario, `tfidf_processor` lo hace.

### 4.5 Con `FileSystemDocumentStore` (dentro del mismo módulo)

- Son **complementarios**, no redundantes:
  - `FileSystemDocumentStore` guarda el **texto completo** + metadata de cada documento, direccionable por `doc_id`. Lo puebla `LSIRetriever.fit` vía `repository.add_documents`.
  - `IndexStore` guarda la **estructura de índice** (postings, vocabulario, doc_ids resumidos, Trie).
- Una query consulta **ambos**: `LSIRetriever.retrieve` hace similarity search contra `ChromaRepository` (vectores), obtiene IDs, y luego `FileSystemDocumentStore.get_by_ids` carga el texto completo.

### 4.6 Con `ChromaRepository` (`infra/chroma_repository.py`)

- No hay interacción directa. Chroma se puebla desde `LSIRetriever.fit(corpus)` con los embeddings que produce `LSIModel.fit(tfidf_matrix)` donde `tfidf_matrix` viene del corpus indexado.
- Tras `update()`/`remove()`, hay que reejecutar `fit` para sincronizar Chroma con la nueva realidad del índice.

## 5. Flujo end-to-end recomendado

### 5.1 Arranque en frío (primera indexación)

```python
from core.models import Query
from modules.document_loader import DocumentLoader
from modules.text_processor import TextProcessor
from modules.indexer import IndexerService, IndexStore
from modules.retriever import LSIRetriever
from modules.indexer import FileSystemDocumentStore
from infra.chroma_repository import ChromaRepository

loader = DocumentLoader()
processor = TextProcessor()
indexer = IndexerService(text_processor=processor)
store = IndexStore("data/indexer")

doc_store = FileSystemDocumentStore("data/documents")
vector_repo = ChromaRepository("data/chroma")
retriever = LSIRetriever(repository=vector_repo, document_store=doc_store)

documents = loader.load_from_directory("data/raw/")

corpus = indexer.build(documents)
print(IndexerService.stats(corpus))

store.save(corpus)
store.save_spell_vocabulary(processor.spell_checker)

retriever.fit(corpus)
retriever.save("models/lsi")
```

### 5.2 Arranque en caliente + actualización incremental

```python
loader = DocumentLoader()
processor = TextProcessor()
indexer = IndexerService(text_processor=processor)
store = IndexStore("data/indexer")

if store.exists():
    corpus = store.load()
    store.load_spell_vocabulary(processor.spell_checker)
    already = store.indexed_doc_ids()

    all_docs = loader.load_from_directory("data/raw/")
    new_docs = [d for d in all_docs if d.doc_id not in already]

    if new_docs:
        corpus = indexer.update(corpus, new_docs)
        store.save(corpus)
        store.save_spell_vocabulary(processor.spell_checker)

        retriever = LSIRetriever.load(vector_repo, doc_store, "models/lsi")
        retriever.fit(corpus)
        retriever.save("models/lsi")
else:
    corpus = indexer.build(loader.load_from_directory("data/raw/"))
    store.save(corpus)
```

### 5.3 Consulta

```python
processor = TextProcessor()
store = IndexStore("data/indexer")
store.load_spell_vocabulary(processor.spell_checker)

indexer = IndexerService(text_processor=processor)
retriever = LSIRetriever.load(vector_repo, doc_store, "models/lsi")

q_corpus = indexer.build_query("hipertensoin arterail")
query = Query(text="hipertensoin arterail", indexed_corpus=q_corpus)

results = retriever.retrieve(query, top_k=5)
for r in results:
    print(f"{r.score:.3f}  {r.document.metadata.get('title', r.document.doc_id)}")
```

### 5.4 Borrado de documentos

```python
corpus = store.load()
corpus = indexer.remove(corpus, ["uuid-de-doc-1", "uuid-de-doc-2"])
store.save(corpus)

retriever.fit(corpus)
retriever.save("models/lsi")
```

### 5.5 Reset total

```python
store.clear()
```

## 6. Casos borde y comportamiento

| Caso | Comportamiento |
|---|---|
| `build([])` | Devuelve `IndexedCorpus` vacío válido (no lanza). |
| `build_query("")` | Igual — corpus vacío con 1 doc sintético y `inverted_index={}`. |
| `update(corpus, [])` | Devuelve el `corpus` sin cambios (early return). |
| `update(corpus, [dup])` | `dup` ya existe por `doc_id` → saltado, log de contador `duplicates`. |
| `remove(corpus, ["unknown"])` | Silenciosamente ignora IDs desconocidos; log el delta real. |
| `store.load()` sin corpus | Lanza `IndexStoreError` explícita. |
| `store.manifest()` sin manifest | Devuelve `{}`. |
| `store.indexed_doc_ids()` sin nada | Devuelve `set()`. |
| `load_spell_vocabulary` sin archivo | WARNING y devuelve `0` — no lanza. |
| Documento con 0 tokens tras procesar | Saltado si `min_document_length ≥ 1` (default). |
| Query con todos términos OOV | `inverted_index` vacío → `TfidfProcessor.transform` produce vector nulo → retriever devuelve top-k normales (sin filtrar) o lista vacía si el threshold corta. |

## 7. Limitaciones conocidas

1. **El retriever no es incremental.** `update`/`remove` sobre el corpus no afectan a TF-IDF ni a SVD hasta que se vuelva a llamar `retriever.fit(corpus)` + `retriever.save()`.
2. **No hay bloqueo multi-proceso.** Si dos procesos corren `save()` concurrentemente, la atomicidad por archivo garantiza que cada archivo queda coherente, pero la consistencia **entre** archivos (`corpus.joblib` vs `manifest.json`) depende de quién escribe último. Para uso académico single-process es irrelevante.
3. **El Trie no se contrae.** `remove()` quita docs pero no limpia palabras del spell checker. Si una palabra solo aparecía en los docs borrados, sigue en el Trie. Impacto: `build_query` puede "corregir" una query hacia un término ya no indexado, que `TfidfProcessor.transform` descartará por OOV de todas formas. No afecta correctitud, solo eficiencia marginal.
4. **Idioma único por `TextProcessor`.** El `TextProcessor` usa `es_core_news_md`. Docs en inglés (NHS, parte de Mayo Clinic) se lematizarán mal. No es un límite del indexer sino del pipeline de preprocesado — se documenta aquí porque el indexer es quien lo arrastra.
5. **`min_term_frequency` no retroactiva.** Diseño explícito, no bug. Si se quiere aplicar a un corpus extendido, hay que reconstruir con `build`.
6. **Sin compactación automática.** Tras muchos `update` seguidos, las postings lists pueden crecer desordenadas. No hay impacto correctivo (el orden de postings no importa para TF-IDF), solo de uso de memoria marginal.

## 8. Extensiones futuras plausibles

- **Persistencia del Trie en formato binario** con el árbol entero, no solo las palabras — aceleraría el `load` a costa de legibilidad.
- **`IndexStore` como interfaz abstracta** con backends alternativos (SQLite, Redis) sin tocar el servicio.
- **Índice invertido comprimido** (gap-coding, VByte) — relevante solo con corpus grandes.
- **Indexación incremental del SVD** vía `IncrementalPCA` o SVD rank-1 updates. Requiere cambiar `modules/retriever/lsi_model.py`.
- **Hooks de plugin** en `build`/`update` (`pre_index`, `post_index`) para conectar con el microkernel de plugins descrito en la arquitectura.
- **Filtro por metadata** (e.g., `build(..., filter_by={"language": "es"})`) para resolver la mezcla EN/ES en origen sin tener que duplicar el servicio.

## 9. Referencias cruzadas

- Contrato de datos: [`core/interfaces.py`](../core/interfaces.py) — `IndexedCorpus`, `DocumentStore`, `BaseRepository`.
- Modelos de dominio: [`core/models.py`](../core/models.py) — `Document`, `Query`, `RetrievedDocument`.
- Productores de `Document`: [`modules/crawler/`](../modules/crawler/) + [`modules/document_loader/`](../modules/document_loader/).
- Consumidor de `IndexedCorpus`: [`modules/retriever/service.py`](../modules/retriever/service.py) — `LSIRetriever.fit` y `.retrieve`.
- Preprocesado compartido: [`modules/text_processor/`](../modules/text_processor/) — `TextProcessor` y `TrieSpellChecker`.
- Persistencia del retriever (complementaria): [`modules/retriever/tfidf_processor.py`](../modules/retriever/tfidf_processor.py) y [`modules/retriever/lsi_model.py`](../modules/retriever/lsi_model.py) — cada uno con su `save`/`load` propios.
