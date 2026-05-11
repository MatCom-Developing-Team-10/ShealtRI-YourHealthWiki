# Informe técnico — Pruebas unitarias 

**Proyecto:** ShealtRI — YourHealthWiki (SRI dominio salud/medicina)
**Fecha:** 2026-05-08
**Alcance:** revisión de todos los módulos del Corte 1 + suite completa de pruebas unitarias e integración.

---

## 1. Resumen ejecutivo

| Métrica | Valor |
|---|---|
| Módulos analizados | 10 paquetes (core + 5 modules + infra + 3 scrapers) |
| Tests implementados | **22 archivos** en `tests/unit/` y `tests/integration/` |
| Casos de prueba | **192** (190 ejecutados + 2 skipped por dependencias opcionales) |
| Estado | **190 pass, 0 fail, 2 skip** |
| Cobertura global | **79%** (líneas) |
| Tiempo de ejecución | ~37 s en una máquina local |
| Bugs reales detectados | **3** (uno demostrado por test) |
| Defectos no bloqueantes | **6** |


El sistema **funciona correctamente** para el flujo principal del Corte 1 (crawler → indexer → retriever LSI), pero presenta defectos que conviene atender antes del Corte 2: un bug en `LSIModel.fit()` con corpus muy pequeños, acoplamiento de imports que arrastra spaCy a usos triviales
---

## 2. Estado del proyecto

### 2.1 Módulos implementados vs. plan del Corte 1



| Componente requerido | Estado | Comentario |
|---|---|---|
| `crawler/` con robots.txt | ✅ implementado | `GenericCrawler`, 3 scrapers (Mayo, MedlinePlus, NHS), respeto a robots.txt configurable |
| `indexer/` con índices invertidos | ✅ implementado | `IndexerService`, `IndexedCorpus`, `IndexStore` (persistencia atómica) |
| `retriever/` con LSI básico | ✅ implementado | `TfidfProcessor` + `LSIModel` (TruncatedSVD) + `LSIRetriever` |
| Base vectorial | ✅ implementado | `ChromaRepository` (persistente) |
| Dockerfile | ❌ **ausente** | Incumple `CLAUDE.md` §"Reglas importantes" |
| `docker-compose.yml` | ❌ **ausente** | Mismo motivo |

Adicionalmente se añadieron funcionalidades de valor: `TrieSpellChecker` para corrección ortográfica (entra en `TextProcessor` automáticamente), `FileSystemDocumentStore` con sanitización de IDs frente a path-traversal, e `IndexStore` con escrituras atómicas (`tmp` + rename).

### 2.2 Estructura

```
src/
├── core/                          # interfaces ABC, modelos, RetrievalContext (Strategy)
├── modules/
│   ├── crawler/                   # GenericCrawler + ScraperRegistry + 3 scrapers
│   ├── document_loader/           # DocumentLoader (LangChain) — opcional
│   ├── indexer/                   # IndexerService + DocumentStore + IndexStore
│   ├── retriever/                 # TfidfProcessor + LSIModel + LSIRetriever
│   └── text_processor/            # TextProcessor (spaCy) + TrieSpellChecker
├── infra/                         # RawDocumentStorage (JSONL) + ChromaRepository
└── tests/                         # NUEVO: 22 archivos, 192 casos
    ├── conftest.py                # fakes in-memory de DocumentStore y BaseRepository
    ├── unit/
    └── integration/
```


---

## 3. Análisis de correctitud por módulo

A continuación se documenta lo encontrado en cada módulo, separando lo que funciona, lo que tiene defectos no bloqueantes, y los bugs reales.

### 3.1 `core/`

#### Lo que funciona
- `Document`, `Query`, `RetrievedDocument` son `@dataclass(slots=True)` correctas. `metadata` usa `field(default_factory=dict)` (no comparte estado entre instancias) — verificado por test.
- `IndexedCorpus.__post_init__` valida la invariante `len(documents) == len(processed_texts)`.
- `RetrievalContext` (Strategy) intercambia estrategia en runtime sin estado oculto.
- Las ABC (`DocumentStore`, `BaseRepository`, `BaseRetriever`) **rechazan instanciación directa** correctamente.

#### Defectos no bloqueantes
- **Nombre engañoso de archivo**: `core/pipeline.py` no contiene un pipeline; debería llamarse `core/retrieval_context.py`.
- `DocumentStore.delete()` levanta `NotImplementedError` por defecto en lugar de hacer no-op o ser abstracto. Es legible, pero la implementación `FileSystemDocumentStore.delete()` sí lo soporta — la asimetría puede sorprender a usuarios de la ABC.

### 3.2 `modules/text_processor/`

#### Lo que funciona
- Pipeline completo (`normalize → tokenize → remove_stopwords → lemmatize → filter`) verificado con texto médico real en español.
- Acentos preservados por defecto (`año ≠ ano`); opción `remove_accents=True` cuando se necesita.
- Cada `TextProcessor` tiene su propio `TrieSpellChecker` (verificado por test).
- `TrieSpellChecker.correct()` con distancia de Levenshtein cubre casos típicos de typos (kitten/sitting=3, hipertensoin→hipertensión).
- Stopwords médicos están **fuera** del set de stopwords (paciente, tratamiento, síntoma, alto, bajo). No hay solapamiento entre `ADDITIONAL_SPANISH_STOPWORDS` y `MEDICAL_ABBREVIATIONS`.

#### Defectos no bloqueantes (vale la pena corregir)
1. **Doble paso por spaCy** en `process()`: `tokenize()` llama `self._nlp(text)` y luego `lemmatize()` vuelve a llamar `self._nlp(" ".join(tokens))`. Es redundante y duplica el costo (el más caro del pipeline). Una sola pasada bastaría.
2. **Acceso a método privado**: `TextProcessor._add_to_vocabulary` invoca `self.spell_checker._insert(token)`. Cruzar la barrera del subrayado interno sugiere que `TrieSpellChecker.insert()` debería ser pública.
3. **`lemmatize()` re-tokeniza**: tras `remove_stopwords`, `lemmatize` hace `" ".join(tokens)` y vuelve a tokenizar con spaCy. spaCy puede re-segmentar los tokens y romper la alineación 1:1.

### 3.3 `modules/indexer/`

#### Lo que funciona
- `IndexerService.build()` produce un `IndexedCorpus` consistente: vocabulario ordenado, índice invertido con frecuencia por documento, longitudes cuadradas.
- `update()` es idempotente por `doc_id`: documentos repetidos se descartan sin duplicar postings (verificado por test).
- `remove()` renumera correctamente los índices supervivientes y elimina términos cuya posting list queda vacía.
- `build_query()` genera un `IndexedCorpus` de un solo documento (con metadato `is_query=True`) y aplica corrección ortográfica vía spell checker.
- `FileSystemDocumentStore` **bloquea path-traversal** (verificado: `Document("../../etc/passwd", …)` se persiste con hash, no escribe fuera del sandbox).
- `IndexStore` escribe atómicamente (`tmp` + `rename`); manifiesto preserva `created_at` entre saves.

#### Bug detectado
4. **Acoplamiento fuerte de imports** (no es un bug funcional pero sí de arquitectura): `modules/indexer/__init__.py` re-exporta `IndexStore`, que importa `TrieSpellChecker` desde el paquete `modules.text_processor`, cuyo `__init__.py` importa `service.py`, **que carga spaCy**. Resultado: importar cualquier cosa de `modules.indexer` (incluyendo `FileSystemDocumentStore`, que no necesita NLP) **carga spaCy y NLTK**. Esto bloqueó dos veces nuestra fase de exploración.
   - Solución: que `index_store.py` importe `TrieSpellChecker` directamente desde `modules.text_processor.spell_checker`, sin tocar el `__init__`.

### 3.4 `modules/retriever/`

#### Lo que funciona
- `TfidfProcessor.fit()` calcula IDF correctamente: `log((N+1)/(df+1)) + 1`. Verificado: términos raros tienen IDF mayor que comunes.
- `TfidfProcessor.transform()` filtra términos OOV silenciosamente (no reventa).
- `LSIModel.fit()` reduce la dimensión y trunca `n_components` cuando el corpus es pequeño.
- `LSIRetriever`: orquesta TF-IDF + LSI + repositorio + document store en dos fases (vector search → fetch full text), filtra por umbral de similitud y preserva orden de ranking. Tests verifican: error claro si no se ha hecho `fit()`, error claro si la query no tiene `indexed_corpus`, ranking decreciente por score.
- `save()` / `load()` round-trip preserva vocabulario, IDF y embeddings.

#### Bug real detectado por test
5. **`LSIModel.fit()` falla con corpus de 1 documento**:
   ```python
   effective_k = min(self.n_components, n_terms - 1, n_docs - 1)
   ```
   Con `n_docs == 1`, `n_docs - 1 == 0` → `effective_k = 0` → `TruncatedSVD(n_components=0)` levanta `InvalidParameterError`.
   - **Severidad:** baja (un corpus de Corte 1 nunca tendrá un solo doc), pero **alta importancia para tests y entornos de desarrollo**.
   - **Fix sugerido:** `effective_k = max(1, min(self.n_components, n_terms - 1, n_docs - 1))` y, si `n_docs < 2`, lanzar un error de dominio claro.
   - Test que documenta el bug: `tests/unit/test_lsi_model.py::TestFit::test_single_document_corpus_currently_fails`.

### 3.5 `modules/crawler/`

#### Lo que funciona
- `BaseScraper.__init_subclass__` valida la presencia de `domain` y `source_name` (verificado).
- `ScraperRegistry` respeta orden de registro (verificado por test con scrapers solapados).
- `GenericCrawler`:
  - Genera `doc_id` estables (UUID5 sobre la URL) — re-crawls producen el mismo ID.
  - Maneja `<sitemapindex>` recursivamente con tope de profundidad.
  - Aísla excepciones por scraper (un scraper que crashea no aborta el crawl). Verificado.
  - Respeta `max_pages`, retraso por dominio, robots.txt configurable.
  - Persistencia incremental por batch (`_BATCH_FLUSH_SIZE = 50`).
- Scrapers (Mayo, MedlinePlus, NHS): correctos en filtrado por path, extracción de título, idioma, fecha y categoría. Devuelven `None` cuando hay HTML insuficiente.

#### Defectos no bloqueantes
6. **`CrawlerService.__init__`** instancia su propio `RawDocumentStorage(self._config.output_dir)`, ignorando un eventual storage inyectado. Para tests deterministas y para inyectar mocks habría sido más limpio aceptarlo como parámetro opcional. Trabajamos alrededor con `monkeypatch`.

### 3.6 `infra/`

#### Lo que funciona
- `RawDocumentStorage`: append a JSONL, `ensure_ascii=False` (preserva acentos), `save_batch` continúa tras documentos no serializables, `clear()` borra el archivo.
- `ChromaRepository`: `1 - distance` clampeado a `[0, 1]`, manejo defensivo cuando la colección está vacía o ChromaDB devuelve estructura inesperada. (Tests skipped en este informe porque chromadb no está instalado en el entorno del análisis; el código fue auditado manualmente).

---

## 5. Suite de pruebas implementada

### 5.1 Layout

```
tests/
├── conftest.py                              # fakes + fixtures + sys.path
├── unit/
│   ├── test_core_models.py
│   ├── test_core_interfaces.py
│   ├── test_pipeline.py
│   ├── test_spell_checker.py
│   ├── test_stopwords.py
│   ├── test_text_processor.py
│   ├── test_indexer_service.py
│   ├── test_document_store.py
│   ├── test_index_store.py
│   ├── test_tfidf_processor.py
│   ├── test_lsi_model.py
│   ├── test_retriever_service.py
│   ├── test_crawler_models.py
│   ├── test_crawler_base.py
│   ├── test_crawler_registry.py
│   ├── test_crawler_generic.py
│   ├── test_crawler_service.py
│   ├── test_scrapers_mayo.py
│   ├── test_scrapers_medlineplus.py
│   ├── test_scrapers_nhs.py
│   ├── test_infra_storage.py
│   ├── test_chroma_repository.py            # importorskip
│   └── test_document_loader.py              # importorskip
└── integration/
    └── test_indexing_to_retrieval.py        # E2E con spaCy real
```

### 5.2 Estrategia de aislamiento

- **`conftest.py`** define dos fakes in-memory:
  - `InMemoryDocumentStore`: implementa la ABC `DocumentStore` con un `dict`.
  - `InMemoryRepository`: implementa `BaseRepository` con coseno en Python puro.


- **`text_processor` fixture** es session-scoped: spaCy carga **una vez** por corrida, no por test (ahorra ~25 s en una suite de 17 tests del módulo).

- **HTTP en crawler tests** está mockeado vía `unittest.mock.patch`: ninguna conexión real durante la suite.

- **Disco**: todos los tests que escriben en filesystem usan `tmp_path` (built-in pytest), por lo que la suite no contamina el repo.

- **Dependencias opcionales** (chromadb, langchain): `pytest.importorskip(...)` salta la prueba si la librería no está. La suite **nunca falla** por dependencias opcionales ausentes; sólo emite skip.

### 5.3 Cobertura por módulo

| Módulo | Cobertura | Notas |
|---|---|---|
| `core/models.py` | **100 %** | Todos los dataclasses ejercitados |
| `core/pipeline.py` | **100 %** | Strategy con tres tests |
| `core/interfaces.py` | 83 % | Líneas restantes: cuerpos de `raise NotImplementedError` en métodos abstractos (no son alcanzables) |
| `modules/text_processor/spell_checker.py` | **100 %** | Trie + Levenshtein + búsqueda |
| `modules/text_processor/service.py` | 90 % | Faltan ramas de error de carga del modelo y de `nltk.download` |
| `modules/text_processor/stopwords.py` | **100 %** | Constantes |
| `modules/indexer/service.py` | **97 %** | Tres líneas de logging no cubiertas |
| `modules/indexer/document_store.py` | 83 % | Algunas ramas de error de I/O |
| `modules/indexer/index_store.py` | 83 % | Caminos de error en `_atomic_*_write` |
| `modules/retriever/lsi_model.py` | **100 %** | Incluye test que documenta el bug |
| `modules/retriever/service.py` | **100 %** | Fit, retrieve, threshold, persistencia |
| `modules/retriever/tfidf_processor.py` | 98 % | Una línea: rama "term not in term_to_idx" durante fit |
| `modules/crawler/crawler.py` | 71 % | Sin red real, varias ramas HTTP no se ejercitan |
| `modules/crawler/registry.py` | 92 % | `__repr__` no cubierto |
| `modules/crawler/scrapers/*.py` | 88-94 % | Faltan algunas ramas defensivas |
| `modules/crawler/service.py` | **100 %** | |
| `infra/storage.py` | 93 % | Líneas faltantes son ramas de OSError |
| `infra/chroma_repository.py` | 0 % | chromadb no instalado en el entorno → tests skipped |
| `modules/document_loader/service.py` | 0 % | langchain no instalado → tests skipped |
| **Total** | **79 %** | 1149/1449 líneas cubiertas |

### 5.4 Cómo ejecutar

```bash
# Suite completa
python -m pytest tests/ -v

# Sólo unit tests
python -m pytest tests/unit/ -v

# Sólo un módulo
python -m pytest tests/unit/test_indexer_service.py -v

# Con cobertura
python -m pytest tests/ --cov=core --cov=modules --cov=infra --cov-report=term-missing

# En paralelo (necesita pytest-xdist instalado)
python -m pytest tests/ -n auto
```

### 5.5 Resultado de la última ejecución

```
======================= 190 passed, 2 skipped in 37.38s =======================
```

Skipped: `tests/unit/test_chroma_repository.py` y `tests/unit/test_document_loader.py`, que dependen de `chromadb` y `langchain_community` — ambos opcionales en este corte.

---

## 6. Hallazgos consolidados (lista de acción)

Ordenados por prioridad para el equipo:

| # | Tipo | Severidad | Descripción | Fix sugerido |
|---|---|---|---|---|
| 1 | Cumplimiento | **alta** | Falta `Dockerfile` (mandato explícito de `CLAUDE.md`) | Crear `Dockerfile` con `python:3.11-slim` + spaCy model |
| 2 | Cumplimiento | **alta** | Falta `docker-compose.yml` | Crear servicio `app` + volumen para `data/chroma` |
| 3 | Bug | media | `LSIModel.fit()` falla con 1 documento (n_components=0) | `effective_k = max(1, min(...))` o validar `n_docs >= 2` |
| 4 | Arquitectura | media | `modules.indexer` arrastra spaCy por re-export transitivo | Importar `TrieSpellChecker` desde el submódulo, no del package |
| 5 | Doc/Código | baja | `core/pipeline.py` no contiene un pipeline | Renombrar a `retrieval_context.py` o crear pipeline real |
| 6 | Doc/Código | baja | Comandos `python -m core.pipeline` y `python -m modules.indexer.service` documentados pero no implementados | Implementar CLI o quitar de docs |
| 7 | Eficiencia | baja | `TextProcessor.process()` invoca spaCy dos veces | Una sola pasada, lematizar usando los `Token` originales |
| 8 | Encapsulamiento | baja | `TextProcessor._add_to_vocabulary` llama `_insert` privado | Exponer `TrieSpellChecker.insert()` |
| 9 | Encapsulamiento | baja | `CrawlerService` instancia su `RawDocumentStorage` sin permitir inyección | Aceptar storage como parámetro opcional |
| 10 | Cobertura | baja | `infra/chroma_repository.py` y `document_loader/` sin cobertura efectiva | Agregar `chromadb`, `langchain-community` a la imagen Docker para CI |

---

## 7. Conclusiones

El proyecto **cumple funcionalmente** con los requisitos del Corte 1: el flujo Crawl → Index → Retrieve LSI está completo, los tres scrapers extraen correctamente contenido médico, y la corrección ortográfica funciona como capa transparente sobre `TextProcessor`. La arquitectura Pipeline + Microkernel está respetada en sus contratos (ABC) aunque el archivo `core/pipeline.py` esté mal nombrado.

Los puntos críticos a cerrar antes de defensa son **el Dockerfile** (mandato explícito) y **el bug del LSI** (que afectaría escenarios de demo con corpora pequeños). Los demás hallazgos son mejoras incrementales que no bloquean la entrega.

La suite de pruebas creada (192 casos, cobertura 79 %) actúa como red de seguridad para el Corte 2: cualquier refactorización del retriever o introducción de plugins puede validarse contra los tests existentes. La estrategia de mocks vía ABCs permite que la suite **no requiera ChromaDB ni LangChain** y se ejecute en menos de 40 segundos.
