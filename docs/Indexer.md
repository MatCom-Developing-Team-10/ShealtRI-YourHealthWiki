# Modulo Indexer — Informe de Implementacion

## Diseno

### `IndexerService.build(documents) -> IndexedCorpus`

Algoritmo single-pass, O(N · avg_tokens):

```
for each doc in documents:
    processed = text_processor.process(doc.text, is_query=False)
    tokens    = processed.split()
    if len(tokens) < config.min_document_length:  skip
    doc_idx   = len(valid_documents)              # contiguous, no gaps
    for term, tf in Counter(tokens).items():
        inverted_index[term].append((doc_idx, tf))
    valid_documents.append(doc)
    prune terms whose total tf < config.min_term_frequency
    vocabulary = sorted(inverted_index.keys())
    
    return IndexedCorpus(documents, processed_texts, dict(inverted_index), vocabulary)
```

**Decisiones clave:**

- **Contiguous numeric `doc_idx` (not `doc.doc_id`).** `TfidfProcessor` builds a
  `csr_matrix` keyed por filas enteras. Los documentos omitidos, de otro modo,
  dejarian filas con huecos; reasignar indices garantiza una matriz densa y sin huecos.
- **`is_query=False`.** Este es el unico punto en el que el spell-checker aprende
  el vocabulary del corpus. El contrato de `TextProcessor.process()` envia tokens al
  `TrieSpellChecker` interno solo cuando `is_query=False`. El momento de indexacion es
  exactamente cuando debe ocurrir ese aprendizaje.
- **`vocabulary = sorted(...)`.** Python `dict` iteration order is insertion-stable
  pero no lexicografico; ordenar aporta reproducibilidad entre ejecuciones y plataformas,
  y facilita el debugging.
- **`defaultdict(list)` during build, `dict(...)` on return.** Coincide con el
  tipo declarado de `IndexedCorpus.inverted_index: dict[str, list[tuple[int, int]]]`.

### 3.2 `IndexerService.build_query(text) -> IndexedCorpus`

Wrapper simetrico. La query se envuelve como un unico
`Document(doc_id="__query__", text=text, url="")` sintetico para que
`TfidfProcessor.transform()` la consuma a traves de exactamente la misma interfaz
que un corpus de documentos.

Aqui el spell-checker corre en *correction mode* (`is_query=True`), por lo que typos como
`hipertensoin arterail` se mapean a terminos del vocabulary indexado.

### 3.3 Por que no hay nuevas dependencias

La implementacion completa usa solo stdlib: `collections.Counter`,
`collections.defaultdict`, `dataclasses`, `logging`. No se agregan nuevos packages.

---

## 4. Integracion con el pipeline general

```
                    ┌─────────────────────┐
                    │   DocumentLoader    │  data/sample/*.json
                    └──────────┬──────────┘
                               │ list[Document]
                               ▼
                    ┌─────────────────────┐
                    │   TextProcessor     │  spaCy lemmatization
                    │   (is_query=False)  │  + Trie spell-checker (LEARN)
                    └──────────┬──────────┘
                               │ str (preprocessed text)
                               ▼
                    ┌─────────────────────┐
                    │   IndexerService    │  ← THIS MODULE
                    │      .build()       │
                    └──────────┬──────────┘
                               │ IndexedCorpus
                               ▼
                    ┌─────────────────────┐
                    │   TfidfProcessor    │  csr_matrix from inverted_index
                    │       .fit()        │
                    └──────────┬──────────┘
                               │ tfidf_matrix (terms × docs)
                               ▼
                    ┌─────────────────────┐
                    │      LSIModel       │  TruncatedSVD
                    │       .fit()        │
                    └──────────┬──────────┘
                               │ doc_embeddings (k-dim)
                               ▼
                    ┌─────────────────────┐
                    │ ChromaRepository +  │
                    │ FileSystemDocStore  │
                    └─────────────────────┘
```

El `IndexedCorpus` producido por `build()` satisface exactamente la forma que requiere
la siguiente etapa, verificado contra
[modules/retriever/tfidf_processor.py](../modules/retriever/tfidf_processor.py)
que itera sobre `corpus.vocabulary` y `corpus.inverted_index` para poblar
su sparse matrix.

Para queries, `build_query()` produce un `IndexedCorpus` de una sola fila que
`TfidfProcessor.transform()` ya sabe consumir (filtra terminos que no estan presentes
en el vocabulary entrenado y devuelve un sparse vector de 1×|V|).

