---
name: testing
description: Patrones de testing para el proyecto SRI. Usa este skill cuando necesites escribir tests unitarios o de integración, crear mocks de interfaces ABC, testear módulos del pipeline de forma aislada, o verificar el flujo end-to-end. También aplica para testear la precisión del modelo LSI con queries de prueba.
---

# Testing del SRI

## Estructura de tests

```
tests/
├── conftest.py              # Fixtures compartidos (corpus de ejemplo, config mock)
├── unit/
│   ├── test_crawler.py
│   ├── test_indexer.py
│   ├── test_retriever.py
│   ├── test_ranker.py
│   ├── test_rag.py
│   └── test_web_search.py
├── integration/
│   ├── test_pipeline.py     # Flujo completo consulta → respuesta
│   └── test_plugins.py      # Hooks de plugins en el pipeline
└── evaluation/
    ├── test_queries.json     # Queries de prueba con relevancia esperada
    └── test_metrics.py       # P, R, F1, NDCG, MRR
```

## Principio: cada módulo es testeable aislado

Gracias a las interfaces ABC, cada módulo se puede testear con mocks de sus dependencias.

```python
# tests/unit/test_retriever.py
import pytest
import numpy as np
from modules.retriever.service import LSIRetriever
from core.models import Query

class TestLSIRetriever:
    @pytest.fixture
    def small_corpus(self):
        """Corpus médico pequeño para tests rápidos."""
        return [
            "dolor de cabeza intenso cefalea",
            "presión arterial alta hipertensión",
            "diabetes tipo dos glucosa insulina",
            "fractura de hueso traumatología",
            "infección respiratoria neumonía",
        ]
    
    @pytest.fixture
    def retriever(self, small_corpus):
        """Retriever LSI entrenado con corpus pequeño."""
        r = LSIRetriever(config={"k": 3})  # k bajo para corpus pequeño
        r.fit(small_corpus)
        return r
    
    def test_retrieve_returns_ranked_documents(self, retriever):
        results = retriever.retrieve(Query(text="dolor de cabeza"))
        assert len(results) > 0
        assert results[0].score >= results[-1].score
    
    def test_lsi_captures_synonyms(self, retriever):
        """LSI debe encontrar 'presión arterial' buscando 'hipertensión'."""
        results = retriever.retrieve(Query(text="hipertensión"))
        top_doc_texts = [r.text for r in results[:2]]
        assert any("presión arterial" in t for t in top_doc_texts)
    
    def test_empty_query_returns_empty(self, retriever):
        results = retriever.retrieve(Query(text=""))
        assert results == []
```

## Mocks para interfaces ABC

```python
# tests/conftest.py
from core.interfaces import BaseRetriever, BaseRanker
from core.models import Query, Document, RankedDocument

class MockRetriever(BaseRetriever):
    """Retriever falso que devuelve documentos predefinidos."""
    
    def __init__(self, fake_docs: list[Document]):
        self.fake_docs = fake_docs
    
    def retrieve(self, query: Query) -> list[Document]:
        return self.fake_docs

class MockRanker(BaseRanker):
    """Ranker que devuelve los docs en el orden que los recibe."""
    
    def rank(self, docs: list[Document], query: Query) -> list[RankedDocument]:
        return [RankedDocument(doc=d, score=1.0 - i * 0.1) for i, d in enumerate(docs)]
```

## Tests de integración del pipeline

```python
# tests/integration/test_pipeline.py
def test_pipeline_full_flow(mock_retriever, mock_ranker):
    """El pipeline ejecuta todas las etapas en orden."""
    pipeline = Pipeline(
        retriever=mock_retriever,
        ranker=mock_ranker,
        plugin_registry=PluginRegistry(),  # Sin plugins
    )
    result = pipeline.run(Query(text="síntomas de gripe"))
    assert result is not None
    assert len(result.documents) > 0

def test_plugin_hook_executes(mock_retriever, mock_ranker):
    """Los plugins registrados se ejecutan en su hook."""
    registry = PluginRegistry()
    spy_plugin = SpyPlugin(hook="pre_retrieval")  # Plugin que registra si fue llamado
    registry.register(spy_plugin)
    
    pipeline = Pipeline(retriever=mock_retriever, ranker=mock_ranker, plugin_registry=registry)
    pipeline.run(Query(text="test"))
    
    assert spy_plugin.was_called
```

## Tests de evaluación (Corte 3)

```python
# tests/evaluation/test_metrics.py
def test_precision_at_k():
    """Precision@10 del sistema sobre queries de evaluación."""
    with open("tests/evaluation/test_queries.json") as f:
        eval_set = json.load(f)
    
    precisions = []
    for item in eval_set:
        results = pipeline.run(Query(text=item["query"]))
        relevant = set(item["relevant_doc_ids"])
        retrieved = [r.doc_id for r in results.documents[:10]]
        p = len(set(retrieved) & relevant) / len(retrieved) if retrieved else 0
        precisions.append(p)
    
    avg_precision = sum(precisions) / len(precisions)
    assert avg_precision > 0.3  # Umbral mínimo aceptable
```

## Formato del archivo de queries de evaluación

```json
// tests/evaluation/test_queries.json
[
    {
        "query": "síntomas de hipertensión arterial",
        "relevant_doc_ids": ["doc_42", "doc_108", "doc_215"],
        "notes": "Debe encontrar docs sobre presión arterial alta"
    },
    {
        "query": "tratamiento diabetes tipo 2",
        "relevant_doc_ids": ["doc_73", "doc_156"],
        "notes": "Incluye docs sobre insulina y metformina"
    }
]
```

## Ejecutar tests

```bash
# Todos los tests
python -m pytest tests/ -v

# Solo unitarios
python -m pytest tests/unit/ -v

# Solo un módulo
python -m pytest tests/unit/test_retriever.py -v

# Con cobertura
python -m pytest tests/ --cov=modules --cov=core --cov-report=term-missing
```
