---
name: lsi-implementation
description: Guía técnica para implementar el modelo LSI (Latent Semantic Indexing) del proyecto SRI. Usa este skill cuando trabajes con el módulo retriever, necesites construir o reconstruir el índice LSI, implementar la proyección de queries al espacio latente, ajustar el parámetro k de dimensiones, o integrar el Trie de corrección ortográfica. También aplica cuando necesites entender cómo fluyen los datos desde TF-IDF hasta la similitud coseno.
---

# Implementación del modelo LSI

## Visión general del flujo

```
Documentos → TF-IDF → Matriz A (M×N) → SVD truncado → Uk, Σk, VkT
                                                              ↓
Query → Trie (corrección) → TF-IDF → Proyección → Similitud coseno → Ranking
```

## Dependencias

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
```

No se necesitan librerías externas adicionales para el LSI básico. ChromaDB o FAISS se usan para almacenar los vectores resultantes, no para el cálculo LSI en sí.

## Paso 1: Construcción de la matriz TF-IDF

```python
# El vectorizer se entrena con TODO el corpus y se guarda para reusar con queries
vectorizer = TfidfVectorizer(
    max_features=10000,      # Limitar vocabulario si el corpus es muy grande
    stop_words=None,         # Gestionar stopwords médicas manualmente
    min_df=2,                # Ignorar términos que aparecen en menos de 2 docs
    max_df=0.95,             # Ignorar términos que aparecen en más del 95% de docs
)

# A tiene forma (N documentos × M términos) en sklearn — OJO: es la transpuesta
# de la notación clásica. sklearn trabaja con (samples × features).
tfidf_matrix = vectorizer.fit_transform(corpus_texts)  # shape: (N, M)
```

**Importante:** sklearn produce matrices (documentos × términos), que es la transpuesta de la notación matemática clásica (términos × documentos). Esto no afecta el resultado, pero hay que ser consciente al implementar la proyección de queries.

## Paso 2: SVD truncado

```python
k = 100  # Número de dimensiones latentes — ajustar empíricamente

svd = TruncatedSVD(n_components=k, random_state=42)
doc_vectors = svd.fit_transform(tfidf_matrix)  # shape: (N, k) — documentos en espacio latente
```

`doc_vectors` son los vectores de documentos en el espacio latente de dimensión k. Cada fila es un documento representado por k "conceptos".

### Selección de k

- k muy bajo (10-30): Pierde información, agrupa demasiado
- k muy alto (500+): No reduce el ruido, pierde el beneficio de LSI
- Para un corpus médico de tamaño medio: empezar con k=100, ajustar con evaluación
- Regla práctica: k ≈ 100-300 para corpus de 1000-10000 documentos

## Paso 3: Proyección de queries al espacio latente

```python
def project_query(query_text: str, vectorizer, svd) -> np.ndarray:
    """Proyecta una query al espacio latente LSI.
    
    Fórmula matemática: q_proj = q^T · Uk · Σk^{-1}
    En sklearn esto se simplifica porque svd.transform ya aplica la proyección.
    """
    # Vectorizar la query con el MISMO vectorizer entrenado
    query_tfidf = vectorizer.transform([query_text])  # shape: (1, M)
    
    # Proyectar al espacio latente
    query_vector = svd.transform(query_tfidf)  # shape: (1, k)
    
    return query_vector
```

**Punto crítico:** Si la query contiene palabras que NO están en el vocabulario del vectorizer, estas se ignoran silenciosamente. Por eso se usa el Trie de corrección ortográfica ANTES de este paso.

## Paso 4: Similitud coseno y ranking

```python
from sklearn.metrics.pairwise import cosine_similarity

def retrieve(query_vector, doc_vectors, top_n=10):
    """Calcula similitud coseno entre la query y todos los documentos."""
    similarities = cosine_similarity(query_vector, doc_vectors)[0]  # shape: (N,)
    
    # Índices ordenados de mayor a menor similitud
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_scores = similarities[top_indices]
    
    return list(zip(top_indices, top_scores))
```

## Corrección ortográfica con Trie

El Trie se construye con el vocabulario del vectorizer y se usa para corregir términos de la query antes de la vectorización.

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.word = None

class SpellChecker:
    """Corrector ortográfico basado en Trie para el vocabulario del corpus."""
    
    def __init__(self, vocabulary: list[str]):
        self.root = TrieNode()
        for word in vocabulary:
            self._insert(word)
    
    def _insert(self, word: str):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.word = word
    
    def correct(self, word: str, max_distance: int = 2) -> str:
        """Devuelve la palabra más cercana en el vocabulario."""
        # Implementar búsqueda por distancia de edición (Levenshtein)
        # usando BFS sobre el Trie
        ...
```

El vocabulario se extrae del vectorizer:
```python
vocabulary = vectorizer.get_feature_names_out().tolist()
spell_checker = SpellChecker(vocabulary)
```

## Integración con la base vectorial

Los `doc_vectors` (shape N × k) se almacenan en ChromaDB o FAISS para búsqueda eficiente:

```python
# Con ChromaDB
import chromadb

client = chromadb.Client()
collection = client.create_collection("medical_docs")

for idx, vector in enumerate(doc_vectors):
    collection.add(
        ids=[f"doc_{idx}"],
        embeddings=[vector.tolist()],
        metadatas=[{"source": doc_sources[idx]}]
    )
```

Para la búsqueda, se puede usar directamente ChromaDB o calcular la similitud manualmente. Si se usa ChromaDB, la query proyectada se pasa como embedding de búsqueda.

## Persistencia del modelo

Guardar el vectorizer y el SVD para no recalcular cada vez:

```python
import joblib

# Guardar
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(svd, "models/svd_model.pkl")
np.save("models/doc_vectors.npy", doc_vectors)

# Cargar
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
svd = joblib.load("models/svd_model.pkl")
doc_vectors = np.load("models/doc_vectors.npy")
```

## Errores comunes a evitar

1. **Entrenar un vectorizer nuevo para la query**: La query DEBE usar el mismo vectorizer del corpus. Si se entrena uno nuevo, los espacios vectoriales son incompatibles.
2. **Confundir la orientación de la matriz**: sklearn usa (docs × terms), la literatura usa (terms × docs). El resultado es equivalente pero la implementación cambia.
3. **Ignorar términos fuera de vocabulario**: Sin el Trie, queries con typos devuelven vectores degradados silenciosamente.
4. **k demasiado alto para corpus pequeño**: Si k ≥ min(N, M), SVD no reduce nada.
