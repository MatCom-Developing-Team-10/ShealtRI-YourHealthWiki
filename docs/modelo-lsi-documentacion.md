# Modelo de Recuperación de Información basado en Latent Semantic Indexing (LSI)

## Documentación Técnica Formal

**Proyecto:** ShealtRI-YourHealthWiki — Sistema de Recuperación de Información para el Dominio de Salud y Medicina
**Versión:** 1.0
**Fecha:** Marzo 2026

---

## Resumen

Este documento presenta la especificación técnica formal del modelo de recuperación de información basado en *Latent Semantic Indexing* (LSI) implementado en el sistema ShealtRI-YourHealthWiki. Se describen los fundamentos teóricos del modelo, su formulación matemática, la integración con un corrector ortográfico basado en estructuras *Trie*, y las decisiones de implementación utilizando bibliotecas de Python. El modelo LSI permite capturar relaciones semánticas latentes entre términos y documentos mediante la reducción de dimensionalidad de la matriz TF-IDF, superando las limitaciones de los modelos de correspondencia exacta (*exact matching*).

**Palabras clave:** Recuperación de información, Latent Semantic Indexing, SVD, TF-IDF, Trie, corrección ortográfica, similitud coseno.

---

## 1. Introducción

### 1.1 Contexto y Motivación

Los sistemas tradicionales de recuperación de información basados en correspondencia de términos enfrentan dos problemas fundamentales conocidos como **sinonimia** y **polisemia** [1]. La sinonimia ocurre cuando diferentes palabras expresan el mismo concepto (por ejemplo, "hipertensión arterial" y "presión alta"), mientras que la polisemia se presenta cuando una misma palabra tiene múltiples significados dependiendo del contexto.

En el dominio de salud y medicina, estos problemas son particularmente críticos debido a:

- La coexistencia de terminología técnica y coloquial
- Las múltiples denominaciones para fármacos (nombre genérico vs. comercial)
- Las abreviaturas médicas (ECG, DM2, EPOC)
- Los sinónimos clínicos ("infarto de miocardio" vs. "ataque cardíaco")

*Latent Semantic Indexing* (LSI), propuesto por Deerwester et al. en 1990 [2], aborda estos problemas mediante la identificación de patrones de co-ocurrencia de términos a través de técnicas de álgebra lineal, específicamente la *Descomposición en Valores Singulares* (SVD).

### 1.2 Objetivos del Modelo

El modelo LSI implementado tiene los siguientes objetivos:

1. **Captura de semántica latente:** Identificar relaciones conceptuales entre términos que no comparten tokens explícitos.
2. **Reducción de ruido:** Eliminar variabilidad superficial en el vocabulario médico.
3. **Eficiencia computacional:** Reducir la dimensionalidad del espacio vectorial para acelerar las búsquedas.
4. **Tolerancia a errores:** Integrar corrección ortográfica para consultas con errores tipográficos.

---

## 2. Fundamentos Teóricos

### 2.1 Modelo de Espacio Vectorial

El modelo de espacio vectorial (*Vector Space Model*, VSM) [1] representa documentos y consultas como vectores en un espacio de *m* dimensiones, donde *m* es el tamaño del vocabulario. Cada dimensión corresponde a un término único del corpus.

Sea *D* = {*d*₁, *d*₂, ..., *d*ₙ} un corpus de *n* documentos y *T* = {*t*₁, *t*₂, ..., *t*ₘ} el vocabulario de *m* términos. Un documento *d*ⱼ se representa como un vector:

$$\vec{d}_j = (w_{1j}, w_{2j}, ..., w_{mj})$$

donde *w*ᵢⱼ es el peso del término *t*ᵢ en el documento *d*ⱼ.

### 2.2 Esquema de Ponderación TF-IDF

El esquema *Term Frequency - Inverse Document Frequency* (TF-IDF) [1] asigna pesos que reflejan tanto la frecuencia local del término en el documento como su especificidad global en el corpus.

#### 2.2.1 Frecuencia de Término (TF)

La frecuencia de término *tf*(*t*, *d*) mide cuántas veces aparece el término *t* en el documento *d*. Se utiliza comúnmente una normalización logarítmica:

$$tf(t, d) = \begin{cases} 1 + \log(f_{t,d}) & \text{si } f_{t,d} > 0 \\ 0 & \text{en caso contrario} \end{cases}$$

donde *f*ₜ,ₐ es el conteo crudo de ocurrencias.

#### 2.2.2 Frecuencia Inversa de Documento (IDF)

La frecuencia inversa de documento penaliza términos que aparecen en muchos documentos:

$$idf(t) = \log\left(\frac{N}{df(t)}\right) + 1$$

donde *N* es el número total de documentos y *df*(*t*) es el número de documentos que contienen el término *t*.

#### 2.2.3 Peso TF-IDF

El peso final se calcula como:

$$w_{t,d} = tf(t, d) \times idf(t)$$

### 2.3 Latent Semantic Indexing (LSI)

#### 2.3.1 Intuición

LSI parte de la observación de que la matriz término-documento contiene redundancia debido a las correlaciones entre términos. Mediante SVD, se proyecta el espacio original de alta dimensionalidad a un espacio latente de menor dimensión que preserva las relaciones semánticas más importantes [2].

#### 2.3.2 Construcción de la Matriz Término-Documento

Sea **A** ∈ ℝᵐˣⁿ la matriz término-documento donde:

- *m* = número de términos (filas)
- *n* = número de documentos (columnas)
- **A**ᵢⱼ = peso TF-IDF del término *i* en el documento *j*

$$\mathbf{A} = \begin{bmatrix} w_{1,1} & w_{1,2} & \cdots & w_{1,n} \\ w_{2,1} & w_{2,2} & \cdots & w_{2,n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m,1} & w_{m,2} & \cdots & w_{m,n} \end{bmatrix}$$

#### 2.3.3 Descomposición en Valores Singulares (SVD)

La SVD factoriza la matriz **A** en tres matrices [1, 2]:

$$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T$$

donde:

- **U** ∈ ℝᵐˣᵐ : Matriz ortogonal de vectores singulares izquierdos (representa términos en el espacio conceptual)
- **Σ** ∈ ℝᵐˣⁿ : Matriz diagonal de valores singulares σ₁ ≥ σ₂ ≥ ... ≥ σᵣ > 0, donde *r* = rango(**A**)
- **V** ∈ ℝⁿˣⁿ : Matriz ortogonal de vectores singulares derechos (representa documentos en el espacio conceptual)

Las propiedades fundamentales de estas matrices son:

$$\mathbf{U}^T\mathbf{U} = \mathbf{I}_m, \quad \mathbf{V}^T\mathbf{V} = \mathbf{I}_n$$

#### 2.3.4 Aproximación de Bajo Rango (SVD Truncada)

La clave de LSI es la **aproximación de bajo rango**. Se trunca la SVD reteniendo solo los *k* valores singulares más grandes (*k* << min(*m*, *n*)):

$$\mathbf{A}_k = \mathbf{U}_k \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$

donde:

- **U**ₖ ∈ ℝᵐˣᵏ : Las primeras *k* columnas de **U**
- **Σ**ₖ ∈ ℝᵏˣᵏ : Submatriz diagonal con los *k* valores singulares más grandes
- **V**ₖ ∈ ℝⁿˣᵏ : Las primeras *k* columnas de **V**

Por el **Teorema de Eckart-Young-Mirsky**, **A**ₖ es la mejor aproximación de rango *k* de **A** en términos de la norma de Frobenius:

$$\mathbf{A}_k = \arg\min_{\text{rango}(\mathbf{B}) \leq k} \|\mathbf{A} - \mathbf{B}\|_F$$

#### 2.3.5 Interpretación Semántica

Las *k* dimensiones del espacio latente pueden interpretarse como "conceptos" o "tópicos" semánticos:

- Las filas de **U**ₖ representan la asociación de cada término con cada concepto
- Las columnas de **V**ₖᵀ (equivalentemente, las filas de **V**ₖ) representan la asociación de cada documento con cada concepto
- Los valores singulares σᵢ indican la "importancia" de cada concepto

### 2.4 Representación de Documentos en el Espacio Latente

La representación de los documentos en el espacio latente de *k* dimensiones se obtiene como:

$$\mathbf{D}_{latente} = \boldsymbol{\Sigma}_k \mathbf{V}_k^T$$

Cada columna de **D**ₗₐₜₑₙₜₑ es un vector de *k* dimensiones que representa un documento en el espacio semántico reducido.

Alternativamente, también se puede usar:

$$\mathbf{D}_{latente} = \mathbf{V}_k$$

donde cada fila de **V**ₖ es la representación de un documento.

### 2.5 Proyección de Consultas al Espacio Latente

Una consulta *q* se representa primero como un vector en el espacio TF-IDF original:

$$\vec{q} = (w_{q,1}, w_{q,2}, ..., w_{q,m})$$

Para proyectar la consulta al espacio latente, se aplica la transformación [2]:

$$\vec{q}_{latente} = \boldsymbol{\Sigma}_k^{-1} \mathbf{U}_k^T \vec{q}$$

Donde:

- **U**ₖᵀ**q** proyecta la consulta al espacio de los vectores singulares izquierdos
- **Σ**ₖ⁻¹ escala por los valores singulares inversos para normalizar

El resultado **q**ₗₐₜₑₙₜₑ ∈ ℝᵏ es un vector de *k* dimensiones comparable directamente con los vectores de documentos en **V**ₖ.

### 2.6 Cálculo de Similitud

La relevancia de un documento *d*ⱼ para una consulta *q* se calcula mediante la **similitud coseno** en el espacio latente [1]:

$$sim(q, d_j) = \frac{\vec{q}_{latente} \cdot \vec{d}_{j,latente}}{\|\vec{q}_{latente}\| \cdot \|\vec{d}_{j,latente}\|}$$

$$sim(q, d_j) = \frac{\sum_{i=1}^{k} q_i \cdot d_{ji}}{\sqrt{\sum_{i=1}^{k} q_i^2} \cdot \sqrt{\sum_{i=1}^{k} d_{ji}^2}}$$

El rango de la similitud coseno es [-1, 1], donde:

- 1 indica máxima similitud (vectores paralelos)
- 0 indica ortogonalidad (sin relación)
- -1 indica oposición (vectores antiparalelos)

En la práctica, los valores suelen estar en [0, 1] debido a la naturaleza no negativa de los pesos TF-IDF.

### 2.7 Selección del Parámetro *k*

La elección del número de dimensiones latentes *k* es crucial y representa un compromiso (*trade-off*):

| *k* pequeño | *k* grande |
|-------------|------------|
| Mayor generalización | Menor generalización |
| Pérdida de información específica | Preserva más información |
| Más robusto al ruido | Puede capturar ruido |
| Consultas más rápidas | Consultas más lentas |

**Criterios para seleccionar *k*:**

1. **Análisis del scree plot:** Graficar los valores singulares y buscar el "codo" donde la pendiente disminuye significativamente.

2. **Varianza explicada acumulada:** Seleccionar *k* tal que se preserve un porcentaje objetivo de varianza:

$$\frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} \geq \text{umbral}$$

Típicamente se busca preservar entre 80% y 95% de la varianza.

3. **Validación empírica:** Evaluar el rendimiento de recuperación (precisión, recall, F1) con diferentes valores de *k* en un conjunto de validación.

**Valores típicos:** Para colecciones de tamaño moderado (miles a decenas de miles de documentos), *k* suele estar en el rango de 100-500 [2].

---

## 3. Corrección Ortográfica con Estructura Trie

### 3.1 Motivación

Las consultas de usuarios frecuentemente contienen errores ortográficos ("hipertencion" en lugar de "hipertensión", "diabetis" en lugar de "diabetes"). Antes de vectorizar la consulta, es necesario corregir estos errores para obtener una proyección válida al espacio latente.

### 3.2 Estructura de Datos Trie

Un **Trie** (del inglés *retrieval*), también conocido como *árbol de prefijos*, es una estructura de datos de árbol ordenado que almacena un conjunto dinámico de cadenas [1]. A diferencia de un árbol binario de búsqueda, cada nodo no almacena una clave completa, sino que la posición del nodo en el árbol define la clave asociada.

#### 3.2.1 Definición Formal

Un Trie *T* sobre un alfabeto Σ es un árbol donde:

- Cada arista está etiquetada con exactamente un símbolo σ ∈ Σ
- Las aristas que salen de un nodo tienen etiquetas distintas
- Cada nodo tiene un atributo booleano *is_end* que indica si la concatenación de las etiquetas desde la raíz hasta ese nodo forma una palabra válida del diccionario

#### 3.2.2 Estructura del Nodo

```
TrieNode {
    children: Map<Character, TrieNode>
    is_end: Boolean
}
```

#### 3.2.3 Complejidad Algorítmica

| Operación | Complejidad Temporal | Complejidad Espacial |
|-----------|---------------------|---------------------|
| Inserción | O(*m*) | O(*m*) |
| Búsqueda exacta | O(*m*) | O(1) |
| Construcción (*n* palabras) | O(*n* × *m̄*) | O(ΣALFABETO × *m̄* × *n*) |

donde *m* es la longitud de la palabra y *m̄* es la longitud promedio.

### 3.3 Distancia de Levenshtein

La **distancia de Levenshtein** (o distancia de edición) [3] entre dos cadenas es el número mínimo de operaciones de edición necesarias para transformar una cadena en la otra. Las operaciones permitidas son:

1. **Inserción** de un carácter
2. **Eliminación** de un carácter
3. **Sustitución** de un carácter por otro

#### 3.3.1 Definición Recursiva

Sea *a* = *a*₁*a*₂...*a*ₘ y *b* = *b*₁*b*₂...*b*ₙ dos cadenas. La distancia de Levenshtein *lev*(*a*, *b*) se define como:

$$lev(a, b) = \begin{cases} |a| & \text{si } |b| = 0 \\ |b| & \text{si } |a| = 0 \\ lev(tail(a), tail(b)) & \text{si } a_1 = b_1 \\ 1 + \min \begin{cases} lev(tail(a), b) \\ lev(a, tail(b)) \\ lev(tail(a), tail(b)) \end{cases} & \text{en caso contrario} \end{cases}$$

#### 3.3.2 Algoritmo de Programación Dinámica

La implementación eficiente utiliza una matriz *D* de (*m*+1) × (*n*+1):

```
Para i de 0 a m:
    D[i, 0] = i
Para j de 0 a n:
    D[0, j] = j

Para i de 1 a m:
    Para j de 1 a n:
        costo = 0 si a[i] == b[j], sino 1
        D[i, j] = min(
            D[i-1, j] + 1,      // eliminación
            D[i, j-1] + 1,      // inserción
            D[i-1, j-1] + costo // sustitución
        )

Retornar D[m, n]
```

**Complejidad:** O(*m* × *n*) en tiempo y espacio.

### 3.4 Algoritmo de Corrección Ortográfica

El corrector ortográfico integrado en el sistema utiliza el siguiente algoritmo:

```
Función corregir_palabra(palabra, vocabulario_trie, max_distancia):
    Si vocabulario_trie.contiene(palabra):
        Retornar palabra  // No necesita corrección

    mejor_candidato = palabra
    menor_distancia = max_distancia + 1

    Para cada palabra_vocab en vocabulario_trie:
        dist = levenshtein(palabra, palabra_vocab)
        Si dist < menor_distancia:
            menor_distancia = dist
            mejor_candidato = palabra_vocab

    Si menor_distancia <= max_distancia:
        Retornar mejor_candidato
    Sino:
        Retornar palabra  // Sin corrección posible
```

### 3.5 Optimización con Trie

El Trie permite optimizar la búsqueda de candidatos mediante **poda temprana**. Al recorrer el Trie, se puede descartar ramas completas cuando la distancia mínima posible excede el umbral máximo permitido.

---

## 4. Implementación en Python

### 4.1 Bibliotecas Utilizadas

El sistema utiliza las siguientes bibliotecas de Python para la implementación del modelo LSI:

#### 4.1.1 scikit-learn (sklearn)

**Versión:** ≥1.5.0
**Licencia:** BSD-3-Clause
**Repositorio oficial:** https://github.com/scikit-learn/scikit-learn
**Documentación:** https://scikit-learn.org/stable/

Módulos utilizados:

| Clase | Módulo | Propósito |
|-------|--------|-----------|
| `TfidfVectorizer` | `sklearn.feature_extraction.text` | Construcción de la matriz TF-IDF |
| `TruncatedSVD` | `sklearn.decomposition` | Descomposición en valores singulares truncada |
| `cosine_similarity` | `sklearn.metrics.pairwise` | Cálculo de similitud coseno entre vectores |

**TfidfVectorizer** [4] convierte una colección de documentos de texto crudo en una matriz TF-IDF con las siguientes características:

- Tokenización configurable
- Eliminación de stopwords
- Control de frecuencia mínima/máxima de documentos
- Normalización L2 opcional

**TruncatedSVD** [4] implementa SVD truncada mediante el algoritmo `randomized_svd`, que es más eficiente que la SVD completa para matrices dispersas de gran tamaño.

#### 4.1.2 NumPy

**Versión:** ≥1.26.0
**Licencia:** BSD-3-Clause
**Repositorio oficial:** https://github.com/numpy/numpy
**Documentación:** https://numpy.org/doc/stable/

NumPy proporciona:

- Arrays n-dimensionales eficientes (`ndarray`)
- Operaciones de álgebra lineal optimizadas
- Broadcasting para operaciones vectorizadas

#### 4.1.3 joblib

**Versión:** ≥1.4.0
**Licencia:** BSD-3-Clause
**Repositorio oficial:** https://github.com/joblib/joblib
**Documentación:** https://joblib.readthedocs.io/

joblib se utiliza para la serialización eficiente de:

- El vectorizador TF-IDF entrenado
- Las matrices U, Σ, V de la descomposición SVD
- El índice de documentos

### 4.2 Arquitectura de Clases

```
modules/retriever/
├── __init__.py
├── service.py          # LSIRetriever - Capa de servicio
├── lsi_model.py        # LSIModel - Núcleo del modelo LSI
└── spell_checker.py    # TrieSpellChecker - Corrector ortográfico
```

#### 4.2.1 Clase LSIModel

**Ubicación:** `modules/retriever/lsi_model.py`

**Responsabilidades:**

1. Entrenar el modelo TF-IDF + SVD sobre un corpus de documentos
2. Proyectar consultas al espacio latente
3. Calcular similitudes y recuperar documentos relevantes
4. Persistir y cargar el modelo entrenado

**Parámetros de configuración:**

| Parámetro | Tipo | Valor por defecto | Descripción |
|-----------|------|------------------|-------------|
| `n_components` | int | 100 | Número de dimensiones latentes (*k*) |
| `max_features` | int | 10,000 | Tamaño máximo del vocabulario |
| `min_df` | int | 1 | Frecuencia mínima de documento para incluir un término |
| `max_df` | float | 0.95 | Frecuencia máxima de documento (elimina términos muy comunes) |
| `random_state` | int | 42 | Semilla para reproducibilidad |

**Métodos principales:**

```python
def fit(self, documents: Sequence[Document]) -> None:
    """Entrena el modelo TF-IDF + SVD sobre el corpus."""

def project_query(self, query_text: str) -> np.ndarray:
    """Proyecta una consulta al espacio latente de k dimensiones."""

def retrieve(self, query_vector: np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
    """Retorna los índices y scores de los top_k documentos más similares."""

def save(self, model_dir: str | Path) -> None:
    """Persiste el modelo en disco."""

@classmethod
def load(cls, model_dir: str | Path) -> "LSIModel":
    """Carga un modelo previamente guardado."""
```

#### 4.2.2 Clase LSIRetriever

**Ubicación:** `modules/retriever/service.py`

**Responsabilidades:**

1. Implementar la interfaz `BaseRetriever` definida en `core/interfaces.py`
2. Integrar el modelo LSI con el corrector ortográfico
3. Orquestar el flujo completo de recuperación

**Flujo de recuperación:**

```
Consulta del usuario
       ↓
Normalización de texto
       ↓
Corrección ortográfica (Trie)
       ↓
Vectorización TF-IDF
       ↓
Proyección al espacio latente
       ↓
Cálculo de similitud coseno
       ↓
Ranking de documentos
       ↓
Retorno de resultados
```

#### 4.2.3 Clase TrieSpellChecker

**Ubicación:** `modules/retriever/spell_checker.py`

**Responsabilidades:**

1. Construir el Trie a partir del vocabulario del corpus
2. Verificar existencia de palabras
3. Sugerir correcciones basadas en distancia de Levenshtein

**Métodos principales:**

```python
def __init__(self, vocabulary: list[str]) -> None:
    """Construye el Trie insertando todas las palabras del vocabulario."""

def correct_word(self, word: str, max_distance: int = 2) -> str:
    """Retorna la palabra del vocabulario más cercana dentro del umbral de distancia."""
```

### 4.3 Ejemplo de Uso

```python
from core.models import Query, Document
from modules.retriever import LSIRetriever

# Crear documentos del corpus médico
documents = [
    Document(doc_id="1", text="La hipertensión arterial es una enfermedad crónica..."),
    Document(doc_id="2", text="La diabetes mellitus tipo 2 afecta el metabolismo..."),
    Document(doc_id="3", text="El asma es una enfermedad respiratoria crónica..."),
]

# Inicializar y entrenar el retriever
retriever = LSIRetriever(n_components=50, max_spell_distance=2)
retriever.fit(documents)

# Realizar una consulta (con error ortográfico intencional)
query = Query(text="sintomas de hipertencion")
results = retriever.retrieve(query, top_k=3)

# Los resultados incluyen documentos ordenados por relevancia
for result in results:
    print(f"Doc {result.document.doc_id}: score={result.score:.4f}")
```

---

## 5. Ventajas y Limitaciones del Modelo LSI

### 5.1 Ventajas

1. **Captura de semántica latente:** Identifica relaciones conceptuales entre términos basándose en patrones de co-ocurrencia.

2. **Reducción de dimensionalidad:** Comprime el espacio de miles de términos a cientos de conceptos, acelerando las búsquedas.

3. **Robustez a sinonimia:** Términos que aparecen en contextos similares quedan cerca en el espacio latente.

4. **Reducción de ruido:** Al descartar componentes de menor varianza, se eliminan correlaciones espurias.

5. **Interpretabilidad matemática:** Basado en álgebra lineal bien comprendida con garantías teóricas.

### 5.2 Limitaciones

1. **Complejidad computacional de SVD:** La descomposición inicial es O(min(*m*²*n*, *mn*²)) para SVD exacta, aunque SVD truncada randomizada es más eficiente.

2. **Modelo estático:** Agregar nuevos documentos requiere recalcular la SVD completa (o usar técnicas de actualización incremental).

3. **No maneja polisemia explícitamente:** Una palabra con múltiples significados tendrá una única representación agregada.

4. **Sensibilidad al parámetro *k*:** Requiere ajuste empírico para cada corpus.

5. **Limitaciones frente a modelos neuronales:** Los embeddings de modelos como BERT capturan contexto a nivel de oración, mientras que LSI opera a nivel de documento.

---

## 6. Integración en el Sistema SRI

### 6.1 Arquitectura Pipeline

El modelo LSI se integra como el módulo `retriever` dentro de la arquitectura Pipeline + Microkernel del sistema:

```
┌──────────────────────────────────────────────────────────┐
│                      Pipeline SRI                         │
├──────────────────────────────────────────────────────────┤
│  [Consulta] → [Parser] → [Retriever (LSI)] → [Ranker]    │
│                              ↓                            │
│                    [Corrección ortográfica]               │
│                    [Vectorización TF-IDF]                 │
│                    [Proyección SVD]                       │
│                    [Similitud coseno]                     │
├──────────────────────────────────────────────────────────┤
│                    → [RAG] → [Respuesta]                  │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Hooks de Plugins

El módulo `retriever` expone hooks que permiten a los plugins modificar el comportamiento:

| Hook | Momento | Uso típico |
|------|---------|------------|
| `pre_retrieval` | Antes de la vectorización | Expansión de consulta con sinónimos médicos |
| `post_retrieval` | Después del ranking inicial | Re-ranking con señales adicionales |

---

## 7. Referencias Bibliográficas

[1] C. D. Manning, P. Raghavan, and H. Schütze, *Introduction to Information Retrieval*. Cambridge University Press, 2008. [En línea]. Disponible: https://nlp.stanford.edu/IR-book/

> Libro de referencia estándar en recuperación de información. Cubre el modelo de espacio vectorial (Cap. 6), TF-IDF (Cap. 6.2), similitud coseno (Cap. 6.3), y Latent Semantic Indexing (Cap. 18). Disponible gratuitamente en línea.

[2] S. Deerwester, S. T. Dumais, G. W. Furnas, T. K. Landauer, and R. Harshman, "Indexing by latent semantic analysis," *Journal of the American Society for Information Science*, vol. 41, no. 6, pp. 391–407, 1990. DOI: 10.1002/(SICI)1097-4571(199009)41:6<391::AID-ASI1>3.0.CO;2-9

> Paper fundacional de LSI. Introduce la aplicación de SVD truncada para capturar semántica latente en colecciones de documentos. Describe la proyección de consultas al espacio latente.

[3] V. I. Levenshtein, "Binary codes capable of correcting deletions, insertions, and reversals," *Soviet Physics Doklady*, vol. 10, no. 8, pp. 707–710, 1966.

> Define la distancia de edición (distancia de Levenshtein), métrica fundamental para la corrección ortográfica utilizada en el corrector basado en Trie.

[4] F. Pedregosa *et al.*, "Scikit-learn: Machine Learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011. [En línea]. Disponible: https://jmlr.org/papers/v12/pedregosa11a.html

> Paper de referencia de scikit-learn, la biblioteca utilizada para la implementación de TF-IDF (`TfidfVectorizer`) y SVD truncada (`TruncatedSVD`).

---

## 8. Anexos

### Anexo A: Configuración de Dependencias

**Archivo:** `requirements.txt`

```
numpy>=1.26.0
scikit-learn>=1.5.0
joblib>=1.4.0
```

### Anexo B: Enlaces a Documentación Oficial

| Biblioteca | Documentación |
|------------|---------------|
| scikit-learn | https://scikit-learn.org/stable/documentation.html |
| NumPy | https://numpy.org/doc/stable/ |
| joblib | https://joblib.readthedocs.io/en/stable/ |

---

*Documento generado para el proyecto ShealtRI-YourHealthWiki.*
*Sistema de Recuperación de Información — Dominio de Salud y Medicina.*
