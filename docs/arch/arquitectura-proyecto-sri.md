# Arquitectura recomendada para el Proyecto Integrador de SRI

## Resumen

Este documento describe la arquitectura propuesta para el sistema de recuperación de información (SRI) del proyecto integrador. La recomendación es una combinación de **Pipeline + Microkernel** empaquetada como un **monolito modular**.

---

## Arquitecturas descartadas y por qué

### Microservicios — descartada

Es la trampa más común en proyectos académicos. Son 3 personas, no un equipo de 30. La sobrecarga operacional (comunicación inter-servicios, orquestación, despliegue individual, debugging distribuido) consume más tiempo que el desarrollo real. Docker ya es requisito — meter Kubernetes o docker-compose con 8 servicios es buscar problemas.

### Clean Architecture pura (estilo Uncle Bob) — descartada

Las capas concéntricas de Entities → Use Cases → Interface Adapters → Frameworks funcionan bien para aplicaciones empresariales CRUD, pero un SRI tiene un flujo fundamentalmente diferente. Es un **pipeline**, no un request-response clásico. Forzar Clean Architecture pura lleva a crear abstracciones ceremoniales que no aportan nada.

### MVC/MVT tradicional — descartada

Demasiado plano. No captura la complejidad de los módulos internos ni la naturaleza pipeline del sistema.

---

## Arquitectura elegida: Pipeline + Microkernel como Modular Monolith

La arquitectura natural de un SRI es un **pipeline** (la consulta fluye: entrada → procesamiento → recuperación → ranking → presentación). Pero el proyecto tiene un segundo patrón claro: **módulos opcionales que se conectan o no** (expansión, multimodal, recomendación, evaluación). Eso es exactamente un **Microkernel** — un núcleo con plugins.

La combinación de ambos patrones, empaquetada como un **monolito modular** (un solo proceso, módulos bien separados), es lo óptimo para un equipo de 3 personas con entregas incrementales.

### Visión general

![Visión General](generalVision.png)

---

## Estructura de carpetas propuesta

![Estructura de carpetas propuesta](structure.png)

```
sri-project/
├── core/                          # Núcleo del sistema
│   ├── pipeline.py                ← Orquesta las etapas del pipeline
│   ├── interfaces.py              ← Contratos ABC (clases abstractas)
│   ├── registry.py                ← Registro de plugins
│   ├── config.py
│   ├── models.py
│   └── exceptions.py
│
├── modules/                       # Módulos obligatorios del pipeline
│   ├── crawler/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── models.py
│   ├── indexer/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── inverted_index.py
│   ├── retriever/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── lsi_model.py           # (o el modelo no básico elegido)
│   ├── ranker/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── strategies.py
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── service.py
│   │   └── generator.py
│   └── web_search/
│       ├── __init__.py
│       ├── service.py
│       └── fallback.py
│
├── plugins/                       # Módulos opcionales (microkernel)
│   ├── expansion/
│   └── multimodal/
│
├── infra/                         # Infraestructura / capa de datos
│   ├── vector_db.py
│   ├── embedding.py
│   ├── storage.py
│   └── database.py
│
├── ui/                            # Interfaz visual
│   ├── app.py
│   └── templates/
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## El núcleo (`core/`) — la pieza más importante

### `interfaces.py` — Contratos abstractos

Define los ABCs que cada módulo debe cumplir. Esto es lo que hace que el sistema sea extensible sin ser frágil.

```python
class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: Query) -> list[Document]: ...

class BaseRanker(ABC):
    @abstractmethod
    def rank(self, docs: list[Document], query: Query) -> list[RankedDocument]: ...

class Plugin(ABC):
    @abstractmethod
    def hook_name(self) -> str: ...  # "pre_retrieval", "post_retrieval", etc.
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext: ...
```

### `registry.py` — Registro de plugins

Un simple diccionario de plugins. El corazón del patrón Microkernel.

```python
class PluginRegistry:
    def __init__(self):
        self._plugins: dict[str, list[Plugin]] = {}
    
    def register(self, plugin: Plugin):
        hook = plugin.hook_name()
        self._plugins.setdefault(hook, []).append(plugin)
    
    def get_plugins(self, hook: str) -> list[Plugin]:
        return self._plugins.get(hook, [])
```

### `pipeline.py` — Orquestación

Recibe la consulta, la pasa por cada etapa en orden, y en cada punto de enganche (hook) pregunta al registry si hay plugins registrados. Si los hay, los ejecuta. Si no, sigue adelante.

---

## Flujo de una consulta con hooks de plugins

![Flujo](flow.png)

---

## Por qué esta arquitectura es la correcta

### 1. Se alinea con los cortes de evaluación

- **Corte 1:** Implementas `crawler/`, `indexer/`, `retriever/` y la base vectorial — son módulos independientes dentro de `modules/` que no necesitan al resto para funcionar.
- **Corte 2:** Agregas `rag/`, `web_search/`, y conectas los plugins opcionales.
- **Corte 3:** Pones la `ui/` y el `ranker/` final.

Cada corte es un incremento natural, no una reescritura.

### 2. Los plugins opcionales no contaminan el código obligatorio

Este es el beneficio más grande del patrón Microkernel. Si se decide no implementar el módulo multimodal, literalmente no se registra el plugin y el pipeline lo ignora. No hay `if multimodal_enabled:` repartidos por todo el código. La expansión de consultas se engancha en `pre_retrieval`, la recomendación en `post_ranking`, la evaluación corre como un proceso aparte que observa el pipeline sin modificarlo.

### 3. Cada módulo es testeable de forma aislada

Como todos implementan una interfaz ABC, se pueden escribir tests unitarios con mocks triviales. El módulo de evaluación puede inyectar consultas de prueba directamente al pipeline y medir métricas sin tocar la UI.

### 4. No sobrecomplica lo que no necesita complejidad

Un equipo de 3 personas no necesita event buses, message queues, ni service mesh. Un diccionario de plugins y llamadas a métodos en Python es suficiente. La sofisticación está en la separación de responsabilidades, no en la infraestructura.

---

## Consejos adicionales

### Elección de modelo de recuperación

- **LSI (Latent Semantic Indexing):** Implementación más directa con `scikit-learn` (TruncatedSVD sobre la matriz TF-IDF).
- **Modelo basado en redes neuronales:** Con `sentence-transformers` da embeddings semánticos que se integran naturalmente con la base vectorial. Más ambicioso y más coherente con el módulo RAG.

Ambos son defensibles, pero el segundo es más coherente con el módulo RAG.

### Base de datos vectorial y almacenamiento

Usar **ChromaDB** o **FAISS** para la búsqueda vectorial. ChromaDB es más fácil de usar, FAISS es más rápido. No usar Pinecone ni nada que requiera servicio en la nube — el proyecto tiene que ser reproducible con Docker.

**Importante:** Implementar una **arquitectura de almacenamiento de dos niveles**:

1. **Vector Store (ChromaDB):** Almacena solo IDs de documentos + embeddings + metadata mínima (URL)
2. **Document Store (FileSystem/DB):** Almacena el contenido completo de documentos + metadata rica

Esta separación ofrece múltiples beneficios:
- **Escalabilidad:** ChromaDB no se infla con texto completo
- **Flexibilidad:** Actualizar contenido sin re-indexar vectores
- **Rendimiento:** Cada store optimizado para su propósito
- **Mantenibilidad:** Separación clara de responsabilidades

El retriever coordina ambos stores usando recuperación en dos fases:
1. Búsqueda vectorial → retorna IDs + scores
2. Fetch de contenido completo → retorna documentos

Ver documentación detallada en [`docs/arch/almacenamiento-dos-niveles.md`](./almacenamiento-dos-niveles.md).

### Interfaz visual

**Streamlit** o **Gradio** para la UI. No meterse con React/Vue a menos que alguien del equipo realmente lo domine. El proyecto se evalúa por la calidad del SRI, no por el frontend.

### Docker

Definir el `Dockerfile` **desde el Corte 1**. No dejarlo para el final. Un `docker-compose.yml` simple con un solo servicio que levante todo es suficiente — es un monolito modular, no microservicios.
