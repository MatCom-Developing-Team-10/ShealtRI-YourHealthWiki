# SRI Project — Sistema de Recuperación de Información (Dominio: Salud y Medicina)

## Resumen del proyecto

Sistema de recuperación de información para el dominio de salud y medicina, desarrollado como proyecto integrador del curso SRI 2025-2026. Equipo de 3 personas, 3 cortes de evaluación.

El sistema permite a usuarios realizar consultas en lenguaje natural sobre temas médicos y recupera documentos relevantes usando LSI (Latent Semantic Indexing), con generación de respuestas enriquecidas vía RAG.

## Arquitectura

**Pipeline + Microkernel empaquetado como Monolito Modular.**

- El flujo de una consulta es un **pipeline**: entrada → parseo → recuperación → ranking → RAG → respuesta.
- Los módulos opcionales se conectan como **plugins** vía hooks (`pre_retrieval`, `post_retrieval`, `post_ranking`).
- Todo corre en un **solo proceso Python** — no hay microservicios, no hay message queues.

### Estructura de carpetas

```
sri-project/src
├── core/                  # Núcleo: pipeline, interfaces ABC, registry de plugins, config
├── modules/               # Módulos obligatorios del pipeline
│   ├── crawler/           # Adquisición de datos (scraping web médica)
│   ├── indexer/           # Índices invertidos + procesamiento de texto
│   ├── retriever/         # Modelo LSI (TruncatedSVD sobre TF-IDF)
│   ├── ranker/            # Ranking y posicionamiento de resultados
│   ├── rag/               # Retrieval-Augmented Generation
│   └── web_search/        # Fallback cuando no hay resultados suficientes
├── plugins/               # Módulos opcionales (microkernel)
│   ├── expansion/         # Expansión de consultas + retroalimentación
│   └── multimodal/        # Recuperación multimodal (imágenes médicas, etc.)
├── infra/                 # Capa de datos: vector DB, embeddings, storage
├── ui/                    # Interfaz visual (Streamlit o Gradio)
├── tests/                 # Tests unitarios y de integración
├── docs/                  # Documentación LNCS en LaTeX/PDF
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Modelo de recuperación: LSI

Se usa Latent Semantic Indexing como modelo "no básico". El flujo es:

1. Construir matriz TF-IDF (términos × documentos).
2. Aplicar SVD truncado (`sklearn.decomposition.TruncatedSVD`) para obtener Uk, Σk, VkT.
3. Los vectores de documentos en el espacio latente (columnas de VkT) se almacenan en la base vectorial.
4. Para una consulta: vectorizar con TF-IDF → proyectar al espacio latente (q_proj = qT · Uk · Σk⁻¹) → similitud coseno contra documentos → ranking.

Se usa un **Trie** para corrección ortográfica de la query antes de la vectorización.

## Stack tecnológico

- **Lenguaje:** Python 3.11+
- **Modelo LSI:** scikit-learn (TruncatedSVD, TfidfVectorizer)
- **Base vectorial:** ChromaDB (preferido) o FAISS
- **Embeddings:** sentence-transformers (para RAG) + TF-IDF (para LSI)
- **RAG:** LangChain o implementación directa con API de LLM
- **UI:** Streamlit o Gradio
- **Contenedores:** Docker + docker-compose
- **Control de versiones:** Git (obligatorio)

## Convenciones de código

- Todo el código en **inglés** (para docstrings y comentarios, nombres de variables, funciones y clases).
- Cada módulo en `modules/` y `plugins/` sigue la estructura: `__init__.py`, `service.py`, archivos específicos.
- Cada módulo implementa la interfaz ABC correspondiente definida en `core/interfaces.py`.
- Los plugins implementan `Plugin` con `hook_name()` y `execute(context)`.
- Type hints en todas las funciones públicas.
- Docstrings en formato Google style.
- Nombres descriptivos, sin abreviaciones crípticas

## Patrón de módulo

Todos los módulos obligatorios del pipeline siguen este patrón:

```python
# modules/retriever/service.py
from core.interfaces import BaseRetriever
from core.models import Query, Document

class LSIRetriever(BaseRetriever):
    """Recuperador basado en Latent Semantic Indexing."""
    
    def retrieve(self, query: Query) -> list[Document]:
        # Implementación específica
        ...
```

## Patrón de plugin

Los plugins opcionales siguen este patrón:

```python
# plugins/expansion/service.py
from core.interfaces import Plugin
from core.models import PipelineContext

class QueryExpansionPlugin(Plugin):
    """Expande la consulta con sinónimos médicos."""
    
    def hook_name(self) -> str:
        return "pre_retrieval"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        # Expandir query y devolver contexto modificado
        ...
```

## Cortes de evaluación

### Corte 1 (Semana 7-8): Adquisición, indexación, recuperación básica
- crawler/ funcional con políticas de robots.txt
- indexer/ con índices invertidos
- retriever/ con LSI básico
- Base vectorial inicial con embeddings
- Dockerfile configurado desde este corte

### Corte 2 (Semana 12-13): Integración avanzada
- Mejoras al retriever LSI
- rag/ completo (recuperador + generador)
- web_search/ como fallback automático
- Plugins opcionales: expansion/, multimodal/

### Corte 3 (Semana 14-16): Sistema completo
- ui/ con interfaz visual completa
- ranker/ con posicionamiento avanzado
- Integración end-to-end
- Plugins opcionales: recomendación, evaluación (P, R, F1, NDCG)

## Documentación

- Formato LNCS (Lecture Notes in Computer Science) en LaTeX.
- Se entrega en PDF en el repositorio.
- Cada corte actualiza y expande la documentación previa.

## Control de versiones (Git)

### Estrategia de ramas

El repositorio usa tres ramas permanentes:

- **main**: Código estable y entregable. Solo recibe merges de `dev` (para cortes) y de `hotfix/*` (para correcciones urgentes).
- **dev**: Rama de integración. Todo el desarrollo se mergea aquí primero.
- **hotfix/**: Ramas temporales para correcciones urgentes sobre `main`.

Para cualquier trabajo nuevo, Claude debe sugerir crear una rama desde `dev`:

- Features: `feature/nombre-descriptivo` (ej: `feature/lsi-retriever`, `feature/crawler-robots-txt`)
- Bug fixes: `fix/nombre-descriptivo` (ej: `fix/query-empty-crash`, `fix/tfidf-dimension-mismatch`)
- Refactors: `refactor/nombre-descriptivo` (ej: `refactor/pipeline-hooks`)

Nunca hacer commits directamente en `main` ni en `dev`. Siempre trabajar en una rama y mergear vía pull request.

### Mensajes de commit

Seguir la convención Conventional Commits. Formato:
```
tipo(alcance opcional): descripción breve en minúsculas
```

Tipos permitidos:

- `feat`: Nueva funcionalidad (ej: `feat(retriever): implementar proyección LSI de queries`)
- `fix`: Corrección de bugs (ej: `fix(crawler): respetar límite de profundidad en robots.txt`)
- `docs`: Cambios en documentación (ej: `docs: agregar sección de arquitectura al informe LNCS`)
- `refactor`: Reestructuración sin cambiar funcionalidad (ej: `refactor(core): extraer hooks a clase separada`)
- `test`: Agregar o modificar tests (ej: `test(retriever): agregar test de sinonimia LSI`)
- `chore`: Tareas de mantenimiento (ej: `chore: actualizar requirements.txt`)
- `style`: Formato, espacios, imports — sin cambio lógico (ej: `style: aplicar black a modules/`)
- `ci`: Cambios en Docker o configuración de entorno (ej: `ci: agregar volumen persistente para ChromaDB`)

Reglas:
- La descripción va en minúsculas, sin punto final, en **inglés**.
- Si el cambio afecta un módulo específico, incluir el alcance entre paréntesis.
- Un commit = un cambio lógico. No mezclar feat + fix en el mismo commit.

## Comandos útiles

```bash
# Levantar el sistema
docker-compose up --build

# Ejecutar tests
python -m pytest tests/ -v

# Ejecutar solo el pipeline (sin UI)
python -m core.pipeline --query "síntomas de hipertensión"

# Reconstruir índice LSI
python -m modules.indexer.service --rebuild

# Ejecutar crawler
python -m modules.crawler.service --domain salud --depth 3
```

## Reglas importantes

- NO usar microservicios. Es un monolito modular.
- NO usar servicios en la nube (Pinecone, etc.). Todo debe ser reproducible con Docker.
- NO dejar Docker para el final. Dockerfile desde el Corte 1.
- Los plugins NO contaminan el código obligatorio. Si no se registra un plugin, el pipeline lo ignora.
- Cada módulo debe ser testeable de forma aislada usando mocks de las interfaces ABC.

## Mandamientos principales

1. **Compañerismo crítico y cognitivo**

   - Rol: Eres más que un escritor de código, eres un arquitecto senior y un compañero crítico y pensante.
   - Criticismo constructivo: Si se te pide un feature o cambio que es potencialmente inseguro, pobre, redundante o técnicamente débil, DEBES proveer críticas que ayuden y sugerir mejores alternativas ANTES de implementar.
   - Seguridad primero: Siempre aplica las mejores prácticas de seguridad. Nunca introduzcas código que exponga datos sensibles, o siga patrones de seguridad caducados.

2. **Planeo Estratégico y Aprobación**

   - Cambios No-Triviales: Para cualquier cambio que no sea una simple corrección o una pequeña adición, DEBES presentar un plan detallado primero.
   - Verificación del usuario: Usa la herramienta ask_user para presentar tu plan y esperar aprobación o modificaciones antes de proceder con la implementación.

3. **Estándares de ingeniería**

    - Documentación: EL código debe estar bien documentado (ejemplo: docstrings, comentarios para lógica compleja). Actualiza la documentación en Markdown relevante cuando sea necesario.
