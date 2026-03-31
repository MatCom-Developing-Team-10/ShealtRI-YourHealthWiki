---
name: create-module
description: Cómo crear un nuevo módulo o plugin para el SRI siguiendo la arquitectura Pipeline + Microkernel. Usa este skill cuando necesites agregar un módulo en modules/ o un plugin en plugins/, implementar una interfaz ABC, o registrar un componente en el pipeline. También aplica cuando necesites entender la estructura interna de un módulo existente o refactorizar uno.
---

# Crear un módulo o plugin del SRI

## Decisión inicial: ¿módulo o plugin?

- **Módulo** (`src/modules/`): Parte obligatoria del pipeline. Siempre se ejecuta. Implementa una interfaz ABC específica (`BaseRetriever`, `BaseRanker`, `BaseCrawler`, etc.).
- **Plugin** (`src/plugins/`): Opcional. Se engancha en un hook del pipeline (`pre_retrieval`, `post_retrieval`, `post_ranking`). Si no se registra, el pipeline lo ignora.

## Crear un módulo obligatorio

### 1. Crear la estructura de archivos

```
modules/nombre_modulo/
├── __init__.py          # Exporta la clase principal
├── service.py           # Clase principal que implementa la interfaz ABC
├── models.py            # Modelos de datos específicos del módulo (opcional)
└── [archivos_extra].py  # Lógica específica (ej: inverted_index.py, lsi_model.py)
```

### 2. Definir la interfaz ABC en `core/interfaces.py` (si no existe)

```python
from abc import ABC, abstractmethod
from core.models import Query, Document

class BaseNombreModulo(ABC):
    """Contrato para el módulo de [descripción]."""
    
    @abstractmethod
    def metodo_principal(self, entrada: TipoEntrada) -> TipoSalida:
        """Descripción clara de qué hace este método."""
        ...
```

Reglas para interfaces:
- Solo métodos abstractos que definan el contrato público.
- Type hints obligatorios en parámetros y retorno.
- Usar los modelos de `src/core/models.py` como tipos de entrada/salida.
- No incluir lógica de implementación.

### 3. Implementar el servicio

```python
# src/modules/nombre_modulo/service.py
from core.interfaces import BaseNombreModulo
from core.models import Query, Document

class NombreModuloService(BaseNombreModulo):
    """Implementación concreta del módulo [nombre].
    
    Responsabilidades:
    - [Lista de responsabilidades]
    """
    
    def __init__(self, config: dict = None):
        """Inicializa el módulo con configuración opcional."""
        self.config = config or {}
        # Inicializar dependencias internas
    
    def metodo_principal(self, entrada: TipoEntrada) -> TipoSalida:
        """Implementación del contrato ABC."""
        ...
```

### 4. Exportar en `__init__.py`

```python
# src/modules/nombre_modulo/__init__.py
from .service import NombreModuloService

__all__ = ["NombreModuloService"]
```

### 5. Registrar en el pipeline (`core/pipeline.py`)

El pipeline instancia el módulo y lo usa en la etapa correspondiente. El orden de ejecución es fijo:
1. Parseo/normalización de query
2. *Hook: pre_retrieval*
3. Recuperación (retriever)
4. Verificación de resultados suficientes → web_search si no
5. *Hook: post_retrieval*
6. Ranking
7. *Hook: post_ranking*
8. RAG → respuesta

## Crear un plugin opcional

### 1. Crear la estructura

```
src/plugins/nombre_plugin/
├── __init__.py
├── service.py           # Implementa Plugin ABC
└── [archivos_extra].py
```

### 2. Implementar el plugin

```python
# src/plugins/nombre_plugin/service.py
from core.interfaces import Plugin
from core.models import PipelineContext

class NombrePlugin(Plugin):
    """Plugin de [descripción]. Se engancha en [hook_name]."""
    
    def hook_name(self) -> str:
        return "pre_retrieval"  # o "post_retrieval", "post_ranking"
    
    def execute(self, context: PipelineContext) -> PipelineContext:
        """Modifica el contexto del pipeline y lo devuelve."""
        # Lógica del plugin
        # NUNCA mutar el contexto in-place sin necesidad clara
        return context
```

### 3. Registrar el plugin

```python
# En el punto de entrada de la aplicación (app.py o main.py)
from core.registry import PluginRegistry
from plugins.nombre_plugin import NombrePlugin

registry = PluginRegistry()
registry.register(NombrePlugin())
```

El plugin solo existe si se registra. Sin registro = sin efecto.

## Hooks disponibles

| Hook | Cuándo se ejecuta | Caso de uso típico |
|------|-------------------|-------------------|
| `pre_retrieval` | Antes de buscar documentos | Expansión de consulta, corrección ortográfica |
| `post_retrieval` | Después de recuperar documentos | Filtrado multimodal, feedback |
| `post_ranking` | Después del ranking | Recomendación, personalización |

## Checklist antes de dar por terminado

- [ ] La interfaz ABC está definida en `src/core/interfaces.py`
- [ ] El servicio implementa TODOS los métodos abstractos
- [ ] Type hints en todas las funciones públicas
- [ ] Docstrings in English (Google style format)
- [ ] `__init__.py` exporta la clase principal
- [ ] El módulo es testeable de forma aislada (sin dependencias duras a otros módulos)
- [ ] Si es plugin: el hook_name es uno de los válidos
- [ ] Si es plugin: execute recibe y devuelve PipelineContext
