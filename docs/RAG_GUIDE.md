# Guía del Módulo RAG con Perfiles de Usuario

## Introducción

El módulo RAG (Retrieval-Augmented Generation) de ShealtRI genera respuestas médicas **adaptadas al perfil del usuario**. Cada usuario se identifica con uno de 6 roles, y el sistema ajusta automáticamente el tono, vocabulario y nivel de detalle de la respuesta. Las respuestas son generadas por el modelo **Llama 3.1 8B** a través de la API de **Groq**, ofreciendo un tier gratuito generoso (30 req/min, ~14,400 req/día).

## Perfiles Soportados

| Perfil | Alias CLI | Tono | Vocabulario | Caso de Uso |
|--------|-----------|------|-------------|-------------|
| **Paciente** | `paciente` | Empático, tranquilizador | Lenguaje cotidiano | Personas buscando información sobre síntomas |
| **Estudiante de medicina** | `estudiante` | Didáctico, sistemático | Terminología técnica | Estudiantes aprendiendo fisiopatología |
| **Profesional médico** | `medico` | Clínico, conciso | Especializado | Médicos en toma de decisiones clínicas |
| **Diagnóstico asistido** | `diagnostico` | Estructurado | Formato diferencial | Apoyo a procesos diagnósticos |
| **Medicina natural y tradicional** | `natural` | Integrativo, respetuoso | Medicina integrativa | Usuarios interesados en enfoques alternativos |
| **Cuidador/Familiar** | `cuidador` | Compasivo, práctico | Accesible | Familiares cuidando a pacientes |

## Uso del CLI

### Modo interactivo con perfil

```bash
python cli.py --profile estudiante
```

Se mostrará el banner con el perfil seleccionado. Todas las consultas subsecuentes usarán ese perfil:

```
╔══════════════════════════════════════════════════╗
║         ShealtRI — Medical Information SRI       ║
║   Type a query, 'stats', 'help', or 'quit'       ║
╚══════════════════════════════════════════════════╝
  Perfil: Estudiante

Query> síntomas de diabetes

  Results (3 found):
  
  [0.92] Diabetes Mellitus
       http://example.com/diabetes
       Trastorno endocrino caracterizado por hiperglucemia...
  
  [0.88] Fisiopatología de la diabetes tipo 2
       http://example.com/diabetes-t2
       ...
  
  [0.85] Tratamiento farmacológico
       http://example.com/treatment
       ...

  ──── Respuesta generada [Estudiante De Medicina | llama-3.1-8b-instant] ────
  
  La diabetes mellitus es un trastorno metabólico caracterizado por
  defectos en la secreción de insulina y/o resistencia periférica...
  
  Fisiopatología:
  En la diabetes tipo 1, la destrucción autoinmune de células beta...
```

### Modo one-shot con perfil

```bash
python cli.py --query "síntomas de hipertensión" --profile paciente
```

Devuelve resultados y respuesta RAG adaptada a pacientes (lenguaje simple, énfasis en cuándo consultar al médico).

### Cambiar perfil

El alias `--profile` aceta uno de:
- `paciente` (default)
- `estudiante`
- `medico`
- `diagnostico`
- `natural`
- `cuidador`

```bash
python cli.py --profile medico --query "manejo de hipertensión resistente"
```

## Arquitectura Técnica

### Flujo de datos

```
Query + UserProfile
    ↓
LSIRetriever.retrieve()
    ↓
list[RetrievedDocument]
    ↓
RAGService.generate()
    ├─→ ProfileRegistry.get(profile_type)
    ├─→ build_context_block(docs)
    ├─→ render_prompt(profile_config, query, context)
    ├─→ _call_llm(prompt)
    │   ├─ API Success → answer_text
    │   └─ API Error → "" (fallback)
    ├─→ build_fallback_response() [si no hubo API]
    └─→ RAGResponse
```

### Componentes principales

#### `ProfileRegistry`
Registro inmutable de las 6 configuraciones de perfil. Cada perfil tiene:
- `profile_type`: Enum `UserProfileType`
- `tone`: Descripción del tono vocal
- `vocabulary_level`: Nivel de tecnicismo
- `focus_areas`: Áreas de enfoque prioritario
- `system_prompt`: Plantilla de instrucción para Groq

#### `RAGService(BaseRAG)`
Implementa la interfaz `BaseRAG`. Encargado de:
1. Resolver el perfil (default a PATIENT si no hay)
2. Construir bloque de contexto desde documentos
3. Renderizar el prompt con placeholders {query}, {context}, {focus_areas}
4. Llamar a Groq API (o fallback si no hay API key)
5. Retornar `RAGResponse` con provenance

#### `build_context_block(docs, max_docs=3)`
Construye una sección de contexto desde los documentos. Cada documento incluye:
- Título
- URL
- Snippet del texto (máx. 800 caracteres)

#### `render_prompt(profile_config, query_text, context_block)`
Renderiza el prompt usando `str.format_map` con:
- `{query}`: texto de la consulta
- `{context}`: bloque de contexto
- `{focus_areas}`: áreas de enfoque del perfil

#### `build_fallback_response(profile_config, query_text, docs)`
Genera una respuesta template-based adaptada al perfil cuando Groq no está disponible.

### Configuración

#### Variable de entorno: `GROQ_API_KEY`

La API key de Groq se configura vía:
1. Variable de entorno `GROQ_API_KEY`
2. Parámetro `api_key` en el constructor de `RAGService`

Si ninguno está seteado, el sistema usa fallback (respuestas template).

En Docker Compose:
```yaml
environment:
  - GROQ_API_KEY=${GROQ_API_KEY:-}
```

Ejecutar con API key:
```bash
export GROQ_API_KEY="gsk_..."
python cli.py --query "fiebre alta" --profile paciente
```

Para obtener una API key de Groq:
1. Visita https://console.groq.com/keys
2. Crea una cuenta gratuita o inicia sesión
3. Genera una nueva API key
4. Cópiala a tu archivo `.env` en la variable `GROQ_API_KEY`

#### Parámetros de Groq

En `RAGService._call_llm()`:
- `temperature: 0.3` — bajo para respuestas factuales
- `top_p: 0.9` — diversidad controlada
- `max_tokens: 512` — aproximadamente 400 palabras en español

## Ejemplos de Uso

### Ejemplo 1: Paciente buscando información

```bash
$ python cli.py --profile paciente
Query> síntomas de diabetes

[Retrievals con puntuación...]

  ──── Respuesta generada [Paciente | template_fallback] ────

  La diabetes es una enfermedad donde el cuerpo tiene dificultad para
  controlar el nivel de azúcar en la sangre. Los síntomas principales
  incluyen...
  
  ⚠️ Importante: Consulta a tu médico para un diagnóstico definitivo.
```

### Ejemplo 2: Estudiante de medicina

```bash
$ python cli.py --profile estudiante --query "mecanismo de acción de la metformina"

[Retrievals...]

  ──── Respuesta generada [Estudiante De Medicina | llama-3.1-8b-instant] ────

  La metformina es un agente antihiperglucemiante de la clase de las
  biguanidas. Su mecanismo principal es:

  1. Reducción de la gluconeogénesis hepática mediante inhibición
     de la cadena respiratoria mitocondrial...
```

### Ejemplo 3: Diagnóstico asistido

```bash
$ python cli.py --profile diagnostico --query "cefalea progresiva con fotofobia"

  ──── Respuesta generada [Diagnostico Asistido | llama-3.1-8b-instant] ────

  **Diagnósticos más probables:**
  1. Migraña — fotofobia y cefalea progresiva son hallazgos típicos
  2. Meningitis — fiebre + fotofobia + cefalea (riesgo alto)
  3. Sinusitis aguda — cefalea frontal/periorbitaria

  **Diagnósticos a descartar:**
  - Arteritis temporal — típicamente en >50 años
  - Glaucoma — dolor ocular más prominente

  **Estudios complementarios sugeridos:**
  - Lumbar si se sospecha meningitis
  - TAC cerebral si hallazgos neurológicos focales

  **Señales de alarma:**
  - Rigidez de nuca
  - Cambios en la conciencia
```

## Fallback: Comportamiento sin Groq API

Si `GROQ_API_KEY` no está seteada o la API falla:

1. El sistema sigue funcionando normalmente
2. Las respuestas se generan con plantillas adaptadas al perfil
3. Se incluyen los documentos recuperados en formato de lista
4. El campo `response.used_llm` es `False` y `model_name` es `"template_fallback"`

Esto asegura que el sistema **nunca crashea** por falta de API key.

## Testing

### Tests unitarios

```bash
python -m pytest tests/unit/test_rag_*.py -v
```

Cubre:
- Registro de perfiles
- Renderizado de prompts
- Construcción de contexto
- Manejo de fallback
- Resolución de perfiles

### Tests de integración

```bash
python -m pytest tests/integration/test_rag_pipeline.py -v
```

Cubre:
- Flujo end-to-end sin Groq API
- Adaptación de respuestas a perfiles
- Manejo de listas vacías
- Respeto a límites de documentos

## Ejemplos de Código

### Usar RAGService directamente

```python
from core.models import Query, UserProfile, UserProfileType, Document, RetrievedDocument
from modules.rag.service import RAGService

rag = RAGService(api_key="tu-api-key")

# Crear documentos recuperados
doc = Document(
    doc_id="d1",
    text="La diabetes es una enfermedad del metabolismo de la glucosa.",
    url="http://example.com/diabetes",
    metadata={"title": "Diabetes Mellitus"}
)
retrieved = [RetrievedDocument(document=doc, score=0.95)]

# Crear perfil y query
profile = UserProfile(
    profile_type=UserProfileType.MEDICAL_STUDENT,
    name="Estudiante"
)
query = Query(text="diabetes tipo 2", user_profile=profile)

# Generar respuesta
response = rag.generate(query, retrieved)

print(f"Respuesta: {response.answer}")
print(f"Perfil: {response.profile_type.value}")
print(f"Usó LLM: {response.used_llm}")
print(f"Modelo: {response.model_name}")
```

### Crear un perfil personalizado

El sistema viene con 6 perfiles predefinidos. Para agregar más:

1. Editar `modules/rag/user_profiles.py`
2. Agregar nuevo `UserProfileType` enum en `core/models.py`
3. Crear nueva `ProfileConfig` en `user_profiles.py`
4. Registrar en `ProfileRegistry._registry`
5. Agregar alias en `cli.py` `_PROFILE_MAP`

## Notas Importantes

1. **Sin API key**: El sistema funciona con fallback, pero sin respuestas generadas por LLM.
2. **Tasa límite de Groq**: El tier gratuito permite 30 req/min y ~14,400 req/día — muy generoso para desarrollo, testing y defensa.
3. **Lenguaje**: Todas las respuestas están en español. Las instrucciones a Groq están en español.
4. **Reproducibilidad**: Sin servicios en la nube (excepto Groq API). Todo corre localmente.

## Preguntas Frecuentes

**P: ¿Qué pasa si Groq está muy lento?**
A: El `timeout` es de 60 segundos. Si se excede, se usa fallback.

**P: ¿Puedo cambiar el modelo de Groq?**
A: Sí, pasar `model_name="mixtral-8x7b-32768"` u otro modelo disponible al constructor de `RAGService`. Los modelos disponibles están en https://console.groq.com/docs/models.

**P: ¿Cómo agrego más perfiles?**
A: Ver sección "Crear un perfil personalizado" arriba.

**P: ¿El fallback es bueno?**
A: Es funcional e informativo, pero no tan pulido como Groq. La API key es gratuita y no requiere tarjeta de crédito.
