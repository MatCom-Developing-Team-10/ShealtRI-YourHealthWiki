# Selección del Modelo de IR para el Dominio Salud y Medicina

## Conclusión

Para construir un sistema que sea:

- Coherente con la arquitectura del proyecto (RAG + base vectorial + búsqueda web).
- Capaz de manejar sinónimos, abreviaturas y contexto médico.
- Alineado con el estado del arte en IR.
- Específicamente adaptado al dominio Salud y Medicina,


**El modelo más apto para el dominio Salud y Medicina es un modelo de Recuperación de Información basado en Redes Neuronales (Neural IR), idealmente combinado con un Modelo de Lenguaje para RAG.**

## Justificación

- El dominio médico exige alta capacidad semántica y manejo de terminología compleja.
- Los modelos neuronales permiten representaciones vectoriales ricas, integrables con la base vectorial.
- La arquitectura del proyecto ya incluye RAG, lo que encaja perfectamente con un enfoque Neural IR + LM.
- Frente a modelos clásicos (booleanos, vectoriales, LSI), los modelos neuronales ofrecen una mejor adaptación al lenguaje médico y a la diversidad de fuentes.

## Análisis y Comparación de Modelos de IR

### 1. Particularidades del dominio Salud y Medicina

El dominio de Salud y Medicina tiene características que condicionan fuertemente la elección del modelo de IR:

- **Lenguaje altamente especializado**: términos técnicos, abreviaturas (ECG, COPD, DM2), nombres de fármacos, procedimientos, síndromes.
- **Sinónimos y variantes terminológicas**: “infarto de miocardio” vs “ataque al corazón”; “hipertensión arterial” vs “presión alta”.
- **Alta sensibilidad semántica**: pequeñas diferencias en términos pueden implicar conceptos muy distintos (benigno vs maligno, agudo vs crónico).
- **Necesidad de precisión y seguridad**: recuperar documentos irrelevantes o ambiguos puede ser problemático.
- **Fuentes heterogéneas**: artículos científicos, guías clínicas, blogs médicos, noticias de salud, fichas de medicamentos, etc.

Esto implica que el modelo debe:
- Manejar sinónimos, variaciones léxicas y contexto.
- Ser capaz de trabajar con texto largo y denso (artículos, guías).
- Integrarse bien con embeddings y bases vectoriales.
- Ser compatible con un enfoque RAG (muy natural en este dominio).

### 2. Modelos clásicos y sus limitaciones

#### 2.1 Booleano Difuso / Booleano Extendido

| Ventajas | Problemas en Salud/Medicina |
|----------|------------------------------|
| Interpretabilidad, fácil explicación | No captan sinónimos ni relaciones semánticas |
| Permiten consultas con operadores lógicos y pesos | Lenguaje médico rico en variantes y abreviaturas |
| Útiles si el usuario es experto y formula consultas estructuradas | Coincidencia pobre sin capa adicional de expansión |

**Veredicto**: Útiles como capa lógica adicional o para filtros (ej. “tipo: ensayo clínico AND año > 2018”), pero no como modelo principal de relevancia en este dominio.

#### 2.2 Modelo Vectorial Generalizado

| Ventajas | Problemas en Salud/Medicina |
|----------|------------------------------|
| Extiende el modelo vectorial clásico con funciones de similitud flexibles | Muy dependiente de coincidencia de términos |
| Fácil implementación y explicación | No maneja sinónimos, abreviaturas, polisemia sin capas adicionales (tesauros, ontologías) |
| Buena base para un primer corte del proyecto | En dominio médico, “aspirina”, “ácido acetilsalicílico” y “AAS” deben ser reconocidos como equivalentes |

**Veredicto**: Aceptable como punto de partida, pero insuficiente como modelo final para explotar el dominio médico.

#### 2.3 LSI (Latent Semantic Indexing)

| Ventajas | Problemas en Salud/Medicina |
|----------|------------------------------|
| Introduce semántica latente mediante reducción de dimensionalidad | Escala regular para colecciones grandes y dinámicas |
| Puede agrupar términos relacionados en un espacio reducido | Superado por embeddings neuronales (word2vec, GloVe, BERT) |
| | La semántica médica es muy rica; LSI se queda corto frente a modelos preentrenados biomédicos |

**Veredicto**: Interesante como ejercicio académico, pero no es la mejor elección para un sistema competitivo y moderno en Salud/Medicina.

### 3. Modelos probabilísticos y estructurados

#### 3.1 Modelos de Lenguaje (Language Models clásicos)

| Ventajas | Limitaciones |
|----------|--------------|
| Basados en probabilidad de generar la consulta dado el documento | En su forma clásica (sin embeddings neuronales), limitados en semántica |
| Base teórica sólida, usados históricamente en IR | No aprovechan directamente modelos preentrenados biomédicos |
| Se integran bien con smoothing y colecciones | |

**Veredicto**: Buenos como base probabilística, pero hoy su mejor versión es cuando se combinan con modelos neuronales de lenguaje.

#### 3.2 Redes de Inferencia / Redes de Creencia

| Ventajas | Problemas |
|----------|-----------|
| Modelan dependencias entre términos, conceptos y documentos | Complejas de diseñar y mantener para un dominio amplio |
| Podrían integrar conocimiento médico estructurado (ontologías) | Difíciles de escalar y ajustar con datos reales |
| | Coste de diseño alto frente al beneficio en un proyecto de curso |

**Veredicto**: Conceptualente atractivas, pero poco prácticas como modelo central en un proyecto integrador con tiempo limitado.

### 4. Modelos Neuronales (Neural IR) en Salud y Medicina

Aquí es donde el dominio Salud/Medicina brilla con los modelos adecuados.

#### 4.1 Ventajas clave
- **Captura de semántica profunda**: modelos tipo BERT, BioBERT, ClinicalBERT, PubMedBERT captan relaciones entre términos médicos, sinónimos, abreviaturas y contexto.
- **Robustez a variaciones léxicas**: “ataque al corazón”, “infarto de miocardio”, “IAM” pueden quedar cerca en el espacio vectorial.
- **Integración natural con bases vectoriales**: embeddings de documentos y consultas se almacenan en la base vectorial y se realiza búsqueda por similitud.
- **Compatibilidad directa con RAG**: el mismo tipo de representaciones se puede usar para recuperar y luego alimentar al generador.

#### 4.2 Riesgos y desafíos
- Requieren más recursos computacionales que modelos clásicos.
- Es necesario cuidar la evaluación: un modelo que “suena bien” puede recuperar cosas semánticamente cercanas pero no clínicamente relevantes.
- Hay que ser muy claro en la documentación: el sistema no sustituye criterio médico ni es un sistema de soporte clínico oficial.

#### 4.3 Por qué encajan tan bien con el proyecto
- El proyecto exige una base vectorial y un módulo RAG: los modelos neuronales son el pegamento natural entre ambos.
- El dominio Salud/Medicina se beneficia enormemente de embeddings especializados.
- Permiten explotar mejor la búsqueda web: se pueden indexar documentos externos y seguir usando el mismo espacio vectorial.

**Veredicto**: Para el dominio Salud y Medicina, un modelo de IR basado en redes neuronales (Neural IR) es, en la práctica, la opción más potente y coherente con la arquitectura del proyecto.

### 5. Tabla comparativa resumida

| Modelo | Ventajas clave en Salud/Medicina | Riesgos / Limitaciones | Idoneidad global |
|--------|----------------------------------|------------------------|------------------|
| Booleano Difuso / Extendido | Control lógico fino, interpretabilidad | Poca capacidad semántica, vocabulario complejo | Media–Baja |
| Vectorial Generalizado | Simplicidad, buena base clásica | No capta bien sinónimos ni contexto clínico | Media |
| LSI | Capta cierta semántica latente | Obsoleto frente a embeddings modernos | Media–Baja |
| Redes de Inferencia / Creencia | Modelan dependencia entre términos y conceptos | Complejos de diseñar y escalar | Media |
| Modelo de Lenguaje (LM) | Probabilístico, bien alineado con texto médico | Si es clásico, limitado frente a modelos neuronales | Alta (si se combina) |
| **Modelos Neuronales (Neural IR)** | **Capturan semántica, sinónimos, contexto, abreviaturas** | **Requieren más recursos y cuidado en evaluación** | **Muy Alta** |
