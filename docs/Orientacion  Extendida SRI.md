# Proyecto Integrador de Sistemas de Recuperación de Información  
## Documento ampliado, explicado y reorganizado 
---

![Infografia](Infografia.png)

---
## 1. Introducción general al proyecto y al campo de los Sistemas de Recuperación de Información

Los Sistemas de Recuperación de Información (SRI) son una pieza central en la informática moderna. Están presentes en motores de búsqueda web, sistemas de recomendación, bibliotecas digitales, plataformas de e-commerce, asistentes inteligentes y prácticamente cualquier sistema que deba **buscar, filtrar y presentar información relevante** a partir de grandes volúmenes de datos.

En el contexto del curso de **Sistemas de Recuperación de Información (SRI)** , se propone un **proyecto integrador** cuyo objetivo principal es que los estudiantes diseñen e implementen un sistema completo de recuperación de información. Este sistema debe integrar de forma lógica y natural los contenidos abordados durante el curso, tanto teóricos como prácticos.
”

Se trata de construir un sistema realista que abarque:

- **Adquisición de datos** desde la web (crawling y scraping).
- **Procesamiento y normalización** de la información.
- **Indexación** eficiente de documentos y otros tipos de contenido.
- **Implementación de un modelo de recuperación no básico**.
- **Uso de bases de datos vectoriales** para representaciones numéricas.
- **Integración de RAG (Retrieval-Augmented Generation)** para respuestas enriquecidas.
- **Ranking y posicionamiento** de resultados.
- **Diseño de una interfaz visual** para consultas y presentación de resultados.
- **Módulo de búsqueda web** para ampliar la información cuando el sistema local no es suficiente.
- **Módulos opcionales** como expansión de consultas, multimodalidad, recomendación y evaluación.

Cada integrante del equipo debe conocer y ser capaz de defender el proyecto en su totalidad. La evaluación no se centra solo en el código, sino también en la comprensión conceptual, la integración de los módulos y la calidad de la documentación.

---

## 2. Panorama conceptual: ¿Qué es un Sistema de Recuperación de Información?

Un **Sistema de Recuperación de Información (SRI)** es un conjunto de métodos, algoritmos y estructuras de datos diseñado para **almacenar, indexar y recuperar información relevante** en respuesta a una consulta del usuario. Su objetivo no es devolver toda la información disponible, sino aquella que resulta **más relevante** para la necesidad expresada.

En términos generales, un SRI debe responder tres preguntas fundamentales:

1. **¿Qué información tengo?**  
   - Se relaciona con la **adquisición de datos**, el crawling, el scraping y el almacenamiento.
2. **¿Cómo represento esa información para poder buscarla eficientemente?**  
   - Involucra **indexación**, estructuras como índices invertidos, y representaciones vectoriales (embeddings).
3. **¿Cómo determino qué documentos son relevantes para una consulta?**  
   - Aquí entran los **modelos de recuperación**, el ranking y el posicionamiento de resultados.

Los SRI modernos combinan múltiples áreas de la informática:

- **Procesamiento de lenguaje natural (PLN)**.
- **Aprendizaje automático y modelos neuronales**.
- **Representaciones vectoriales y bases de datos especializadas**.
- **Modelos probabilísticos y estadísticos**.
- **Crawling y scraping web**.
- **Diseño de interfaces de usuario**.
- **Evaluación de sistemas** mediante métricas como Precision, Recall, NDCG, MRR, etc.

El proyecto integrador está diseñado para que los estudiantes vivan el ciclo completo de un SRI: desde la adquisición de datos hasta la presentación visual de resultados, pasando por la selección de un modelo de recuperación avanzado y la integración de técnicas modernas como RAG.

---

## 3. Información general del proyecto

Cuando el equipo decida comenzar a trabajar en la solución del proyecto, deberá seguir los siguientes pasos:

1. **Leer detenidamente el documento de orientación en su totalidad.**
2. **Seleccionar un dominio temático** para el sistema.
3. **Investigar a profundidad el modelo de recuperación de información no básico** seleccionado y fundamentar explícitamente la elección en relación con el dominio temático.
4. **Implementar todas las funcionalidades (obligatorias y opcionales)** según los cortes establecidos.
5. **Completar la información requerida en cada entrega.**
6. **Presentar y defender el trabajo** ante los evaluadores en cada corte y en la revisión final.

La calificación final del proyecto se determinará con base en:

- **Calidad técnica** de la implementación de cada componente (módulos imprescindibles y opcionales).
- **Integración coherente y funcional** entre los diferentes módulos.
- **Completitud y claridad de la documentación** en cada corte.
- **Solidez en la discusión y defensa** del trabajo realizado.
- **Progreso continuo y mejoras** entre cortes.
- **Justificación adecuada** de la selección de los módulos opcionales.

---

## 4. Arquitectura general del sistema

El sistema debe seguir una **arquitectura modular**, donde cada componente cumple un rol específico pero se integra con los demás para formar un flujo completo de recuperación de información.

### 4.1 Módulos imprescindibles (obligatorios)

A continuación se describen los módulos obligatorios y su propósito conceptual:

#### 4.1.1 Módulo de adquisición de datos

Responsable de **buscar información dinámica y actualizada en la web**. Incluye:

- Implementación de un **crawler** que recorra páginas del dominio seleccionado.
- Mecanismos de **scraping** para extraer contenido estructurado.
- **Políticas de crawling**: respeto a `robots.txt`, límites de profundidad, frecuencia de acceso, etc.
- **Almacenamiento inicial** de los documentos recopilados, organizados por tipo de contenido.

Este módulo responde a la pregunta: *¿Qué información tengo y de dónde proviene?*

#### 4.1.2 Módulo de indexación

Encargado de la **construcción y mantenimiento de índices** para la recuperación eficiente. Incluye:

- Construcción de **índices invertidos** u otras estructuras adecuadas.
- **Procesamiento y normalización** de los datos (tokenización, lematización, eliminación de stopwords, etc.).
- Preparación de la infraestructura para soportar **diferentes tipos de contenido**, no solo texto (si aplica).
- **Almacenamiento y recuperación eficiente** de los índices.

Este módulo transforma documentos en estructuras que permiten búsquedas rápidas y escalables.

#### 4.1.3 Módulo recuperador (modelo no básico)

Implementa un **modelo no básico de SRI** para la recuperación de documentos relevantes. Debe:

- Procesar consultas del usuario.
- Recuperar documentos relevantes según el modelo elegido.
- Proporcionar una **funcionalidad básica de ranking** de resultados (que luego se puede refinar).

Este módulo es el núcleo lógico del sistema: decide qué documentos son relevantes.

#### 4.1.4 Base de datos vectorial

Se encarga del **almacenamiento y gestión de representaciones vectoriales** de la información (embeddings). Permite:

- Búsqueda por **similitud** (por ejemplo, usando distancia coseno).
- Integración con **modelos neuronales** y técnicas modernas de IR.
- Soporte para **RAG** y, potencialmente, para **multimodalidad**.

Es un componente clave en sistemas modernos que trabajan con semántica y no solo con coincidencia de palabras.

#### 4.1.5 Módulo RAG (Retrieval-Augmented Generation)

Integra la **recuperación de información** con la **generación de respuestas**. Debe:

- Utilizar el módulo recuperador para obtener documentos relevantes.
- Pasar esos documentos a un componente generador (por ejemplo, un modelo de lenguaje).
- Generar **respuestas enriquecidas** basadas en la información recuperada.

Este módulo refleja el enfoque actual de muchos sistemas avanzados que combinan búsqueda y generación.

#### 4.1.6 Módulo de posicionamiento

Responsable de los **algoritmos de ranking y ordenamiento de resultados**, con especial atención al **posicionamiento de la información** cuando se presenta al usuario. Considera:

- Relevancia.
- Popularidad.
- Frescura.
- Tipo de contenido.
- Organización visual y experiencia de usuario.

No solo importa qué documentos se recuperan, sino **en qué orden** y **cómo se muestran**.

#### 4.1.7 Módulo de interfaz visual

Proporciona una **interfaz gráfica de usuario** donde:

- El usuario puede definir consultas (idealmente en lenguaje natural).
- Se presentan los resultados recuperados.
- Se cuida la **organización visual**, el orden de importancia y la facilidad de navegación.

Este módulo es clave para demostrar el flujo completo: desde la consulta hasta la visualización de resultados.

#### 4.1.8 Módulo de búsqueda web

Es un mecanismo para **buscar información adicional en la web** cuando el sistema detecta que **no tiene suficiente información almacenada** para responder una consulta. Debe:

- Activarse automáticamente cuando la información local es insuficiente.
- Integrarse con el módulo recuperador para incorporar los resultados web.
- Procesar e indexar la nueva información para su uso futuro.

Este módulo convierte al sistema en una plataforma **dinámica y autoexpandible**.

---

### 4.2 Módulos opcionales

Para obtener calificaciones superiores (4 o 5), es necesario implementar al menos **dos módulos opcionales**, justificados en relación con el dominio temático.

#### 4.2.1 Módulo de expansión y retroalimentación

Incluye:

- Técnicas de **expansión de consultas** (por ejemplo, usando sinónimos, términos relacionados, pseudo-relevance feedback).
- **Retroalimentación por relevancia** (explícita o implícita).
- Mecanismos para **mejorar resultados** basándose en la interacción del usuario.

#### 4.2.2 Módulo multimodal

Permite procesar y recuperar información de **diferentes tipos de contenido**:

- Texto.
- Imágenes.
- Audio.
- Video.

El sistema debe ser capaz de recuperar información que no se limite únicamente a texto, incorporando al menos **un tipo adicional de contenido** acorde al dominio.

#### 4.2.3 Módulo de recomendación

Implementa un sistema de recomendación:

- Basado en contenido.
- Colaborativo.
- Híbrido.

Considera el **perfil del usuario** y el contenido recuperado para personalizar resultados.

#### 4.2.4 Módulo de evaluación

Incluye:

- Implementación de métricas como **Precision, Recall, F1, NDCG, MRR**.
- Conjunto de consultas de prueba y relevancia asociada.
- Análisis cuantitativo del rendimiento del sistema.

---

## 5. Modelos de Recuperación de Información (no básicos)

El componente recuperador debe implementar un **modelo no básico** de recuperación de información. Las opciones válidas incluyen:

- **Modelo Booleano Difuso (Fuzzy Boolean)**.
- **Modelo Booleano Extendido (Extended Boolean)**.
- **Modelo Vectorial Generalizado (Generalized Vector Space)**.
- **Modelo de Semántica Latente (Latent Semantic Indexing - LSI)**.
- **Modelo basado en Redes Neuronales (Neural IR Models)**.
- **Modelo de Redes de Inferencia (Inference Networks)**.
- **Modelo de Redes de Creencia (Belief Networks)**.
- **Modelo Probabilístico de Lenguaje (Language Model)**.

La selección del modelo debe:

- Estar **justificada** en relación con el dominio temático.
- Incluir la **fuente bibliográfica** de donde se tomó la definición para la implementación.
- Explicar por qué ese modelo es adecuado para el tipo de documentos y consultas del sistema.

---

## 6. Dominios temáticos

El sistema debe enfocarse en un **dominio específico** que permita aplicar coherentemente todos los componentes. Algunos dominios admitidos son:

- Noticias y periodismo digital.
- Investigación científica y académica.
- E-commerce y productos.
- Turismo y viajes.
- Educación y recursos didácticos.
- Salud y medicina.
- Tecnología y software.
- Cultura y entretenimiento.

Cada equipo es responsable de:

- **Identificar, validar y gestionar sus propias fuentes de datos**.
- Asegurarse de que las fuentes sean **confiables, actualizadas y pertinentes**.
- Garantizar que la **cantidad de información recopilada e indexada** sea adecuada para el dominio.

Se recomienda que las fuentes sean **diversas y representativas** del dominio elegido.

---

## 7. Cortes de evaluación

El proyecto se evalúa en **tres cortes**, que permiten verificar el progreso y la integración de los componentes.

### 7.1 Corte 1: Adquisición, indexación y recuperación básica

**Fecha aproximada:** Semana 7–8 del curso.

#### Componentes a entregar

**Módulos imprescindibles:**

- **Módulo de crawling y scraping:**
  - Crawler funcional que recopile información del dominio.
  - Scraping para extraer contenido estructurado.
  - Políticas de crawling (robots.txt, límites de profundidad, etc.).
  - Almacenamiento inicial de documentos, organizados por tipo de contenido.

- **Módulo de indexación:**
  - Construcción de índices invertidos u otras estructuras.
  - Procesamiento y normalización de datos.
  - Preparación para soportar diferentes tipos de contenido (si aplica).
  - Almacenamiento y recuperación eficiente de índices.

- **Módulo recuperador básico:**
  - Implementación inicial del modelo no básico seleccionado.
  - Capacidad de procesar consultas y recuperar documentos relevantes.
  - Funcionalidad básica de ranking.

- **Base de datos vectorial inicial:**
  - Estructura básica para almacenar representaciones vectoriales.
  - Generación inicial de embeddings para documentos.

#### Documentación requerida

- Definición del dominio seleccionado.
- Documentación del modelo de recuperación implementado.
- Fuentes bibliográficas utilizadas.
- Estadísticas básicas del corpus (cantidad de documentos, tamaño promedio, etc.).

---

### 7.2 Corte 2: Integración avanzada y evaluación

**Fecha aproximada:** Semana 12–13 del curso.

#### Componentes a entregar

**Módulos imprescindibles:**

- **Mejoras al módulo recuperador:**
  - Refinamiento del modelo de recuperación.
  - Integración de técnicas avanzadas según el modelo seleccionado.

- **Módulo RAG completo:**
  - Implementación funcional de Retrieval-Augmented Generation.
  - Integración del recuperador con el generador.
  - Capacidad de generar respuestas enriquecidas basadas en documentos recuperados.

- **Base de datos vectorial mejorada:**
  - Refinamiento de la gestión de vectores.
  - Técnicas de búsqueda por similitud eficientes.

- **Módulo de búsqueda web:**
  - Mecanismos para buscar información en la web para ampliar la búsqueda.
  - Activación automática cuando la información almacenada es insuficiente.
  - Integración con el módulo recuperador.
  - Procesamiento e indexación de la información recuperada para uso futuro.

**Módulos opcionales (si se implementan en este corte):**

- **Módulo de expansión y retroalimentación:**
  - Técnicas de expansión de consultas.
  - Integración con el módulo de recuperación.
  - Retroalimentación por relevancia.
  - Mejora de resultados basada en la interacción del usuario.

- **Módulo de recuperación multimodal:**
  - Capacidad de procesar y recuperar información de diferentes tipos de contenido.
  - Incorporación de al menos un tipo adicional (imágenes, audio, video, etc.).
  - Integración de representaciones multimodales en la base vectorial.
  - Mecanismos de búsqueda que permitan consultas sobre contenido no textual o combinado.

#### Documentación requerida

- Actualización de la documentación técnica con las nuevas funcionalidades.
- Descripción de las mejoras implementadas y su justificación técnica.
- Justificación de la selección de los módulos opcionales en relación con el dominio.

---

### 7.3 Corte 3 (Revisión final): Sistema completo integrado

**Fecha aproximada:** Semanas 14–16 del curso.

#### Componentes a entregar

**Módulos imprescindibles:**

- **Interfaz visual del sistema:**
  - Interfaz donde el usuario define consultas en lenguaje natural.
  - Presentación de resultados con especial atención al posicionamiento de la información.
  - Diseño que facilite la comprensión y navegación de los resultados.

- **Módulo de posicionamiento:**
  - Algoritmos para ranking y ordenamiento de resultados.
  - Consideración de factores adicionales (popularidad, frescura, tipo de contenido).
  - Estrategias de posicionamiento en la presentación de resultados.
  - Mecanismos para determinar el orden y la ubicación de diferentes tipos de contenido.

- **Integración completa del sistema:**
  - Flujo end-to-end funcional desde la consulta hasta la presentación de resultados.
  - Integración coherente entre interfaz visual, módulo recuperador y módulo de posicionamiento.
  - Manejo robusto de errores.

**Módulos opcionales:**

- **Módulo de recomendación:**
  - Sistema de recomendación basado en contenido, colaborativo o híbrido.
  - Integración con el módulo recuperador para personalizar resultados.
  - Consideración del perfil del usuario y su comportamiento de búsqueda.

- **Módulo de evaluación:**
  - Implementación de métricas (Precision, Recall, F1, NDCG, MRR).
  - Conjunto de consultas de prueba y relevancia asociada.
  - Análisis cuantitativo del rendimiento del sistema.

#### Documentación requerida

- Documentación técnica completa del sistema.
- Manual sencillo de usuario o guía de uso, incluyendo descripción de la interfaz visual y cómo definir consultas.
- Descripción del diseño de la interfaz visual y las decisiones sobre posicionamiento de la información.
- Justificación de las estrategias de posicionamiento implementadas.
- Opinión crítica (bondades e insuficiencias) del proyecto.
- Lista de deficiencias detectadas y posibles soluciones.

---

## 8. Restricciones programáticas

El proyecto permite libertad de implementación, con las siguientes consideraciones:

- Se recomienda el uso de **Python** como lenguaje base.
- Se permite el uso de bibliotecas y frameworks reconocidos (**scikit-learn, transformers, langchain**, etc.).
- No se aceptan dependencias que requieran **licencias comerciales** o limiten la reproducibilidad.
- El código debe ser **ejecutable y reproducible** en un entorno estándar.
- Es obligatorio el uso de **Git** para el control de versiones.
- Es obligatorio incluir la definición de una **imagen Docker** y los pasos para el despliegue del sistema en entornos ajenos.

---

## 9. Sobre la documentación

La documentación se construye de forma **incremental** en cada corte. Debe ser:

- Clara, concisa y técnica.
- Escrita con lenguaje apropiado para el contexto académico.
- Actualizada en cada corte, reflejando el progreso del proyecto.

Debe utilizar la plantilla **LNCS (Lecture Notes in Computer Science)** como formato base, y el formato **PDF** es el que se almacena en el repositorio.

---

## 10. Otras notas importantes

- El proyecto integrador es una oportunidad para demostrar la **comprensión integral** de los contenidos del curso.
- Se valorará especialmente la **creatividad**, la **calidad técnica** y la **capacidad de integración** de los componentes.
- Cada corte se mantendrá en el repositorio del equipo con la información y requisitos solicitados.
- Se recomienda **no comenzar a última hora**: es necesario planificar el tiempo y trabajar de forma continua.

---

## 11. Conclusión crítica y panorama general

Este proyecto de Sistemas de Recuperación de Información está diseñado para que los estudiantes:

- Comprendan el ciclo completo de un SRI: desde la adquisición de datos hasta la presentación visual.
- Apliquen modelos de recuperación **no básicos**, conectando teoría y práctica.
- Integren técnicas modernas como **bases vectoriales** y **RAG**.
- Desarrollen habilidades de **ingeniería de software**, incluyendo Git, Docker y documentación académica.
- Reflexionen críticamente sobre las **bondades e insuficiencias** de su sistema.

Al finalizar, el equipo habrá construido un sistema que refleja muchos de los principios utilizados en motores de búsqueda y asistentes modernos, y habrá adquirido una visión integral de cómo se diseñan, implementan y evalúan los Sistemas de Recuperación de Información en la práctica.

![Mapa Mental](Mapa%20Mental.png)