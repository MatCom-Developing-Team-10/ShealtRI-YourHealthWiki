# Guión de Video — ShealtRI: Sistema de Recuperación de Información Médica

> **Duración objetivo:** 13-14 minutos  
> **Formato:** video continuo, sin cortes, narración en voz alta  
> **Antes de grabar:** ejecuta `python demo_reset.py --yes` para partir de cero

---

## PREPARACIÓN (antes de grabar — NO en cámara)

```bash
# 0. Activar el entorno virtual (SIEMPRE primero)
source .venv/bin/activate

# 1. Generar JSONL de los PDFs (UNA SOLA VEZ — tarda ~15 min para 20 libros)
#    Solo necesario si data/raw/ no tiene archivos .jsonl todavía
python scripts/ingest_pdfs.py

# 2. Verificar que se generaron los JSONL
python demo_status.py   # debe mostrar "JSONL: 20"
```

**Por qué este paso es previo y no en cámara:**
`ingest_pdfs.py` extrae texto de los PDFs con pypdf — tarda ~59s por libro, ~15 min en total.
Es pre-procesamiento, no indexación. El **cold start del LSI** (que sí se muestra en cámara)
lee esos JSONL ya generados. El `demo_reset.py` preserva `data/raw/` intacto, incluyendo
los JSONL, así que solo hay que correr `ingest_pdfs.py` una vez.

---

## INICIO DE LA GRABACIÓN — Reset y estado inicial

```bash
# En cámara: eliminar datos derivados (ChromaDB + document store + modelo LSI)
python demo_reset.py --yes

# Confirmar estado limpio
python demo_status.py
```

**Lo que debe verse en pantalla en este punto:**
- LSI MODEL: ✗  NOT TRAINED
- ChromaDB: ✗  EMPTY  (0 vectors)
- Document Store: ✗  EMPTY
- data/raw/: 20 PDF books, 20 JSONL

---

## SECCIÓN 1 — Dominio e indexación inicial

### [00:00 – 00:45] Introducción y Q1: Dominio temático

**Narración:**
> "Bienvenidos a la demostración de ShealtRI, nuestro Sistema de Recuperación de
> Información especializado en salud y medicina. Antes de comenzar, acabo de ejecutar
> `demo_reset.py` para limpiar todos los datos derivados. Pueden ver en pantalla que el
> modelo LSI no está entrenado, la base vectorial está vacía, y el almacén de documentos
> está en cero."
>
> "La pregunta uno: ¿qué dominio elegimos y qué características tienen los documentos?
> El dominio es ciencias biomédicas. El corpus inicial está compuesto por **20 libros de
> texto médicos en español**, incluyendo títulos clásicos como el Tratado de Fisiología de
> Guyton (12ª y 13ª edición), el Harrison Manual de Medicina, Anatomía y Fisiología de
> Saladin, Bioquímica de Curtis, y textos cubanos de Morfofisiología y Propedéutica Clínica.
> Son documentos densos, técnicos, con vocabulario especializado y estructurados por
> capítulos y páginas. En total, de esos 20 libros extrajimos aproximadamente **13.600 páginas**,
> que son los chunks que el sistema indexa."

**[Mostrar en pantalla: `ls data/raw/*.pdf`]**

```bash
ls data/raw/*.pdf | sed 's|data/raw/||' | nl
```

---

### [00:45 – 02:00] Indexación inicial (cold start)

**Narración:**
> "Ahora vamos a ver el proceso de carga del corpus. Ejecutamos `python cli.py --stats`
> y el sistema arranca en modo cold start: lee los documentos, construye el índice
> invertido, ajusta el modelo LSI, y persiste los artefactos para inicios futuros."

**[Ejecutar en terminal:]**
```bash
python cli.py --stats
```

**[Lo que verás en consola — narrar mientras aparece:]**
```
[ShealtRI] Loading pipeline...
  loading NLP model (spaCy)... done
  reading documents from data/raw/... done
  source  : data/raw/ (N docs)
  indexing... done
  fitting LSI (n_components=100)... done  [N docs, N terms]
  saving model to disk... done  (future startups will be fast)

  n_documents              : 13 XXX
  n_terms                  : X XXX
  avg_tokens_per_doc       : XX.XX
  ...
```

> "El proceso tardó unos X segundos. El sistema leyó las páginas extraídas de los PDFs,
> las tokenizó con spaCy en español, aplicó lematización y eliminación de stopwords
> médicos, y construyó el índice. Después entrenó el modelo LSI con 100 dimensiones
> latentes y guardó los artefactos en `models/lsi/`."

**[Después del cold start, verificar:]**
```bash
python demo_status.py
```

---

## SECCIÓN 2 — Modelo de recuperación

### [02:00 – 02:45] Q2: Modelo LSI

**Narración:**
> "Pregunta dos: el modelo no básico que implementamos es **LSI, Latent Semantic Indexing**.
> El flujo es: primero construimos una matriz TF-IDF con la fórmula `log(1 + tf) × IDF`
> normalizada con L2. Luego aplicamos Descomposición en Valores Singulares Truncada con
> **k = 100 componentes**, implementada con `sklearn.decomposition.TruncatedSVD`, solver
> randomizado con 7 iteraciones de potencia.
>
> ¿Por qué LSI y no un modelo booleano o vectorial clásico? LSI captura relaciones
> semánticas latentes. Si el usuario escribe 'glucemia', el sistema puede recuperar
> documentos que hablan de 'glucosa en sangre' o 'control glucémico', porque en el espacio
> latente esos conceptos están próximos. Para un dominio médico con alta sinonimia —
> fármacos con nombres genéricos y comerciales, términos en latín, abreviaciones— esta
> capacidad es crítica.
>
> La fuente bibliográfica es: **Deerwester et al., 'Indexing by Latent Semantic Analysis',
> Journal of the American Society for Information Science, 41(6), 1990**, y la
> implementación sigue la formulación de **Manning, Raghavan & Schütze,
> 'Introduction to Information Retrieval', Cambridge University Press, 2008**, cap. 18."

---

### [02:45 – 03:15] Q3: Módulos opcionales

**Narración:**
> "Pregunta tres. Implementamos cuatro módulos opcionales:
>
> **Uno — Query Expansion**: plugin `pre_retrieval` que amplía la consulta con sinónimos
> del tesauro médico estático y opcionalmente Pseudo-Relevance Feedback. Agrega hasta
> 6 términos expandidos, filtrados al vocabulario del modelo para no introducir ruido.
>
> **Dos — Ranker híbrido BM25 + LSI**: combina similitud semántica LSI (peso 0.6) con
> overlap léxico BM25 (peso 0.4). BM25 es especialmente útil para consultas con términos
> técnicos exactos como nombres de enfermedades raras.
>
> **Tres — Búsqueda web como fallback**: cuando el índice local no satisface la consulta
> (menos de 3 resultados o score máximo < 0.35), el sistema activa `InternetSearchRetriever`
> que busca en la web y enriquece los resultados.
>
> **Cuatro — Módulo de evaluación**: implementa P@k, R@k, F1@k, NDCG@k, MAP y MRR sobre
> un conjunto de 10 consultas con juicios de relevancia manuales sobre el corpus de libros."

---

## SECCIÓN 3 — Crawler

### [03:15 – 04:30] Q4: Crawler en vivo

**Narración:**
> "Pregunta cuatro. El sistema tiene un crawler web completo implementado en
> `modules/crawler/`. Vamos a ejecutarlo en vivo."

**[Ejecutar:]**
```bash
python demo_crawl.py --source medlineplus --max-pages 5 --delay 1.0
```

> "Mientras corre, observen el log. Primero, el crawler pide `robots.txt` a
> `medlineplus.gov` — el User-Agent configurado es `MedSRIBot/1.0 (proyecto académico UH)`.
> Después descarga el `sitemap.xml` de MedlinePlus, que lista miles de URLs. El crawler
> expande sitemaps anidados con una profundidad máxima de 3 niveles y evita visitar URLs
> duplicadas con un conjunto `visited_urls`.
>
> Tenemos scrapers para **tres fuentes**:
> - **MedlinePlus** (medlineplus.gov) — enciclopedia médica del NIH, en inglés
> - **NHS UK** (www.nhs.uk) — servicio nacional de salud británico
> - **Mayo Clinic** (www.mayoclinic.org) — referencia clínica con sitemaps en inglés y español
>
> Son fuentes confiables porque son instituciones médicas reconocidas y gubernamentales.
> El crawler respeta `robots.txt` mediante `RobotFileParser` de la stdlib de Python,
> aplica un delay configurable entre requests al mismo dominio, y tiene retry con backoff
> exponencial para errores 429 y 5xx.
>
> ¿La búsqueda sale del dominio semilla? **No**. La estrategia es sitemap-only: no
> seguimos ningún link de las páginas. Solo visitamos URLs declaradas en los sitemaps XML
> de cada dominio. Esto garantiza que el corpus se mantiene dentro del dominio médico de
> las fuentes elegidas y evita el crawl drift."

**[Después del crawl, mostrar el archivo JSONL generado:]**
```bash
ls -lh data/raw/*.jsonl
head -c 500 data/raw/medlineplus*.jsonl 2>/dev/null | python3 -m json.tool 2>/dev/null | head -30
```

---

## SECCIÓN 4 — Indexación y embeddings

### [04:30 – 05:20] Q5 y Q6: Documentos indexados e indexación

**Narración:**
> "Pregunta cinco: ¿cuántos documentos tenemos indexados? Vamos a verlo con el status."

**[Ejecutar:]**
```bash
python demo_status.py
```

> "Tenemos aproximadamente **13.600 vectores** en ChromaDB y **13.663 documentos** en el
> almacén de documentos. Son exclusivamente documentos de **texto** — páginas de libros
> médicos extraídas de PDF. No hay imágenes ni audio.
>
> ¿Cómo garantizamos que es un corpus suficiente y representativo? Los 20 libros cubren
> fisiología, bioquímica, anatomía, medicina interna, genética, biología molecular,
> propedéutica clínica y tratamiento de enfermedades. Son los mismos textos usados en
> la carrera de Medicina. El corpus es profundo en estas áreas, aunque no abarca
> especialidades clínicas raras —ahí es donde el fallback a web es especialmente útil."
>
> "Pregunta seis: la indexación. El flujo es:"
>
> "Primero, cada página del PDF se fragmenta en **chunks de 300 tokens con 50 tokens de
> overlap** (ventana deslizante). Luego el `TextProcessor` con spaCy en español aplica:
> normalización unicode, tokenización, lematización con POS tagging, eliminación de
> stopwords (NLTK en español más stopwords médicos personalizados como 'paciente',
> 'clínico', 'médico'), y filtrado por longitud mínima de 2 caracteres.
>
> El `IndexerService` construye un **índice invertido** en memoria: un diccionario de
> `término → [(índice_doc, frecuencia), ...]`. Los términos con frecuencia de corpus
> menor a 2 se eliminan del vocabulario. Sobre ese índice invertido, la `TfidfProcessor`
> construye la matriz dispersa TF-IDF como `scipy.sparse.csr_matrix`.
>
> La estructura dual — índice invertido para acceso exacto + TF-IDF sparse + ChromaDB
> para búsqueda por similitud — permite consultas tanto léxicas como semánticas."

---

### [05:20 – 06:15] Q7: Embeddings y base vectorial

**Narración:**
> "Pregunta siete: ¿cómo se genera un embedding para información nueva y cómo se almacena?
>
> Una aclaración importante: este sistema no usa embeddings de redes neuronales preentrenadas
> como BERT o Sentence-BERT. Los vectores de documentos **son las proyecciones LSI** —
> el resultado de descomponer la matriz TF-IDF con SVD.
>
> Para información nueva encontrada en la web, el proceso es el siguiente. Dado un nuevo
> documento, se vectoriza con el mismo `TfidfProcessor` ya ajustado (vocabulario fijo).
> Luego se proyecta al espacio latente mediante la fórmula de *folding-in*:
> `q_proj = tfidf_vector × Vk` — multiplicando por los vectores singulares derechos del SVD.
> Este vector resultante de 100 dimensiones se almacena en **ChromaDB** (colección
> `medical_documents`) con distancia coseno (`hnsw:space: cosine`).
>
> ¿Por qué coseno? Porque los vectores LSI normalizados en L2 hacen que la distancia
> coseno sea equivalente al producto punto, que es la medida de similitud semántica
> estándar en el espacio latente.
>
> ¿Por qué ChromaDB? Usa HNSW (Hierarchical Navigable Small World), una estructura de
> grafos para búsqueda aproximada de vecinos más cercanos. Es eficiente en memoria,
> permite persistencia local sin servidor, y cumple la restricción del proyecto de no
> usar servicios en la nube.
>
> El sistema detecta si los documentos incrementales superan el **20% del índice base**.
> Si es así, dispara un `rebalance` — un cold start completo que re-ajusta el SVD para
> evitar la degradación del modelo por folding-in excesivo."

---

## SECCIÓN 5 — Consulta y pipeline end-to-end

### [06:15 – 07:00] Q8: Detección de consultas no satisfacibles

**Narración:**
> "Pregunta ocho: ¿cómo detectamos que la consulta no puede ser satisfecha con el índice local?
>
> El `FallbackRetriever` evalúa **tres condiciones** después del retrieval LSI, y dispara
> el fallback web si cualquiera se cumple:
>
> **Uno** — el número de resultados es **menor a 3** (`min_results=3`).
>
> **Dos** — el score coseno del mejor resultado es **menor a 0.35** (`min_score=0.35`).
> Ese umbral fue elegido empíricamente: en el espacio LSI de 100 dimensiones scores
> por debajo de 0.35 indican que la proyección de la query cae en una región sin
> vecinos cercanos en el índice.
>
> **Tres** — la query contiene señales de recencia: años 2020–2029, palabras como
> 'reciente', 'actual', 'actualizado', 'nuevo estudio', 'nuevo tratamiento', 'vigente'.
> Nuestro corpus son libros pre-2020; si el usuario pregunta por información actual,
> el sistema lo detecta con una expresión regular y activa el fallback aunque el
> índice local devuelva resultados con score aceptable.
>
> Si cualquiera de los tres se cumple, activa automáticamente el `InternetSearchRetriever`."

---

### [07:00 – 08:00] Q9: Flujo end-to-end de una consulta

**Narración:**
> "Pregunta nueve. Vamos a ejecutar una consulta en la interfaz web y voy a narrar cada
> módulo mientras se activa."

**[Abrir la UI en el navegador — si no está corriendo:]**
```bash
uvicorn ui.app:app --host 0.0.0.0 --port 8000
```
> "Escribimos: **'síntomas y tratamiento de la diabetes tipo 2'**"

> "Módulo 1 — **TextProcessor**: spaCy tokeniza y lematiza la query, aplica corrección
> ortográfica con el Trie construido durante la indexación."
>
> "Módulo 2 — **QueryExpansionPlugin** (hook `pre_retrieval`): busca en el tesauro médico
> términos relacionados con 'diabetes'. Puede agregar 'glucemia', 'insulina', 'hiperglucemia'.
> Solo agrega términos que existen en el vocabulario del modelo."
>
> "Módulo 3 — **LSIRetriever**: vectoriza la query expandida con TF-IDF, proyecta al
> espacio latente con la fórmula de folding-in, consulta ChromaDB por los k vecinos más
> cercanos por coseno."
>
> "Módulo 4 — **FallbackRetriever**: comprueba si hay ≥ 3 resultados con score ≥ 0.35.
> Si sí, usa los resultados LSI. Si no, activa la búsqueda web."
>
> "Módulo 5 — **HybridRanker** (hook entre `post_retrieval` y `post_ranking`): recalcula
> scores combinando LSI (60%) + BM25 (40%) y reordena los resultados."
>
> "Módulo 6 — **RAGService**: toma los 3 documentos mejor rankeados, construye el contexto,
> y llama al modelo `llama-3.1-8b-instant` en Groq con el prompt del perfil de usuario
> seleccionado. Devuelve la respuesta enriquecida."

---

## SECCIÓN 6 — RAG

### [08:00 – 09:00] Q10: Módulo RAG en detalle

**Narración:**
> "Pregunta diez. Vamos a hacer una consulta técnica para ver el RAG en detalle."

**[Consulta sugerida en UI:]** `"función de los ribosomas en la síntesis de proteínas"`  
**[Perfil sugerido:]** `estudiante_medicina`

> "¿Cómo evitamos que el generador alucine? Tres mecanismos:
>
> **Primero**, el prompt le indica explícitamente al modelo que responda **solo** con
> información presente en los fragmentos de contexto proporcionados y que cite los
> documentos.
>
> **Segundo**, la temperatura es 0.3 — muy baja, casi determinista. El modelo tiene poco
> margen para 'inventar'.
>
> **Tercero**, si la API de Groq no está disponible, el sistema cae a una respuesta
> plantilla construida directamente de los snippets recuperados, sin LLM. Nunca retorna
> texto no fundamentado en el corpus.
>
> ¿Qué información se pasa al generador? Los **3 documentos mejor rankeados** por
> HybridRanker, cada uno truncado a 800 caracteres para no exceder el context window.
> No se pasa el resultado completo — hay un mecanismo de priorización explícito: el
> ranker BM25+LSI decide qué tres documentos entran al contexto del LLM.
>
> El sistema tiene **6 perfiles de usuario**, cada uno con su propio system prompt:
> paciente, estudiante de medicina, profesional médico, asistencia diagnóstica,
> medicina natural, y cuidador familiar."

---

## SECCIÓN 7 — Fallback web

### [09:00 – 09:50] Q11: Activar el fallback web

**Narración:**
> "Pregunta once. Vamos a provocar el caso donde el índice local no tiene suficiente
> información. Usaré el botón 'Buscar en Web' de la interfaz, o haré una consulta
> sobre un tema muy específico y reciente."

**[Opción A — forzar desde UI:]** activar toggle "Buscar en Web" antes de la consulta  
**[Opción B — consulta con CLI:]**
```bash
python cli.py --query "nuevas terapias con ARN mensajero para enfermedades cardiovasculares 2024"
```

> "El criterio para detectar 'información insuficiente' es dual: menos de 3 resultados
> locales, o score máximo por debajo de 0.35. Para consultas sobre avances recientes
> (2024, nuevas terapias, estudios recientes), el corpus de libros — que fue publicado
> antes de 2020 — no puede responder, lo que dispara automáticamente el fallback.
>
> ¿Cómo se incorporan los resultados web al ranking? Los documentos web son recuperados
> por `WebContentFetcher`, se limpian con BeautifulSoup, y se asigna un score con TF-IDF
> local sobre keywords. Después se mezclan con los resultados LSI locales y pasan por
> el `HybridRanker`.
>
> Los documentos web también se almacenan en el `FileSystemDocumentStore` y se indexan
> con folding-in al espacio LSI, de modo que futuras consultas similares ya los encuentran
> en el índice local."

---

## SECCIÓN 8 — Interfaz y posicionamiento

### [09:50 – 10:45] Q12 y Q13: Interfaz y ranking

**Narración:**
> "Pregunta doce. La interfaz es una SPA servida por FastAPI desde `ui/static/`. Las
> decisiones de diseño del posicionamiento visual son:
>
> Los resultados se muestran en **orden de score descendente** — el de mayor relevancia
> combinada BM25+LSI aparece primero. El score está normalizado [0,1] y es visible en
> cada tarjeta.
>
> **Elementos de cada resultado:**
> - Título y badge de tipo de fuente — **Local** (libros PDF indexados) o **Web**
>   (resultados del fallback de internet), con colores distintos para identificación rápida.
> - URL clickeable — tanto fuentes web como archivos PDF locales son clicables.
> - Snippet de 300 caracteres extraído de forma query-aware: el sistema desliza una
>   ventana sobre el texto y elige la posición donde más términos de la consulta aparecen.
>
> **Elementos adicionales de la UI:**
> - La respuesta RAG se renderiza como **Markdown** (usando marked.js) con formato
>   enriquecido — listas, negritas, encabezados — no texto plano.
> - **Toggle 'Buscar en Web'** — pill que fuerza el fallback web independientemente
>   del score LSI, útil para consultas sobre información reciente.
> - **Banner de fallback** — aparece automáticamente cuando el backend confirma que
>   se usó búsqueda web (`used_web_fallback: true` en la respuesta).
> - **Hint de corrección ortográfica** — si el Trie corrije la query, aparece debajo
>   del buscador: 'Buscando por: [término corregido]'.
> - **Contador y tiempo** — muestra 'X resultados en Y ms' al recibir la respuesta.
> - **Historial** — panel lateral que persiste las últimas consultas en localStorage del navegador."
>
> "Pregunta trece — factores del ranking más allá de la relevancia. Actualmente el
> HybridRanker combina dos señales: **LSI semántico (60%)** y **BM25 léxico (40%)**.
>
> Eso significa que una consulta como 'hemoglobina oxígeno transporte' favorece documentos
> con esos términos exactos (BM25 sube) aunque LSI ya los captura semánticamente. Para
> demostrar que el orden cambia, hagamos dos consultas:"

**[Consulta 1:]** `"hemoglobina"` — resultado: documentos con el término exacto en título rankean más alto por BM25  
**[Consulta 2:]** `"proteína transporte oxígeno en sangre"` — resultado: LSI domina, sube documentos sobre hemoglobina aunque no usen la palabra exacta

> "La frescura no está implementada en el ranker (los libros no tienen timestamp relevante).
> La popularidad tampoco. El perfil de usuario influye indirectamente: determina cuáles de
> los top documentos entran al contexto RAG y cómo se genera la respuesta, pero no
> reordena los resultados visibles."

---

## SECCIÓN 9 — Expansión y evaluación

### [10:45 – 11:30] Q14: Expansión de consultas

**Narración:**
> "Pregunta catorce. La expansión de consultas está implementada como plugin `pre_retrieval`.
>
> Cuando la consulta llega al pipeline, el `QueryExpansionPlugin` la procesa antes de
> pasarla al retriever. Por ejemplo, para la query 'hipertensión':
> el tesauro médico la expande con términos como 'presión alta', 'tensión arterial'.
> Solo agrega términos que ya existen en el vocabulario TF-IDF (para no inventar vectores
> fuera del espacio latente). El máximo son 6 términos expandidos.
>
> El plugin también puede usar Pseudo-Relevance Feedback (PRF): hace una primera
> búsqueda con la query original, toma los 3 mejores resultados, extrae sus términos
> más discriminativos, y los agrega a la query. Esta opción está disponible pero
> desactivada por defecto (`use_prf=False`) para evitar latencia adicional."

**[Demostrar con CLI para ver metadata de expansión:]**
```bash
python cli.py --query "hipertensión arterial" --profile medico
```
_(La metadata de expansión se registra en `context.metadata["expansion"]` — puede ser visible en logs DEBUG)_

---

### [11:30 – 12:30] Q15: Evaluación cuantitativa

**Narración:**
> "Pregunta quince. Vamos a correr la evaluación formal."

**[Ejecutar — puede tardar 30-60 segundos:]**
```bash
python cli.py --eval --eval-k 10
```

**[Opcional — con HybridRanker para comparar:]**
```bash
python cli.py --eval --eval-k 10 --rerank
```

> "El sistema tiene 10 consultas de prueba con juicios de relevancia manuales que
> cubrimos nosotros mismos sobre el corpus de libros: insulina y glucosa, metabolismo de
> glucosa, potencial de acción, hemoglobina, mutaciones genéticas, ciclo de Krebs,
> fotosíntesis, anticuerpos, presión arterial, síntesis de proteínas.
>
> Las métricas reportadas son:
> - **P@10**: precisión — fracción de los 10 resultados que son relevantes
> - **R@10**: recall — fracción de los relevantes totales que encontramos en los 10
> - **F1@10**: media armónica de P y R
> - **NDCG@10**: Normalized Discounted Cumulative Gain — considera la posición: un
>   resultado relevante en posición 1 vale más que en posición 10
> - **MAP**: Mean Average Precision — precisión promedio sobre todos los rangos relevantes
> - **MRR**: Mean Reciprocal Rank — inversa del rango del primer resultado relevante
>
> Los resultados del LSI puro son:
> **P@10 = 0.32 · R@10 = 0.21 · NDCG@10 = 0.33 · MAP = 0.20 · MRR = 0.77**
>
> Con LSI + BM25 (`--rerank`):
> **P@10 = 0.30 · R@10 = 0.20 · NDCG@10 = 0.31 · MAP = 0.19 · MRR = 0.72**
>
> El reranker híbrido no mejora en este conjunto de prueba porque las 10 queries son
> semánticas — 'potencial de acción', 'ciclo de Krebs', 'síntesis de proteínas'. LSI
> captura esas relaciones latentes directamente. BM25 penaliza documentos que usan
> variantes terminológicas ('potencial de membrana', 'ciclo del ácido cítrico',
> 'traducción ribosomal') sin contener el término exacto de la query. El reranker
> híbrido sería más útil con queries de términos técnicos exactos como nombres de
> fármacos o enfermedades raras, que no están en este conjunto de evaluación.
>
> La única query con P@10 = 0.00 es fotosíntesis: el retriever devuelve páginas sobre
> botánica general en lugar de los capítulos específicos de fotosíntesis. Esto refleja
> una limitación real del modelo LSI con ese término en particular."

---

## SECCIÓN 10 — Autocrítica

### [12:30 – 13:30] Q16: Deficiencias y mejoras

**Narración:**
> "Pregunta dieciséis — deficiencias detectadas y qué haríamos diferente.
>
> **Deficiencia 1 — Sin ranking por frescura**: los documentos del corpus son libros
> de texto de hasta 10 años de antigüedad. Para consultas sobre tratamientos actuales,
> el sistema no penaliza documentos desactualizados. La solución: añadir un factor de
> decay temporal en el HybridRanker usando la fecha de publicación del documento.
>
> **Deficiencia 2 — Corpus monolingüe**: los libros están en español cubano y los scrapers
> web cubren sitios en inglés (NHS, MedlinePlus). Hay un desajuste lingüístico que hace
> que el modelo LSI, entrenado en español, no proyecte bien los documentos web en inglés.
> La solución: añadir un paso de traducción automática o entrenar sobre un corpus
> verdaderamente bilingüe.
>
> **Deficiencia 3 — k=100 no validado con precisión**: elegimos 100 dimensiones por la
> heurística common en la literatura, pero nunca hicimos un barrido sistemático de k
> para maximizar NDCG en nuestro corpus. Podría ser que k=150 o k=50 sea mejor. Un
> ejemplo concreto es la query 'fotosíntesis', donde el sistema obtiene P@10 = 0.00:
> el término no está bien representado en el espacio latente de 100 dimensiones con
> nuestro corpus, y un k mayor o un corpus más balanceado podría corregirlo.
>
> **Deficiencia 4 — Folding-in degrada el modelo**: cuando se agregan muchos documentos
> web por folding-in sin re-ajustar el SVD, los nuevos documentos no están bien
> representados en el espacio latente. El umbral de rebalance al 20% mitiga esto, pero
> no es una solución perfecta. Idealmente, usaríamos SVD incremental o embeddings
> neuronales que no requieran re-fit completo.
>
> **¿Qué haríamos diferente desde cero?**
> Usaríamos embeddings densos con un modelo preentrenado en español biomédico (como
> `PlanTL-GOB-ES/roberta-base-biomedical-clinical-es`) en lugar de LSI, lo que daría
> mejor semántica sin el compromiso de la dimensionalidad fija. Separado del sistema de
> recuperación tendríamos una capa de re-ranking neuronal (cross-encoder) para el
> top-k de resultados. Y el corpus habría sido curado desde el inicio mezclando libros
> con artículos científicos recientes de PubMed en español."

---

## CIERRE

**Narración:**
> "Con esto concluye la demostración de ShealtRI. El sistema integra un pipeline completo
> de recuperación de información médica: crawler web, indexación con LSI, ranking híbrido,
> expansión de consultas, RAG con perfiles de usuario, fallback a web, y evaluación
> cuantitativa. Gracias."

---

## CHEATSHEET DE COMANDOS PARA EL DÍA DE LA GRABACIÓN

```bash
# ── ANTES DE GRABAR (solo una vez) ────────────────────────────────────
source .venv/bin/activate
python scripts/ingest_pdfs.py          # extrae PDFs → JSONL (~15 min, solo 1 vez)

# ── EN CÁMARA ─────────────────────────────────────────────────────────
# Estado del sistema
python demo_status.py

# RESET (eliminar ChromaDB + document store + modelo LSI)
python demo_reset.py --yes

# Indexar corpus desde cero (cold start — esperar ~8-10 min con los 20 libros)
python cli.py --stats

# Consulta simple
python cli.py --query "síntomas de diabetes tipo 2"

# Consulta con perfil
python cli.py --query "diabetes tipo 2 tratamiento" --profile estudiante
python cli.py --query "diabetes tipo 2 diagnóstico diferencial" --profile diagnostico
python cli.py --query "diabetes tipo 2 cuidados en casa" --profile cuidador

# Forzar fallback web (desde CLI)
python cli.py --query "terapia ARN mensajero cardiovascular 2024" --profile paciente

# Evaluación (P@k, R@k, F1, NDCG, MAP, MRR)
python cli.py --eval --eval-k 10
python cli.py --eval --eval-k 10 --rerank

# Crawler en vivo (solo 5 páginas para demo rápida)
python demo_crawl.py --source medlineplus --max-pages 5 --delay 1.0

# UI web (abrir http://localhost:8000)
uvicorn ui.app:app --host 0.0.0.0 --port 8000

# Logs del crawler (verbose)
python demo_crawl.py --source mayo --max-pages 3
```

---

## NOTAS DE REVISIÓN CRÍTICA

> ⚠️ **Para discutir antes de la grabación:**

1. **El crawler y el corpus no son la misma cosa.** El corpus actual viene de PDFs
   preprocesados. El crawler web (MedlinePlus/NHS/Mayo) es para adquisición incremental.
   En la pregunta 4, debes ser transparente: mostrar el crawler funcionando EN VIVO
   y explicar que el corpus inicial fue cargado desde los libros PDF, no desde el crawler.
   Esto es válido y correcto — la pregunta pide mostrar cómo funciona el crawler, no que
   todo el corpus haya venido de él.

2. **La pregunta 7 habla de "embeddings"** — en este sistema los "embeddings" son las
   proyecciones LSI, no vectores neuronales. Usa esa terminología en la defensa: "el
   vector de representación distribuida que usamos es la proyección en el espacio latente
   LSI, análogo funcional a un embedding neuronal".

3. **La pregunta 13 pide dos consultas donde el orden cambie por factores adicionales.**
   Con el ranker actual (solo LSI+BM25), el factor que varía es el balance semántico vs
   léxico. Busca una consulta donde un documento con el término exacto en el título
   sube con BM25, versus una consulta semántica donde ese mismo doc baja porque LSI
   encuentra mejores candidatos.

4. **No tienes implementada popularidad ni frescura en el ranker.** Admítelo en la
   pregunta 13 y en la 16. Es mejor ser honesto que inventar algo que no está en el código.
