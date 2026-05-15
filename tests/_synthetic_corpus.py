"""Synthetic Spanish medical documents for testing and local development.

These 20 documents cover major medical topics and are used by:
    - tests/conftest.py  (pytest fixtures)
    - cli.py             (fallback corpus when data/raw/ is empty)
"""

from __future__ import annotations

RAW_DOCUMENTS: list[dict] = [
    {
        "doc_id": "doc_hipertension_001",
        "url": "https://medlineplus.gov/hipertension",
        "title": "Hipertensión Arterial",
        "text": (
            "La hipertensión arterial, también conocida como presión arterial alta, "
            "es una condición crónica en la que la fuerza de la sangre contra las paredes "
            "de las arterias es constantemente demasiado alta. La presión arterial se mide "
            "en milímetros de mercurio y se registra con dos cifras: sistólica sobre diastólica. "
            "Una lectura de 130/80 mmHg o más se considera hipertensión. Los factores de riesgo "
            "incluyen obesidad, tabaquismo, sedentarismo, dieta alta en sal y antecedentes familiares. "
            "El tratamiento incluye cambios en el estilo de vida y medicamentos como diuréticos, "
            "betabloqueantes e inhibidores de la ECA. Sin tratamiento puede causar infarto, "
            "accidente cerebrovascular e insuficiencia renal."
        ),
    },
    {
        "doc_id": "doc_diabetes_002",
        "url": "https://medlineplus.gov/diabetes",
        "title": "Diabetes Tipo 2",
        "text": (
            "La diabetes tipo 2 es una enfermedad metabólica crónica caracterizada por niveles "
            "elevados de glucosa en sangre debido a resistencia a la insulina o producción "
            "insuficiente de insulina por el páncreas. Los síntomas incluyen sed excesiva, "
            "micción frecuente, fatiga, visión borrosa y heridas que tardan en sanar. "
            "El diagnóstico se realiza mediante prueba de glucemia en ayunas o hemoglobina "
            "glucosilada HbA1c. El tratamiento incluye dieta saludable, ejercicio regular, "
            "pérdida de peso y medicamentos como metformina. Las complicaciones a largo plazo "
            "incluyen neuropatía, retinopatía, nefropatía y enfermedades cardiovasculares."
        ),
    },
    {
        "doc_id": "doc_asma_003",
        "url": "https://www.nhs.uk/conditions/asthma",
        "title": "Asma Bronquial",
        "text": (
            "El asma es una enfermedad inflamatoria crónica de las vías respiratorias que causa "
            "episodios recurrentes de sibilancias, disnea, opresión torácica y tos. "
            "Los desencadenantes comunes incluyen alérgenos como polen, ácaros del polvo, "
            "pelo de animales, infecciones respiratorias, aire frío, ejercicio físico y "
            "contaminación ambiental. El diagnóstico se confirma con espirometría y prueba "
            "de reversibilidad bronquial. El tratamiento incluye broncodilatadores de acción "
            "corta como salbutamol para alivio inmediato y corticosteroides inhalados como "
            "budesonida para control a largo plazo. El asma bien controlada permite una "
            "vida normal con actividad física regular."
        ),
    },
    {
        "doc_id": "doc_artritis_004",
        "url": "https://www.mayoclinic.org/artritis-reumatoide",
        "title": "Artritis Reumatoide",
        "text": (
            "La artritis reumatoide es una enfermedad autoinmune crónica que afecta "
            "principalmente las articulaciones, causando inflamación, dolor, rigidez y "
            "destrucción articular progresiva. A diferencia de la artrosis, afecta las "
            "articulaciones de forma simétrica y puede comprometer órganos internos. "
            "Los síntomas incluyen rigidez matutina mayor de una hora, articulaciones "
            "calientes e hinchadas, y fatiga generalizada. El diagnóstico incluye análisis "
            "de factor reumatoide y anticuerpos anti-CCP. El tratamiento con fármacos "
            "antirreumáticos modificadores de la enfermedad como metotrexato y agentes "
            "biológicos puede reducir la progresión y mejorar la calidad de vida."
        ),
    },
    {
        "doc_id": "doc_cancer_colon_005",
        "url": "https://www.cancer.org/colon",
        "title": "Cáncer de Colon y Recto",
        "text": (
            "El cáncer colorrectal es el tercer cáncer más diagnosticado en el mundo. "
            "Se origina en el revestimiento del colon o recto y generalmente comienza como "
            "pólipos benignos que pueden volverse malignos con el tiempo. Los síntomas incluyen "
            "cambios en los hábitos intestinales, sangre en las heces, dolor abdominal y "
            "pérdida de peso inexplicable. Los factores de riesgo son dieta baja en fibra, "
            "sedentarismo, obesidad, tabaquismo, consumo de alcohol y antecedentes familiares. "
            "La colonoscopia es el método de detección más efectivo. El tratamiento depende "
            "del estadio e incluye cirugía, quimioterapia y radioterapia."
        ),
    },
    {
        "doc_id": "doc_depresion_006",
        "url": "https://medlineplus.gov/depresion",
        "title": "Depresión Mayor",
        "text": (
            "La depresión mayor es un trastorno del estado de ánimo caracterizado por "
            "tristeza persistente, pérdida de interés en actividades, alteraciones del sueño "
            "y apetito, dificultad para concentrarse y en casos graves, pensamientos suicidas. "
            "Para el diagnóstico se requieren al menos dos semanas de síntomas que afecten "
            "el funcionamiento diario. Las causas incluyen factores genéticos, biológicos, "
            "psicológicos y sociales. El tratamiento combina psicoterapia cognitivo-conductual "
            "con antidepresivos como inhibidores selectivos de la recaptación de serotonina. "
            "La detección temprana y el tratamiento adecuado permiten la recuperación completa "
            "en la mayoría de los pacientes."
        ),
    },
    {
        "doc_id": "doc_osteoporosis_007",
        "url": "https://www.nhs.uk/conditions/osteoporosis",
        "title": "Osteoporosis",
        "text": (
            "La osteoporosis es una enfermedad ósea caracterizada por disminución de la "
            "densidad mineral ósea y deterioro de la microarquitectura del hueso, lo que "
            "aumenta el riesgo de fracturas. Es más común en mujeres posmenopáusicas. "
            "Los factores de riesgo incluyen déficit de calcio y vitamina D, tabaquismo, "
            "consumo excesivo de alcohol, inactividad física y uso prolongado de corticosteroides. "
            "El diagnóstico se realiza mediante densitometría ósea DXA. El tratamiento incluye "
            "suplementos de calcio y vitamina D, ejercicio de carga y bifosfonatos como alendronato. "
            "La prevención desde la infancia mediante dieta adecuada y ejercicio es fundamental."
        ),
    },
    {
        "doc_id": "doc_infarto_008",
        "url": "https://www.mayoclinic.org/infarto-miocardio",
        "title": "Infarto de Miocardio",
        "text": (
            "El infarto de miocardio ocurre cuando el flujo sanguíneo hacia una parte del "
            "corazón se bloquea durante suficiente tiempo para que el músculo cardíaco muera. "
            "Los síntomas clásicos incluyen dolor torácico intenso que puede irradiarse al brazo "
            "izquierdo, mandíbula y espalda, sudoración fría, náuseas y disnea. Es una emergencia "
            "médica que requiere atención inmediata. El diagnóstico se confirma con electrocardiograma "
            "y marcadores cardíacos como troponina. El tratamiento de urgencia incluye aspirina, "
            "anticoagulantes y angioplastia coronaria o trombolisis. La rehabilitación cardíaca "
            "posterior reduce el riesgo de futuros eventos."
        ),
    },
    {
        "doc_id": "doc_alzheimer_009",
        "url": "https://medlineplus.gov/alzheimer",
        "title": "Enfermedad de Alzheimer",
        "text": (
            "El Alzheimer es la causa más común de demencia, representando entre el 60 y 80 "
            "por ciento de los casos. Es una enfermedad neurodegenerativa progresiva que destruye "
            "la memoria y otras funciones mentales importantes. Los primeros síntomas incluyen "
            "olvidos frecuentes, dificultad para encontrar palabras y desorientación. "
            "Con la progresión, los pacientes pierden la capacidad de realizar actividades "
            "cotidianas y reconocer a sus familiares. Las placas de proteína beta-amiloide y "
            "los ovillos de tau son características patológicas. Aunque no existe cura, "
            "los inhibidores de colinesterasa como el donepezilo pueden aliviar temporalmente "
            "los síntomas cognitivos."
        ),
    },
    {
        "doc_id": "doc_hipotiroidismo_010",
        "url": "https://www.nhs.uk/conditions/hypothyroidism",
        "title": "Hipotiroidismo",
        "text": (
            "El hipotiroidismo ocurre cuando la glándula tiroides no produce suficiente hormona "
            "tiroidea para satisfacer las necesidades del organismo. Los síntomas incluyen fatiga, "
            "aumento de peso, sensación de frío, estreñimiento, piel seca, caída del cabello, "
            "depresión y bradicardia. La causa más común en países desarrollados es la tiroiditis "
            "de Hashimoto, una enfermedad autoinmune. El diagnóstico se realiza midiendo la "
            "hormona estimulante de la tiroides TSH en sangre. El tratamiento consiste en "
            "reemplazo hormonal con levotiroxina, que debe tomarse en ayunas y ajustarse "
            "periódicamente según los valores de TSH. Con tratamiento adecuado los síntomas "
            "desaparecen completamente."
        ),
    },
    {
        "doc_id": "doc_neumonia_011",
        "url": "https://medlineplus.gov/neumonia",
        "title": "Neumonía",
        "text": (
            "La neumonía es una infección que inflama los sacos de aire en uno o ambos pulmones, "
            "que pueden llenarse de líquido o pus. Los síntomas incluyen tos con flema o pus, "
            "fiebre, escalofríos y dificultad para respirar. Las bacterias, especialmente "
            "Streptococcus pneumoniae, son la causa más común en adultos, aunque también "
            "pueden ser virales o fúngicas. Los grupos de mayor riesgo son adultos mayores, "
            "menores de dos años y personas con enfermedades crónicas o inmunosupresión. "
            "El diagnóstico incluye auscultación, radiografía de tórax y cultivo de esputo. "
            "El tratamiento antibiótico debe iniciarse precozmente y puede requerir "
            "hospitalización en casos graves."
        ),
    },
    {
        "doc_id": "doc_anemia_012",
        "url": "https://www.mayoclinic.org/anemia",
        "title": "Anemia por Deficiencia de Hierro",
        "text": (
            "La anemia ferropénica es el tipo más común de anemia, causada por reservas "
            "insuficientes de hierro para producir hemoglobina. Los síntomas incluyen fatiga, "
            "debilidad, palidez, disnea de esfuerzo, palpitaciones y cefalea. Las causas "
            "incluyen ingesta insuficiente de hierro, absorción deficiente, pérdida crónica "
            "de sangre por menstruación abundante o sangrado gastrointestinal, y mayor demanda "
            "durante el embarazo. El diagnóstico se confirma con hemograma completo, niveles "
            "de ferritina sérica y saturación de transferrina. El tratamiento incluye "
            "suplementos de hierro oral y modificaciones dietéticas para aumentar la ingesta "
            "de hierro y vitamina C que mejora su absorción."
        ),
    },
    {
        "doc_id": "doc_ictus_013",
        "url": "https://www.nhs.uk/conditions/stroke",
        "title": "Accidente Cerebrovascular",
        "text": (
            "El accidente cerebrovascular o ictus ocurre cuando el flujo sanguíneo al cerebro "
            "se interrumpe por obstrucción de un vaso sanguíneo o hemorragia. Es una emergencia "
            "médica que requiere atención inmediata. Los síntomas se recuerdan con el acrónimo "
            "FAST: cara caída, debilidad en el brazo, alteraciones del habla y tiempo de "
            "llamar al servicio de emergencias. El diagnóstico se confirma con tomografía "
            "computarizada o resonancia magnética cerebral. El ictus isquémico puede tratarse "
            "con trombolisis si se atiende en las primeras horas. La rehabilitación intensiva "
            "es fundamental para recuperar funciones perdidas. La prevención incluye control "
            "de hipertensión y fibrilación auricular."
        ),
    },
    {
        "doc_id": "doc_epoc_014",
        "url": "https://medlineplus.gov/epoc",
        "title": "Enfermedad Pulmonar Obstructiva Crónica",
        "text": (
            "La EPOC es una enfermedad inflamatoria crónica de los pulmones que obstruye el "
            "flujo de aire. Engloba el enfisema y la bronquitis crónica. El principal factor "
            "de riesgo es el tabaquismo, responsable de más del 85 por ciento de los casos. "
            "Los síntomas incluyen disnea progresiva, tos crónica con producción de esputo y "
            "sibilancias. El diagnóstico se confirma con espirometría que muestra obstrucción "
            "no reversible. El tratamiento incluye abandono del tabaco, broncodilatadores de "
            "larga acción, corticosteroides inhalados en casos graves y oxigenoterapia en "
            "insuficiencia respiratoria. Las exacerbaciones frecuentes aceleran el deterioro "
            "de la función pulmonar."
        ),
    },
    {
        "doc_id": "doc_insuficiencia_renal_015",
        "url": "https://www.mayoclinic.org/kidney-failure",
        "title": "Insuficiencia Renal Crónica",
        "text": (
            "La insuficiencia renal crónica es la pérdida gradual de la función renal a lo "
            "largo del tiempo. Los riñones filtran los desechos y el exceso de líquido de la "
            "sangre, que se eliminan en la orina. La enfermedad avanzada puede requerir "
            "diálisis o trasplante renal. Las causas más comunes son la diabetes y la "
            "hipertensión arterial no controlada. Los síntomas en etapas avanzadas incluyen "
            "náuseas, fatiga, disminución de la micción, retención de líquidos y anemia. "
            "El diagnóstico incluye medición de creatinina sérica y tasa de filtrado "
            "glomerular. El tratamiento busca ralentizar la progresión controlando la "
            "causa subyacente y las complicaciones."
        ),
    },
    {
        "doc_id": "doc_fibromialgia_016",
        "url": "https://www.nhs.uk/conditions/fibromyalgia",
        "title": "Fibromialgia",
        "text": (
            "La fibromialgia es un trastorno caracterizado por dolor musculoesquelético "
            "generalizado acompañado de fatiga, problemas de sueño, memoria y estado de "
            "ánimo. Los investigadores creen que la fibromialgia amplifica las sensaciones "
            "dolorosas al afectar la forma en que el cerebro y la médula espinal procesan "
            "las señales de dolor. Los síntomas suelen comenzar después de un trauma físico, "
            "cirugía, infección o estrés psicológico significativo. El diagnóstico es clínico "
            "y se basa en la presencia de dolor generalizado por más de tres meses. "
            "El tratamiento incluye analgésicos, antidepresivos, anticonvulsivantes y "
            "terapias no farmacológicas como ejercicio aeróbico y terapia cognitivo-conductual."
        ),
    },
    {
        "doc_id": "doc_colesterol_017",
        "url": "https://medlineplus.gov/colesterol",
        "title": "Hipercolesterolemia",
        "text": (
            "La hipercolesterolemia o colesterol alto es una afección en la que hay demasiado "
            "colesterol en la sangre. El colesterol es una sustancia grasa necesaria para "
            "construir células sanas, pero niveles elevados aumentan el riesgo de enfermedad "
            "cardíaca. El colesterol LDL malo se acumula en las paredes arteriales formando "
            "placas ateroscleróticas. El colesterol HDL bueno ayuda a eliminar el LDL. "
            "La hipercolesterolemia no presenta síntomas, por lo que solo se detecta con "
            "análisis de sangre. Los factores de riesgo incluyen dieta rica en grasas "
            "saturadas, obesidad, sedentarismo y factores genéticos. El tratamiento incluye "
            "cambios dietéticos y estatinas como atorvastatina o simvastatina."
        ),
    },
    {
        "doc_id": "doc_migraña_018",
        "url": "https://www.mayoclinic.org/migraine",
        "title": "Migraña",
        "text": (
            "La migraña es un tipo de dolor de cabeza recurrente moderado a severo que "
            "generalmente afecta un lado de la cabeza y se acompaña de náuseas, vómitos y "
            "sensibilidad a la luz y el sonido. Algunos pacientes experimentan aura antes "
            "del dolor, con síntomas visuales como destellos o líneas en zigzag. "
            "Los desencadenantes incluyen estrés, cambios hormonales, ciertos alimentos "
            "como queso y vino tinto, privación del sueño y cambios climáticos. "
            "El diagnóstico es clínico. El tratamiento incluye analgésicos, triptanes para "
            "el ataque agudo y profilaxis con betabloqueantes, topiramato o antidepresivos "
            "tricíclicos en casos frecuentes."
        ),
    },
    {
        "doc_id": "doc_hepatitis_019",
        "url": "https://www.nhs.uk/conditions/hepatitis-c",
        "title": "Hepatitis C",
        "text": (
            "La hepatitis C es una infección viral que ataca principalmente el hígado y puede "
            "causar inflamación hepática grave. Se transmite por contacto con sangre "
            "contaminada, principalmente por compartir agujas en usuarios de drogas "
            "intravenosas, transfusiones antiguas y en menor medida por vía sexual. "
            "La mayoría de las infecciones agudas son asintomáticas. La infección crónica "
            "puede progresar a cirrosis hepática y carcinoma hepatocelular. El diagnóstico "
            "incluye detección de anticuerpos anti-VHC y carga viral por PCR. "
            "Los antivirales de acción directa modernos como sofosbuvir logran tasas de "
            "curación superiores al 95 por ciento con tratamientos de 8 a 12 semanas."
        ),
    },
    {
        "doc_id": "doc_obesidad_020",
        "url": "https://medlineplus.gov/obesidad",
        "title": "Obesidad y Síndrome Metabólico",
        "text": (
            "La obesidad es una enfermedad crónica compleja definida por acumulación excesiva "
            "de grasa corporal con un índice de masa corporal mayor o igual a 30 kg/m2. "
            "El síndrome metabólico es un conjunto de condiciones que ocurren juntas: "
            "obesidad abdominal, hipertensión, glucemia elevada, triglicéridos altos y "
            "colesterol HDL bajo, aumentando el riesgo cardiovascular. Las causas incluyen "
            "exceso calórico, sedentarismo, factores genéticos, hormonales y ambientales. "
            "Las complicaciones incluyen diabetes tipo 2, enfermedad cardiovascular, apnea "
            "del sueño, osteoartritis y ciertos canceres. El tratamiento requiere cambios "
            "en el estilo de vida, dieta hipocalórica, ejercicio regular y en casos seleccionados "
            "medicamentos o cirugía bariátrica."
        ),
    },
]
