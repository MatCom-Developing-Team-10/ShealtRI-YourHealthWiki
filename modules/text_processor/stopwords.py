"""Spanish stopwords with medical domain considerations.

This module provides stopword configuration for preprocessing Spanish medical text.

Stopword sources:
    1. NLTK Spanish stopwords (313 words) - loaded dynamically in TextProcessor
       Covers: articles, prepositions, conjunctions, pronouns, auxiliary verbs

    2. Additional generic stopwords - defined here
       Very common words not in NLTK that don't add semantic value

Medical terms are intentionally NOT included as stopwords because:
    - "paciente", "tratamiento", "síntoma" are relevant for medical searches
    - "alto", "bajo", "mayor", "menor" are important descriptors
    - Domain-specific terms should be preserved for LSI semantic analysis
"""

# Additional generic stopwords not covered by NLTK Spanish
# These are very common words that don't add semantic value
ADDITIONAL_SPANISH_STOPWORDS: set[str] = {
    # Common filler words
    "etc", "etcétera",
    "así", "además", "siendo",
    "hacer", "hace", "hacen", "hecho",
    "decir", "dice", "dicen", "dicho",
    "ver", "vemos", "vez", "veces",
    "ir", "va", "van", "ido",
    "dar", "da", "dan", "dado",
    "poner", "pone", "ponen", "puesto",
    "cada", "cada uno",
    "mientras", "aunque", "sino",
    "según", "mediante", "través",
    # Generic adjectives that don't add medical value
    "mismo", "misma", "mismos", "mismas",
    "propio", "propia", "propios", "propias",
    "cierto", "cierta", "ciertos", "ciertas",
    "varios", "varias",
    "algún", "alguna", "algunos", "algunas",
    "ningún", "ninguna", "ningunos", "ningunas",
    "tal", "tales",
    "cualquier", "cualquiera",
    # Numbers as words (digits are kept)
    "uno", "dos", "tres", "cuatro", "cinco",
    "seis", "siete", "ocho", "nueve", "diez",
    "primero", "segundo", "tercero", "cuarto", "quinto",
    "primera", "segunda", "tercera", "cuarta", "quinta",
}


# Medical abbreviations to PRESERVE (NOT stopwords)
# These are clinically significant and should never be removed
MEDICAL_ABBREVIATIONS: set[str] = {
    # Cardiovascular
    "hta",       # Hipertensión arterial
    "iam",       # Infarto agudo de miocardio
    "icc",       # Insuficiencia cardíaca congestiva
    "ecg", "ekg",  # Electrocardiograma
    "acv",       # Accidente cerebrovascular
    "fa",        # Fibrilación auricular
    # Metabolic
    "dm", "dm1", "dm2",  # Diabetes mellitus
    "hb", "hba1c",       # Hemoglobina
    "imc",       # Índice de masa corporal
    # Respiratory
    "epoc",      # Enfermedad pulmonar obstructiva crónica
    "covid", "covid19", "sars",
    "asma",
    # Renal
    "irc", "erc",  # Insuficiencia/Enfermedad renal crónica
    "tfg",       # Tasa de filtración glomerular
    # Imaging
    "rx",        # Radiografía
    "tac", "tc", # Tomografía axial computarizada
    "rmn", "rm", # Resonancia magnética nuclear
    "eco",       # Ecografía
    # Lab values
    "hcm",       # Hemoglobina corpuscular media
    "vcm",       # Volumen corpuscular medio
    "ldl", "hdl", "vldl",  # Lipoproteínas
    "pcr",       # Proteína C reactiva
    "vsg",       # Velocidad de sedimentación globular
    # Medications
    "aines",     # Antiinflamatorios no esteroideos
    "ieca",      # Inhibidores de la ECA
    "ara2", "araii",  # Antagonistas del receptor de angiotensina
    "bb",        # Betabloqueantes
    # Units (preserve for context)
    "mg", "ml", "kg", "g", "l",
    "mmhg",      # Milímetros de mercurio (presión)
    "mgdl",      # Miligramos por decilitro
}


# Backward compatibility alias
SPANISH_MEDICAL_STOPWORDS = ADDITIONAL_SPANISH_STOPWORDS
