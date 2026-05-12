"""Centralized stopwords configuration for medical text processing.

This module provides stopword sets for both English and Spanish medical domains.
All modules should import stopwords from this single source of truth to maintain
consistency across the pipeline.

Stopword sources and organization:
    1. Common language stopwords - articles, prepositions, conjunctions, etc.
    2. Generic verbs and adjectives - too common to add semantic value
    3. Domain-specific terms are intentionally NOT included as stopwords because
       medical vocabulary ("hypertension", "treatment", "paciente", etc.) is
       crucial for relevance scoring in the retrieval pipeline.
"""

# ==============================================================================
# ENGLISH STOPWORDS
# ==============================================================================

ENGLISH_STOPWORDS: set[str] = {
    # Common articles, prepositions, conjunctions
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "do", "for",
    "from", "had", "has", "have", "he", "her", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "just", "may", "me", "my", "no",
    "not", "now", "of", "on", "or", "out", "over", "own", "same", "she",
    "so", "such", "the", "than", "that", "this", "to", "too", "up",
    "was", "we", "what", "when", "where", "which", "who", "why", "will",
    "with", "you", "your",
    # Common verbs that are too generic for medical search
    "can", "could", "should", "would", "might", "must", "shall",
    "get", "got", "make", "made", "take", "took", "give", "given",
    "go", "went", "come", "came", "say", "said", "see", "seen",
    # Articles and determiners
    "all", "each", "every", "both", "either", "neither", "such",
    # Quantifiers
    "few", "more", "most", "some", "any", "other", "another",
}

ENGLISH_MEDICAL_ABBREVIATIONS: set[str] = {
    # Cardiovascular
    "htn", "hbp",  # Hypertension, high blood pressure
    "mi", "ami",   # Myocardial infarction
    "chf",         # Congestive heart failure
    "ecg", "ekg",  # Electrocardiogram
    "cva",         # Cerebrovascular accident
    "afib", "af",  # Atrial fibrillation
    # Metabolic
    "dm", "dm1", "dm2",  # Diabetes mellitus
    "hba1c", "hgba1c",   # Hemoglobin A1c
    "bmi",         # Body mass index
    # Respiratory
    "copd",        # Chronic obstructive pulmonary disease
    "covid", "covid19",
    "asthma",
    # Renal
    "ckd",         # Chronic kidney disease
    "gfr",         # Glomerular filtration rate
    # Imaging
    "xray", "rx",  # X-ray
    "ct", "cta",   # Computed tomography
    "mri",         # Magnetic resonance imaging
    "ultrasound",
    # Lab values
    "mcv",         # Mean corpuscular volume
    "ldl", "hdl",  # Cholesterol types
    "crp",         # C-reactive protein
    # Medications
    "ace",         # ACE inhibitors
    "arb",         # Angiotensin receptor blocker
    "beta",        # Beta blockers
    "nsaid",       # Non-steroidal anti-inflammatory drug
    # Units
    "mg", "ml", "kg", "g", "l",
    "mmhg",        # Millimeters of mercury
    "mgdl",        # Milligrams per deciliter
}

# ==============================================================================
# SPANISH STOPWORDS
# ==============================================================================

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

SPANISH_MEDICAL_ABBREVIATIONS: set[str] = {
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

# ==============================================================================
# BACKWARD COMPATIBILITY ALIASES
# ==============================================================================

# For modules that expect Spanish-specific naming
SPANISH_MEDICAL_STOPWORDS = ADDITIONAL_SPANISH_STOPWORDS
