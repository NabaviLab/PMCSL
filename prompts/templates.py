GRADE_PROMPT_TEMPLATE = """
You are a computational pathology assistant.
Describe the visually identifiable histopathologic characteristics of {grade_name}
in prostate hematoxylin-and-eosin whole-slide images.

Focus on gland architecture, lumen formation, gland fusion, cribriform growth,
nuclear atypia, stromal organization, and solid or poorly formed tumor patterns.
Return one concise paragraph for semantic supervision. Do not provide treatment,
prognosis, or molecular findings.
""".strip()

CLASS_DESCRIPTIONS = {
    "Grade1": "combined benign and low-grade morphology used as the first study class",
    "Grade2": "the second study class with increasing glandular irregularity",
    "Grade3": "the third study class with advanced architectural distortion",
    "Grade4": "the highest study class with poorly formed or solid tumor growth",
}
