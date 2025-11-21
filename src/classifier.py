from langchain_ollama import OllamaLLM

ASIGNATURAS = [
    "historia",
    "lenguaje",
    "matematicas",
    "ciencias",
    "ingles",
    "geografia",
]

def clasificar_pregunta(pregunta: str, asignatura_usuario: str) -> str:
    """
    Clasifica la pregunta en UNA asignatura escolar.
    - Respeta la preferencia del usuario sólo si la pregunta se relaciona claramente.
    - Evita contaminación desde el contenido del vectorstore.
    - Es estricto: solo acepta una de las 6 asignaturas posibles.
    """

    llm = OllamaLLM(model="mistral")

    prompt = f"""
Eres un clasificador EXCLUSIVO de asignaturas.

Debes decidir a cuál de estas asignaturas pertenece la pregunta:

historia
lenguaje
matematicas
ciencias
ingles
geografia

REGLAS OBLIGATORIAS:
- Responde SOLO con una palabra: exactamente una de las asignaturas listadas.
- No escribas frases, explicaciones ni signos adicionales.
- No inventes asignaturas.
- Si la pregunta es sobre fechas, guerras, sociedades, culturas o eventos reales → historia.
- Si es análisis de textos, comprensión lectora, poemas, narrativa → lenguaje.
- Si incluye números, cálculos, operaciones → matematicas.
- Si menciona experimentos, cuerpo humano, animales, física, química → ciencias.
- Si habla de idioma o traducciones → ingles.
- Si menciona territorios, mapas, climas, continentes → geografia.

REGLAS DE PREFERENCIA:
- Si la pregunta coincide claramente con la asignatura preferida del usuario,
  devuelve la asignatura preferida.
- Si NO coincide, devuelve la asignatura REAL.
- Si no estás seguro, devuelve la asignatura preferida del usuario.

Pregunta:
\"\"\"{pregunta}\"\"\"

Asignatura preferida del usuario:
\"\"\"{asignatura_usuario}\"\"\"

Respuesta (solo una palabra):
"""

    respuesta = llm.invoke(prompt).strip().lower()

    if respuesta not in ASIGNATURAS:
        return asignatura_usuario

    return respuesta
