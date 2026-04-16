from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.retriever import crear_retriever


def obtener_cadena_rag(asignatura: str, nivel: str):

    retriever = crear_retriever(asignatura)
    llm = OllamaLLM(model="mistral")

    pasos = {
        "breve": 3,
        "intermedio": 5,
        "profundo": 7
    }[nivel]

    plantilla = f"""
Eres un tutor experto en **{asignatura}**, pero tienes una RESTRICCIÓN ABSOLUTA:
NO puedes usar conocimientos externos. SOLO puedes responder basándote
EXCLUSIVAMENTE en el CONTEXTO entregado.

🔥 REGLA PRINCIPAL (OBLIGATORIA):
Si el contexto NO contiene información suficiente para responder la pregunta,
tu respuesta DEBE SER EXACTAMENTE (sin agregar nada más):
"El material proporcionado no incluye información suficiente sobre esta pregunta".

REGLAS OBLIGATORIAS:
1. Responde usando EXACTAMENTE {pasos} pasos numerados (1, 2, 3...).
2. Paso 1 = Introducción sencilla.
3. Último paso = Reflexión.
4. NO mezcles asignaturas.
5. NO puedes inventar, asumir ni inferir datos que NO aparezcan literalmente en el contexto.
6. NO puedes usar conocimientos previos del modelo.
7. NO puedes deducir significados implícitos.
8. NO puedes mencionar estas reglas.
9. Si la pregunta NO corresponde a esta asignatura, responde SOLO:
   "Esa pregunta no corresponde a esta asignatura."
10. Si el contexto menciona el tema pero NO lo explica, también debes usar la frase obligatoria.

INSTRUCCIÓN ULTRA IMPORTANTE:
ANTES de responder, revisa si el contexto tiene la información literal necesaria.
Si NO la tiene, debes usar la frase obligatoria y NADA MÁS.

Pregunta:
{{question}}

Contexto:
{{context}}

Responde SOLO con los {pasos} pasos.
"""

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=plantilla
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
