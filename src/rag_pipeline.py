from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.retriever import crear_retriever


def obtener_cadena_rag(asignatura: str, nivel: str):
    """
    Construye una cadena RAG personalizada por asignatura y nivel educativo.
    """

    retriever = crear_retriever(asignatura)
    llm = OllamaLLM(model="mistral")

    pasos = {
        "breve": 3,
        "intermedio": 5,
        "profundo": 7
    }[nivel]

    plantilla = f"""
Eres un tutor experto de la asignatura **{asignatura}**.

Debes guiar al estudiante paso a paso, sin entregar respuestas directas,
siguiendo exactamente las reglas establecidas.

REGLAS OBLIGATORIAS:
1. Responde usando EXACTAMENTE {pasos} pasos numerados.
2. Paso 1 debe ser una introducción simple.
3. El último paso siempre debe ser una reflexión.
4. NO mezcles otras asignaturas bajo ninguna circunstancia.
5. Usa SOLO información del contexto entregado abajo.
6. Si el contexto no contiene suficiente información, dilo claramente:
   "El material proporcionado no incluye información suficiente sobre esta pregunta".
7. NO inventes fechas, nombres, causas o datos que NO estén en el contexto.
8. NO uses conocimientos externos.
9. Si la pregunta NO corresponde a esta asignatura, responde SOLO:
   "Esa pregunta no corresponde a esta asignatura."

Pregunta del estudiante:
{{question}}

Contexto recuperado:
{{context}}

RESPONDE SOLO con los {pasos} pasos. No agregues texto fuera del formato.
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
