from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.retriever import cargar_vectorstore

def obtener_cadena_rag(nivel="medio"):
    vectorstore = cargar_vectorstore("faiss_index")
    recuperador = vectorstore.as_retriever(search_kwargs={"k": 3})

    modelo_llm = OllamaLLM(model="mistral")

    if nivel == "básico":
        tono = (
            "Usa oraciones simples, ejemplos concretos y explicaciones breves. "
            "Evita tecnicismos y explica los conceptos de forma cercana."
        )
        pasos = "3 o 4 pasos claros y cortos."
    elif nivel == "medio":
        tono = (
            "Usa un lenguaje claro y formal, con explicaciones detalladas, ejemplos históricos o conceptuales, "
            "y preguntas que fomenten la reflexión guiada."
        )
        pasos = "5 a 6 pasos, desarrollando el razonamiento de forma progresiva."
    else:  # avanzado
        tono = (
            "Usa un lenguaje académico y preciso, con análisis más profundo, conexiones entre causas y consecuencias, "
            "y referencias a contextos históricos o científicos cuando corresponda."
        )
        pasos = "6 a 8 pasos bien desarrollados con razonamiento complejo."

    plantilla = f"""
Eres un tutor pedagógico experto que siempre responde en español.
Tu tarea es guiar al estudiante paso a paso, ayudándole a construir comprensión y pensamiento crítico.
No entregues respuestas directas, sino que conduce al estudiante con explicaciones graduales.

Instrucciones:
1. Adapta la respuesta al nivel educativo: {tono}
2. Organiza la respuesta en {pasos}
3. Separa cada paso con un salto de línea.
4. El primer paso siempre introduce el contexto general del tema (qué es o por qué es importante).
5. Los pasos intermedios explican causas, desarrollo o conceptos clave.
6. El último paso invita a profundizar o conectar con otro contenido (sin decir la respuesta exacta).
7. Si la pregunta no está relacionada con materias escolares, responde: "Esa pregunta no está relacionada con las materias escolares que puedo enseñar."
8. Si el contexto recuperado no es relevante, ignóralo.

Pregunta del estudiante:
{{question}}

Contexto recuperado:
{{context}}

💡 Orientación del tutor (usa el formato Paso 1:, Paso 2:, ... con saltos de línea entre pasos):
"""

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=plantilla,
    )

    return RetrievalQA.from_chain_type(
        llm=modelo_llm,
        retriever=recuperador,
        chain_type_kwargs={"prompt": prompt}
    )
