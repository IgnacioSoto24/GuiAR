from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.retriever import cargar_vectorstore

def obtener_cadena_rag():
    vectorstore = cargar_vectorstore("faiss_index")
    recuperador = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ✅ Usar mistral en Ollama (CPU en tu setup)
    modelo_llm = OllamaLLM(model="mistral")

    # ✅ Prompt pedagógico para orientar y NO dar respuestas directas
    plantilla = """Eres un tutor pedagógico que SIEMPRE responde en español.
Tu tarea es **orientar al estudiante**, no darle la respuesta final.
Debes:
- Guiar al estudiante paso a paso.
- Dar pistas, sugerencias o ideas clave.
- Promover que piense y razone por sí mismo.
- NO entregar la solución completa ni la frase exacta de la respuesta.

Pregunta del estudiante:
{question}

Contexto recuperado:
{context}

💡 Orienta al estudiante sin darle la respuesta directa:"""

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=plantilla,
    )

    return RetrievalQA.from_chain_type(
        llm=modelo_llm,
        retriever=recuperador,
        chain_type_kwargs={"prompt": prompt}
    )
