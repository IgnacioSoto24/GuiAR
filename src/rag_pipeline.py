# src/rag_pipeline.py
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.retriever import cargar_vectorstore

def obtener_cadena_rag():
    vectorstore = cargar_vectorstore("faiss_index")
    recuperador = vectorstore.as_retriever(search_kwargs={"k": 3})

    modelo_llm = OllamaLLM(model="mistral")

    plantilla = """Eres un tutor pedagógico experto que SIEMPRE responde en español.
Tu función es guiar al estudiante con explicaciones claras, progresivas y relacionadas directamente con la pregunta.
Evita dar la respuesta exacta, pero proporciona información suficiente para que el estudiante pueda deducirla.
Nunca desvíes el tema ni introduzcas información que no responde al foco de la pregunta.
No inventes datos ni menciones política o ejemplos irrelevantes.

Estructura sugerida:
1. Presenta brevemente el contexto del tema.
2. Explica los elementos o hechos que permiten ubicar el momento o la idea central.
3. Da una pista clara pero indirecta para que el estudiante llegue a la respuesta correcta.
4. Termina con una pregunta de comprobación que mantenga el foco original (por ejemplo, si la pregunta era sobre “cuándo”, la pregunta final también debe ser sobre el tiempo).

Pregunta del estudiante:
{question}

Contexto recuperado:
{context}

💬 Respuesta del tutor pedagógico:"""

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=plantilla,
    )

    return RetrievalQA.from_chain_type(
        llm=modelo_llm,
        retriever=recuperador,
        chain_type_kwargs={"prompt": prompt}
    )
