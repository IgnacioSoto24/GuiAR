import streamlit as st
from src.rag_pipeline import obtener_cadena_rag
from src.ingestion import construir_vectorstore
import os

st.set_page_config(page_title="GuiAR - Tutor Pedagógico", layout="wide")
st.title("📘 GuiAR: Tutor Pedagógico con IA")

archivo_pdf = st.file_uploader("📂 Subir un PDF adicional (opcional)", type="pdf")

if archivo_pdf is not None:
    st.info(f"Has cargado el archivo: {archivo_pdf.name}")
    
    if st.button("Procesar PDF"):
        ruta_guardado = os.path.join("data", archivo_pdf.name)
        with open(ruta_guardado, "wb") as f:
            f.write(archivo_pdf.getbuffer())
        
        construir_vectorstore(ruta_guardado, "faiss_index")
        st.success(f"✅ El archivo {archivo_pdf.name} fue procesado y agregado al tutor")

cadena_rag = obtener_cadena_rag()

pregunta = st.text_input("✍️ Haz tu consulta:")

if pregunta:
    with st.spinner("Generando orientación..."):
        respuesta = cadena_rag.run(pregunta)

    st.write("### 💡 Orientación del tutor:")
    st.write(respuesta)

    st.subheader("📊 Evaluación de la respuesta")
    if st.button("Evaluar con métricas"):
        from deepeval.metrics import (
            AnswerRelevancyMetric,
            FaithfulnessMetric,
            ContextualRecallMetric,
            ToxicityMetric,
            ArgumentCorrectnessMetric,
        )
        from deepeval.test_case import LLMTestCase
        from deepeval.models import OllamaModel
        from deepeval import evaluate

        contexto = ["Fragmentos recuperados desde FAISS o texto de apoyo"]

        caso = LLMTestCase(
            input=pregunta,
            actual_output=respuesta,
            expected_output="Referencia curricular o respuesta esperada oficial.",
            retrieval_context=contexto,
            tools_called=[],
        )

        modelo_local = OllamaModel(model="mistral")

        metricas = [
            FaithfulnessMetric(model=modelo_local),
            AnswerRelevancyMetric(model=modelo_local),
            ContextualRecallMetric(model=modelo_local),
            ToxicityMetric(model=modelo_local),
            ArgumentCorrectnessMetric(model=modelo_local),
        ]

        st.info("🧮 Ejecutando evaluación... revisa la terminal para ver los resultados.")
        evaluate([caso], metricas)
        st.success("✅ Evaluación completada. Revisa los resultados en la terminal.")
