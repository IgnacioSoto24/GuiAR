import streamlit as st
from src.rag_pipeline import obtener_cadena_rag
from src.ingestion import construir_vectorstore
import os

st.set_page_config(page_title="GuiAR - Tutor Pedagógico", layout="wide")
st.title("📘 GuiAR: Tutor Pedagógico con IA")

# --- Botón opcional para subir PDF ---
archivo_pdf = st.file_uploader("📂 Subir un PDF adicional (opcional)", type="pdf")

if archivo_pdf is not None:
    ruta_guardado = os.path.join("data", archivo_pdf.name)
    with open(ruta_guardado, "wb") as f:
        f.write(archivo_pdf.getbuffer())
    construir_vectorstore(ruta_guardado, "faiss_index")
    st.success(f"✅ El archivo {archivo_pdf.name} fue procesado y agregado al tutor")

# --- Inicializar pipeline RAG ---
cadena_rag = obtener_cadena_rag()

# --- Entrada del usuario ---
pregunta = st.text_input("✍️ Haz tu consulta:")

if pregunta:
    with st.spinner("Generando orientación..."):
        respuesta = cadena_rag.run(pregunta)

    st.write("### 💡 Orientación del tutor:")
    st.write(respuesta)

    # --- Evaluación con métricas ---
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

        # Contexto mínimo para que no falle Faithfulness
        contexto = ["Fragmentos recuperados desde FAISS o texto de apoyo"]

        caso = LLMTestCase(
            input=pregunta,
            actual_output=respuesta,
            expected_output="Referencia curricular o respuesta esperada oficial.",
            retrieval_context=contexto,
            tools_called=[]  # necesario para ArgumentCorrectnessMetric
        )

        modelo_local = OllamaModel(model="mistral")

        metricas = [
            FaithfulnessMetric(model=modelo_local),
            AnswerRelevancyMetric(model=modelo_local),
            ContextualRecallMetric(model=modelo_local),
            ToxicityMetric(model=modelo_local),
            ArgumentCorrectnessMetric(model=modelo_local),
        ]

        # ✅ Ejecutar evaluación
        resultado = evaluate([caso], metricas)

        # ✅ Convertir resultados a JSON serializable
        tabla = {}
        for m in resultado.metrics:   # 🔥 antes era .metrics_data
            tabla[m.name] = {
                "score": m.score,
                "threshold": m.threshold,
                "reason": m.reason,
            }

        st.json(tabla)
