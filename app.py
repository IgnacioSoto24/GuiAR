import streamlit as st
import os

from src.ingestion import construir_vectorstore
from src.classifier import clasificar_pregunta
from src.rag_pipeline import obtener_cadena_rag

st.set_page_config(
    page_title="GuiAR - Tutor Pedagógico",
    layout="wide",
    page_icon="📘"
)

st.markdown("""
<style>
body {
    background-color: #F5F7FA;
}
.block {
    background: white;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #dcdcdc;
    margin-bottom: 20px;
}
.title {
    color: #2C3E50;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("⚙️ Configuración")

    asignatura_usuario = st.selectbox(
        "📘 Asignatura (preferida):",
        ["historia", "lenguaje", "matematicas", "ciencias", "ingles", "geografia"]
    )

    nivel = st.radio(
        "🎓 Nivel de guía",
        ["breve", "intermedio", "profundo"],
        index=1
    )

    archivo_pdf = st.file_uploader("📄 Subir PDF para esta asignatura", type="pdf")

    if archivo_pdf is not None:
        if st.button("Procesar PDF"):
            ruta = os.path.join("data", archivo_pdf.name)
            with open(ruta, "wb") as f:
                f.write(archivo_pdf.getbuffer())

            try:
                construir_vectorstore(ruta, asignatura_usuario)
                st.success(f"✔ PDF procesado correctamente para **{asignatura_usuario}**")
            except Exception as e:
                st.error(f"⚠ Error procesando PDF: {str(e)}")

st.title("📘 GuiAR — Tutor Pedagógico Inteligente")

pregunta = st.text_input("✍️ Escribe tu pregunta:")

if pregunta:
    asignatura_predicha = clasificar_pregunta(pregunta, asignatura_usuario)

    st.info(f"📌 Asignatura detectada: **{asignatura_predicha}**")

    if asignatura_predicha != asignatura_usuario:
        st.warning(
            f"❗ Esta pregunta NO corresponde a la asignatura seleccionada ({asignatura_usuario})."
        )
        st.write(f"➡ La pregunta pertenece a **{asignatura_predicha}**.")
        st.stop()

    try:
        cadena = obtener_cadena_rag(asignatura_usuario, nivel)
        respuesta = cadena.run(pregunta)

        st.markdown(f"""
        <div class='block'>
          <h3 class='title'>💡 Orientación del tutor ({nivel})</h3>
          {respuesta}
        </div>
        """, unsafe_allow_html=True)

    except FileNotFoundError:
        st.error(
            f"❌ No existe un vectorstore para **{asignatura_usuario}**.\n"
            "Sube un PDF de esta asignatura en el panel izquierdo."
        )

    except Exception as e:
        st.error(f"⚠ Error inesperado: {str(e)}")
