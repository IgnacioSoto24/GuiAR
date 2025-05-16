# app.py

import streamlit as st
from evaluador import (
    generar_actividad,
    generar_rubrica,
    retroalimentar_respuesta,
    generar_preguntas_estructuradas
)

# Configuración de la app
st.set_page_config(page_title="EvaluAR", layout="centered")

# === Inicialización de estados ===
if "preguntas" not in st.session_state:
    st.session_state.preguntas = None

if "respuestas" not in st.session_state:
    st.session_state.respuestas = None

if "retroalimentacion_generada" not in st.session_state:
    st.session_state.retroalimentacion_generada = False

if "rubrica_generada" not in st.session_state:
    st.session_state.rubrica_generada = False

if "rubrica" not in st.session_state:
    st.session_state.rubrica = None

if "puntaje_total" not in st.session_state:
    st.session_state.puntaje_total = 0

if "datos" not in st.session_state:
    st.session_state.datos = {}

# === Formulario del Docente ===
st.title("📘 EvaluAR – Evaluaciones Auténticas con Razonamiento Artificial")
st.subheader("Completa este formulario para generar tu evaluación personalizada:")

asignatura = st.selectbox("Selecciona la asignatura:", [
    "Lenguaje y Comunicación", "Matemática", "Historia", "Ciencias", "Educación Ciudadana", "Otra"
])

nivel = st.selectbox("Selecciona el nivel educativo:", [
    "1° Medio", "2° Medio", "3° Medio", "4° Medio"
])

objetivo_aprendizaje = st.text_area("Escribe el Objetivo de Aprendizaje (puedes copiarlo desde el currículum nacional):")

tipo_evaluacion = st.radio("Tipo de evaluación:", ["Diagnóstica", "Formativa", "Sumativa"])

habilidad = st.selectbox("Habilidad que deseas evaluar:", [
    "Comprensión", "Análisis", "Aplicación", "Argumentación", "Resolución de problemas"
])

formato = st.selectbox("Formato de la actividad:", [
    "Respuesta múltiple", "Desarrollo argumentado", "Análisis de caso", "Proyecto práctico", "Informe escrito", "Producto creativo"
])

clave_groq = "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"

if st.button("✅ Generar evaluación"):
    datos = {
        "asignatura": asignatura,
        "nivel": nivel,
        "objetivo_aprendizaje": objetivo_aprendizaje,
        "tipo_evaluacion": tipo_evaluacion,
        "habilidad": habilidad,
        "formato": formato
    }

    st.session_state.preguntas = generar_preguntas_estructuradas(datos, clave_groq)
    st.session_state.datos = datos
    st.session_state.respuestas = None
    st.session_state.retroalimentacion_generada = False
    st.session_state.rubrica_generada = False
    st.session_state.rubrica = None
    st.session_state.puntaje_total = 0

# === Etapa 1: Mostrar preguntas para responder ===
if st.session_state.preguntas and not st.session_state.retroalimentacion_generada:
    st.subheader("📝 Evaluación para el estudiante")

    respuestas_estudiante = []
    for i, pregunta in enumerate(st.session_state.preguntas):
        st.markdown(f"**{i+1}. {pregunta['enunciado']}**")

        if pregunta["tipo"] == "opcion_multiple":
            opciones = pregunta.get("opciones", [])
            opciones = [str(op).strip() for op in opciones]
            respuesta = st.radio("Selecciona una opción:", opciones, key=f"respuesta_{i}")
        else:
            respuesta = st.text_area("Escribe tu respuesta:", key=f"respuesta_{i}")

        respuestas_estudiante.append(respuesta)

    if st.button("🧠 Enviar evaluación"):
        st.session_state.respuestas = respuestas_estudiante
        st.session_state.retroalimentacion_generada = True

# === Etapa 2: Mostrar retroalimentación por pregunta + botón para generar rúbrica ===
if st.session_state.retroalimentacion_generada and not st.session_state.rubrica_generada:
    st.subheader("📋 Retroalimentación por pregunta")

    for i, (pregunta, respuesta) in enumerate(zip(st.session_state.preguntas, st.session_state.respuestas)):
        ref = pregunta.get("respuesta_correcta", "") if pregunta["tipo"] == "opcion_multiple" else pregunta.get("respuesta_esperada", "")
        retro = retroalimentar_respuesta(
            pregunta=pregunta["enunciado"],
            respuesta_correcta=ref,
            respuesta_estudiante=respuesta,
            api_key=clave_groq
        )
        st.markdown(f"**{i+1}. Retroalimentación:**")
        st.info(retro)

    if st.button("🎯 Generar puntaje y rúbrica"):
        actividad = "\n".join([p["enunciado"] for p in st.session_state.preguntas])
        st.session_state.rubrica = generar_rubrica(actividad, st.session_state.datos, clave_groq)

        puntaje = 0
        for pregunta, respuesta in zip(st.session_state.preguntas, st.session_state.respuestas):
            if pregunta["tipo"] == "opcion_multiple" and respuesta == pregunta.get("respuesta_correcta", ""):
                puntaje += 1

        st.session_state.puntaje_total = puntaje
        st.session_state.rubrica_generada = True

# === Etapa 3: Mostrar rúbrica y puntaje ===
if st.session_state.rubrica_generada:
    st.subheader("📏 Rúbrica generada por IA:")
    st.markdown(st.session_state.rubrica)

    st.success(f"✅ Puntaje obtenido por el estudiante: {st.session_state.puntaje_total} puntos")