# evaluador.py

from modelos.conexion_groq import conectar_groq
import json

def cargar_prompt(ruta_archivo):
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        return archivo.read()

# === Genera texto de actividad (antiguo)
def generar_actividad(datos_formulario, api_key):
    cliente = conectar_groq(api_key)
    prompt_template = cargar_prompt("prompts/generar_tarea.txt")

    prompt = prompt_template.format(
        asignatura=datos_formulario["asignatura"],
        nivel=datos_formulario["nivel"],
        objetivo_aprendizaje=datos_formulario["objetivo_aprendizaje"],
        tipo_evaluacion=datos_formulario["tipo_evaluacion"],
        habilidad=datos_formulario["habilidad"],
        formato=datos_formulario["formato"]
    )

    respuesta = cliente.chat.completions.create(
        model="mistral-saba-24b",
        messages=[{"role": "user", "content": prompt}]
    )

    return respuesta.choices[0].message.content

# === Genera rúbrica automática
def generar_rubrica(actividad_generada, datos_formulario, api_key):
    cliente = conectar_groq(api_key)
    prompt_template = cargar_prompt("prompts/generar_rubrica.txt")

    prompt = prompt_template.format(
        actividad_generada=actividad_generada,
        asignatura=datos_formulario["asignatura"],
        nivel=datos_formulario["nivel"],
        tipo_evaluacion=datos_formulario["tipo_evaluacion"],
        habilidad=datos_formulario["habilidad"],
        formato=datos_formulario["formato"]
    )

    respuesta = cliente.chat.completions.create(
        model="mistral-saba-24b",
        messages=[{"role": "user", "content": prompt}]
    )

    return respuesta.choices[0].message.content

# === Genera retroalimentación a partir de la respuesta del estudiante
def retroalimentar_respuesta(pregunta, respuesta_correcta, respuesta_estudiante, api_key):
    cliente = conectar_groq(api_key)
    prompt_template = cargar_prompt("prompts/retroalimentar.txt")

    prompt = prompt_template.format(
        pregunta=pregunta,
        respuesta_correcta=respuesta_correcta,
        respuesta_estudiante=respuesta_estudiante
    )

    respuesta = cliente.chat.completions.create(
        model="mistral-saba-24b",
        messages=[{"role": "user", "content": prompt}]
    )

    return respuesta.choices[0].message.content

# === Genera preguntas estructuradas en formato JSON
def generar_preguntas_estructuradas(datos_formulario, api_key):
    import re
    from modelos.conexion_groq import conectar_groq
    import json

    cliente = conectar_groq(api_key)
    prompt_template = cargar_prompt("prompts/generar_tarea_json.txt")

    prompt = prompt_template.format(
        asignatura=datos_formulario["asignatura"],
        nivel=datos_formulario["nivel"],
        objetivo_aprendizaje=datos_formulario["objetivo_aprendizaje"],
        tipo_evaluacion=datos_formulario["tipo_evaluacion"],
        habilidad=datos_formulario["habilidad"],
        formato=datos_formulario["formato"]
    )

    respuesta = cliente.chat.completions.create(
        model="mistral-saba-24b",
        messages=[{"role": "user", "content": prompt}]
    )

    texto = respuesta.choices[0].message.content.strip()

    # 🧼 Extraer bloque JSON limpio (entre corchetes)
    match = re.search(r"\[(\s*{.*?})\s*\]", texto, re.DOTALL)
    if match:
        json_bruto = "[" + match.group(1) + "]"
    else:
        json_bruto = texto

    # ✅ Intentar cargar JSON
    try:
        preguntas = json.loads(json_bruto)
    except json.JSONDecodeError as e:
        print("❌ Error al leer el JSON generado por la IA:")
        print(json_bruto)
        print("Error:", e)
        preguntas = []

    return preguntas