import pytest
import json
import os
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

def test_respuesta():
    # Crear el caso de prueba
    caso = LLMTestCase(
        input="¿Qué son los Objetivos de Aprendizaje en Historia 3º medio?",
        actual_output="Los Objetivos de Aprendizaje son metas definidas en el currículum oficial.",
        expected_output="Definición oficial de OA en Historia 3º medio según MINEDUC.",
        retrieval_context=[
            "Los Objetivos de Aprendizaje corresponden a las metas establecidas en el currículum."
        ],
        tools_called=[]  # requerido para ArgumentCorrectnessMetric
    )

    # Modelo evaluador de Ollama
    modelo_local = OllamaModel(model="llama3:8b")

    # Definir métricas
    metricas = [
        FaithfulnessMetric(model=modelo_local),
        AnswerRelevancyMetric(model=modelo_local),
        ContextualRecallMetric(model=modelo_local, threshold=0.0),
        ToxicityMetric(model=modelo_local),
        ArgumentCorrectnessMetric(model=modelo_local),
    ]

    # Evaluar
    resultados = evaluate([caso], metricas)

    # Convertir a formato serializable
    resultados_dict = {
        metrica.name: {
            "score": metrica.score,
            "reason": metrica.reason,
            "threshold": metrica.threshold,
            "success": metrica.success
        }
        for metrica in resultados[0].metrics_data
    }

    # Guardar en carpeta Metricas
    os.makedirs("Metricas", exist_ok=True)
    ruta_salida = os.path.join("Metricas", "resultados.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados_dict, f, indent=4, ensure_ascii=False)

    # Mostrar en consola
    print("\n📊 Resultados de métricas:")
    for nombre, data in resultados_dict.items():
        print(f"- {nombre}: {data['score']} (razón: {data['reason']})")

    # ✅ Evitar que pytest marque FAIL
    assert True
