import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualRecallMetric,
    ToxicityMetric,
    ArgumentCorrectnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval import evaluate
from src.rag_pipeline import obtener_cadena_rag


def test_sqac_subset():
    """
    Evalúa el rendimiento del sistema GuiAR usando un subconjunto del dataset SQuAD español (archivo local).
    """

    tutor_guiar = obtener_cadena_rag()
    modelo_eval = OllamaModel(model="mistral")

    print("\n📥 Cargando dataset local 'squad_es_validation.json'...")
    dataset_path = os.path.join("data", "squad_es_validation.json")

    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        dataset = []
        for article in raw_data["data"][:2]:
            for paragraph in article["paragraphs"]:
                contexto = paragraph["context"]
                for qa in paragraph["qas"][:5]:
                    pregunta = qa["question"]
                    respuesta = qa["answers"][0]["text"] if qa["answers"] else "Sin respuesta"
                    dataset.append({
                        "context": contexto,
                        "question": pregunta,
                        "answers": {"text": [respuesta]}
                    })

        print(f"✅ Dataset local cargado correctamente ({len(dataset)} ejemplos).\n")

    except Exception as e:
        print("⚠️ Error al cargar dataset local:", e)
        raise RuntimeError("❌ Error: No se pudo cargar el archivo local 'squad_es_validation.json'.")

    casos_prueba = []
    for i, ejemplo in enumerate(dataset):
        pregunta = ejemplo["question"]
        contexto = ejemplo["context"]
        respuesta = ejemplo["answers"]["text"][0]
        print(f"🧩 Procesando ejemplo {i+1}: {pregunta[:80]}...")

        salida_tutor = tutor_guiar.run(pregunta)

        caso = LLMTestCase(
            input=pregunta,
            actual_output=salida_tutor,
            expected_output=respuesta,
            retrieval_context=[contexto],
            tools_called=[],
        )
        casos_prueba.append(caso)

    metricas = [
        FaithfulnessMetric(model=modelo_eval),
        AnswerRelevancyMetric(model=modelo_eval),
        ContextualRecallMetric(model=modelo_eval),
        ToxicityMetric(model=modelo_eval),
        ArgumentCorrectnessMetric(model=modelo_eval),
    ]

    print("\n⚙️ Ejecutando evaluación con DeepEval...")
    resultado = evaluate(casos_prueba, metricas)

    resultados_dict = {}
    for metrica in resultado.metrics_data:
        resultados_dict[metrica.name] = {
            "score": metrica.score,
            "threshold": metrica.threshold,
            "reason": metrica.reason,
        }

    os.makedirs("Metricas", exist_ok=True)
    ruta_salida = os.path.join("Metricas", "resultados_sqac.json")
    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(resultados_dict, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Resultados guardados en: {ruta_salida}")
    print("🎯 Evaluación completada correctamente.\n")

    assert True


if __name__ == "__main__":
    test_sqac_subset()