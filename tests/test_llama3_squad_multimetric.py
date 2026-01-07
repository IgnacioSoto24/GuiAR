from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models.llms import OllamaModel

llm = OllamaModel(
    model="llama3:8b",
    temperature=0
)

question = "In what country is Normandy located?"
context = [
    "Normandy is a region in France. The Normans gave their name to Normandy during the 10th and 11th centuries."
]

# Respuesta generada
generated_answer = "Normandía se encuentra en Francia."

# Respuesta esperada
expected_answer = "France"

test_case = LLMTestCase(
    name="caso01_normans_country",
    input=question,
    actual_output=generated_answer,
    expected_output=expected_answer,
    context=context
)

metrics = [
    AnswerRelevancyMetric(
        threshold=0.7,
        model=llm
    ),
    FaithfulnessMetric(
        threshold=0.7,
        model=llm
    )
]

LINE_WIDTH = 60

print("\n" + "=" * LINE_WIDTH)
print("Test Case: caso01_normans_country")
print("Model: llama3:8b (Ollama)")
print("=" * LINE_WIDTH)

print("\nQuestion:")
print(question)

print("\nGenerated Answer:")
print(generated_answer)

print("\nContext:")
for c in context:
    print(f"- {c}")

print("\n" + "-" * LINE_WIDTH)
print(f"{'Metric':25} {'Score':>10} {'Status':>10}")
print("-" * LINE_WIDTH)

overall_pass = True

for metric in metrics:
    metric.measure(test_case)
    status = "PASS" if metric.score >= metric.threshold else "FAIL"
    if status == "FAIL":
        overall_pass = False
    print(f"{metric.__class__.__name__:25} {metric.score:>10.2f} {status:>10}")

print("-" * LINE_WIDTH)
print(f"{'Overall Success Rate':25} {('PASS' if overall_pass else 'FAIL'):>20}")
print("=" * LINE_WIDTH)
