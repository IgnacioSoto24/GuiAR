from src.retriever import cargar_vectorstore

vs = cargar_vectorstore("historia")

query = "¿Cuándo ocurrió la Segunda Guerra Mundial?"
docs = vs.similarity_search(query, k=3)

for i, d in enumerate(docs, 1):
    print(f"\n--- FRAGMENTO {i} ---\n")
    print(d.page_content[:800])  # mostrar los primeros 800 caracteres
