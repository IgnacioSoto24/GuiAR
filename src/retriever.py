import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def cargar_vectorstore(asignatura: str):
    """
    Carga el vectorstore FAISS exacto de la asignatura.
    Se asegura de usar los mismos embeddings que ingestion.py.
    """

    ruta = os.path.join("vectorstores", asignatura, "faiss_index")

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"❌ No existe vectorstore para la asignatura '{asignatura}'. "
            "Primero sube un PDF desde el panel izquierdo."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    try:
        vectorstore = FAISS.load_local(
            ruta,
            embeddings,
            allow_dangerous_deserialization=True
        )
    except Exception as e:
        raise RuntimeError(f"❌ Error cargando FAISS: {e}")

    return vectorstore


def crear_retriever(asignatura: str):
    """
    Crea un retriever configurado y seguro.
    """

    vs = cargar_vectorstore(asignatura)

    retriever = vs.as_retriever(
        search_kwargs={
            "k": 4,
            "fetch_k": 20
        }
    )

    return retriever
