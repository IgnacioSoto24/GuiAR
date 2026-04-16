import os
from typing import Any, List
from pydantic import Field
from langchain.schema import BaseRetriever
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder


class RerankRetriever(BaseRetriever):
    

    vectorstore: Any = Field(...)
    k: int = Field(default=4)
    fetch_k: int = Field(default=20)
    reranker: Any = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def _get_relevant_documents(self, query: str) -> List[Any]:
        # 1) buscar documentos
        docs = self.vectorstore.similarity_search(query, k=self.fetch_k)
        if not docs:
            return []

        # 2) preparar pares para ranking
        pares = [[query, d.page_content] for d in docs]
        scores = self.reranker.predict(pares)

        # 3) ordenar
        rerankeados = list(zip(scores, docs))
        rerankeados.sort(key=lambda x: x[0], reverse=True)

        # 4) devolver los mejores
        return [doc for _, doc in rerankeados[:self.k]]


def cargar_vectorstore(asignatura: str):
    ruta = os.path.join("vectorstores", asignatura, "faiss_index")

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe vectorstore para '{asignatura}'. Sube un PDF primero."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return FAISS.load_local(
        ruta,
        embeddings,
        allow_dangerous_deserialization=True
    )


def crear_retriever(asignatura: str):
    vectorstore = cargar_vectorstore(asignatura)
    return RerankRetriever(
        vectorstore=vectorstore,
        k=4,
        fetch_k=20,
    )
