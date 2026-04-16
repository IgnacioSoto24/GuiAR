import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def construir_vectorstore(ruta_pdf: str, asignatura: str):

    ruta_guardado = os.path.join("vectorstores", asignatura, "faiss_index")

    if os.path.exists(ruta_guardado):
        shutil.rmtree(ruta_guardado)

    os.makedirs(ruta_guardado, exist_ok=True)

    print(f"📘 Procesando PDF para asignatura: {asignatura}")

    try:
        loader = PyPDFLoader(ruta_pdf)
        documentos = loader.load()
    except Exception as e:
        raise RuntimeError(f"❌ Error cargando PDF: {e}")

    if len(documentos) == 0:
        raise RuntimeError("❌ El PDF está vacío o no se pudo leer texto.")

    for d in documentos:
        d.page_content = d.page_content.replace("\x00", " ").replace("\u200b", " ")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=450,
        chunk_overlap=30,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    fragmentos = splitter.split_documents(documentos)

    if len(fragmentos) == 0:
        raise RuntimeError("❌ No se pudieron generar fragmentos desde el PDF.")

    print(f"📄 Fragmentos creados: {len(fragmentos)}")

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    except Exception as e:
        raise RuntimeError(f"❌ Error cargando embeddings: {e}")

    try:
        vectorstore = FAISS.from_documents(fragmentos, embeddings)
        vectorstore.save_local(ruta_guardado)
    except Exception as e:
        raise RuntimeError(f"❌ Error construyendo FAISS: {e}")

    print(f"✔ Vectorstore guardado correctamente en {ruta_guardado}")

    return ruta_guardado
