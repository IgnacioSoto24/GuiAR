from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

def construir_vectorstore(ruta_pdf, ruta_guardado="faiss_index"):
    cargador = PyPDFLoader(ruta_pdf)
    documentos = cargador.load()

    divisor = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    fragmentos = divisor.split_documents(documentos)

    incrustaciones = OllamaEmbeddings(model="mistral")

    vectorstore = FAISS.from_documents(fragmentos, incrustaciones)
    vectorstore.save_local(ruta_guardado)

    print(f"✅ Vectorstore guardado en: {ruta_guardado}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Uso: python -m src.ingestion <ruta_pdf>")
    else:
        construir_vectorstore(sys.argv[1])
