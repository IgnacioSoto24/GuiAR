from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

def cargar_vectorstore(ruta="faiss_index"):
    incrustaciones = OllamaEmbeddings(model="mistral")
    return FAISS.load_local(ruta, incrustaciones, allow_dangerous_deserialization=True)
