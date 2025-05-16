# modelos/conexion_groq.py

from groq import Groq

def conectar_groq(api_key):
    return Groq(api_key=api_key)
