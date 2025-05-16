# Archivo: iniciar_evaluar.ps1

cd "C:\Users\Nacho\OneDrive\Documentos\EvaluAR"

# Activar conda (esto carga el entorno base de Anaconda)
& "C:\Users\Nacho\anaconda3\Scripts\activate" base

# Ejecutar la app
streamlit run app.py
