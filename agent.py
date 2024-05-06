from langchain.agents import tool
import os

@tool
def getDocumentCharged(prompt, carpeta="./md_folder/"): 
    """Devuelve el numero de archivos cargados."""
    listFiles = os.listdir(carpeta)
    numFiles = len(listFiles) 
    return f"Hay cargados {numFiles} archivos"


## Para usar este ultimo Retriever, debes de quitar los archivos de md_folder, y meterle otros nuevos, y luego ya puedes llamar a esta funcion :)

# @tool
# def UpgradeRetriever(prompt):
#     """Ejecuta el script ingest.py para renovar los archivos cargados."""
#     os.system("python ingest.py")
#     return getDocumentCharged(prompt) 


# @tool
# def getHistorial(prompt):
#     """Devuelve el historial de preguntas y respuestas."""
#     from app import lastQuery
#     if lastQuery["query"] == "" and lastQuery["response"] == "":
#         return "No hay preguntas anteriores"
#     lastQuery["query"] = query
#     lastQuery["response"] = response
#     return f"La pregunta fue:{query} y la respuesta fue: {response}"

