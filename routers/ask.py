from fastapi import APIRouter
from langchain.globals import set_verbose
from controller.question import Question
from model.model_ai import ModelAI
from tools.duckduckgo import search
from tools.time import time

set_verbose(True)  # Mensajes de depuración desactivados

router = APIRouter(
    prefix="/IA",
    tags=["Ask Agent"],
    responses={404: {"description": "No encontrado"}},
)
model = ModelAI()
# Se añade las herramientas de tiempo y búsqueda al agente
tools = [time, search]
agent_executor = model.agent_executer(tools)


@router.get("/ask")
def ask_question():
    """Devuelve el formato de pregunta esperado."""
    return {"input": "Pregunta"}


@router.post("/ask")
async def ask_question(question: Question):
    """Recibe una pregunta y devuelve una respuesta del agente."""
    try:
        # Usamos ainvoke para realizar la llamada asíncrona
        respuesta = await agent_executor.ainvoke({"input": question.input})
        return {"respuesta": respuesta["output"]}
    except Exception as e:
        return {"error": str(e)}
