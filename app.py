from fastapi import FastAPI
from langchain.globals import set_verbose
from fastapi.middleware.cors import CORSMiddleware
from routers import ask, api_db_vectorial

set_verbose(True)  # Mensajes de depuración desactivados

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ask.router, prefix="/api")
app.include_router(api_db_vectorial.router, prefix="/api")