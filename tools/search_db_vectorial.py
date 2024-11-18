"""
Usa esto para buscar en una base de datos vectorial, sin embargo te recomiendo que uses una base de datos SQL o no sql para almacenar las conversaciones. 
"""

from langchain.agents import tool
import requests
import os
from dotenv import load_dotenv
from urllib.parse import urljoin

load_dotenv()

API_URL = os.getenv("API_DATABASE_VECTORIAL_URL")
ENDPOINTS = {
    "base": "/db_vectorial",
    "list": "/list",
    "search": "/search",
    "exists": "/exists",
}


@tool
def exists_database(database_name: str) -> str:
    """Check if a vector database exists."""
    url = urljoin(API_URL, f"{ENDPOINTS['base']}{ENDPOINTS['exists']}/{database_name}")
    return str(requests.get(url).json())


@tool
def list_databases() -> str:
    """List vector databases."""
    url = "http://127.0.0.1:7860/db_vectorial/list"
    return str(requests.get(url).json())


@tool
def search_database(
    database_name: str, query: str, k: int = 5, source: str = None
) -> str:
    """Search in a vector database."""
    url = urljoin(API_URL, f"{ENDPOINTS['base']}{ENDPOINTS['search']}")
    data = {"database_name": database_name, "query": query, "k": k, "source": source}
    response = requests.post(url, json=data)
    return str(response.json())
