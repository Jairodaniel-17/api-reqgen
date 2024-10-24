from langchain.agents import tool
from langchain.globals import set_verbose

from database_vectorial.model_db import VectorialDB

set_verbose(True)  # Mensajes de depuración desactivados
vectorial_db = VectorialDB()


@tool
def list_databases():
    """List vector databases."""
    list_databases = vectorial_db.list_vector_stores()
    dict_list_databases = {i: database for i, database in enumerate(list_databases)}
    return str(dict_list_databases)


@tool
def search_database(database_name: str, query: str, k: int = 3, source: str = None):
    """Search similarity in a vector database.

    Args:
        database_name (str): Name of the vector database.
        query (str): Query.
        k (int, optional): Number of results. Defaults to 3.
        source (str, optional): Source of the documents. Defaults to None.

    Returns:
        list: Search results.
    """
    try:
        results = vectorial_db.search_similarity(
            name=database_name, query=query, k=k, source=source
        )
        return str(results)
    except Exception as e:
        return str(e)
