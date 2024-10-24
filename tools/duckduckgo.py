from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.agents import tool


@tool
def search(query: str) -> str:
    """Búsqueda en DuckDuckGo."""
    wrapper = DuckDuckGoSearchAPIWrapper(max_results=10, safesearch="off", time="w")
    search = DuckDuckGoSearchResults(api_wrapper=wrapper)
    return str(search.invoke(query))
