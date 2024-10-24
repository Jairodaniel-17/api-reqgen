from datetime import datetime
from langchain.agents import tool


@tool
def time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current system time as a formatted string.

    Args:
        format (str): The format for the output time string. Default is "%Y-%m-%d %H:%M:%S".

    Returns:
        str: The current system time in the specified format.
    """

    current_time = datetime.now().strftime(format)
    print(f"Current system time: {current_time}")
    return current_time
