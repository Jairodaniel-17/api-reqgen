import os
from typing import Sequence
from dotenv import load_dotenv
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts.prompt import PromptTemplate
from langchain.tools.render import render_text_description
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_log_to_str

from langchain.globals import set_verbose

set_verbose(True)  # Mensajes de depuración desactivados


class ModelAI:
    def __init__(self):
        load_dotenv()
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com"
        self.api_key = os.getenv("DEEPSEEK_API_KEY")

    def agent_executer(self, tools: Sequence[BaseTool]) -> AgentExecutor:
        """
        Create an agent executor with the given tools and the model.

        Args:
            tools: A sequence of tools to be used by the agent.

        Returns:
            An agent executor with the given tools and the model.
        """

        llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=0.5,
        )
        memory = ConversationBufferMemory(memory_key="chat_history")
        prompt = self._load_prompt("prompt_system_agent.txt")

        agent_prompt = PromptTemplate.from_template(prompt)
        prompt = agent_prompt.partial(
            tools=render_text_description(tools),
            tool_names=", ".join([t.name for t in tools]),
        )

        agent = self._create_agent(llm, prompt)
        return AgentExecutor(agent=agent, tools=tools, memory=memory)

    @staticmethod
    def _load_prompt(filepath: str) -> str:
        with open(filepath, "r") as file:
            return file.read()

    @staticmethod
    def _create_agent(llm: ChatOpenAI, prompt: PromptTemplate) -> dict:
        llm_with_stop = llm.bind(stop=["\nObservation"])
        return (
            {
                "input": lambda x: x["input"],
                "agent_scratchpad": lambda x: format_log_to_str(
                    x["intermediate_steps"]
                ),
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt
            | llm_with_stop
            | ReActSingleInputOutputParser()
        )
