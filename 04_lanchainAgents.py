import requests
from langchain import hub
from langchain_core.messages import HumanMessage, AIMessage

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import tool
from langchain.agents import AgentExecutor


@tool
def check_site_alive(site: str) -> bool:
    """Check a site is alive or not."""
    try:
        resp = requests.get(f'https://{site}')
        resp.raise_for_status()
        return True
    except Exception:
        return False


tools = [check_site_alive, ]


llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         "You are very powerful assistant, but don't know current events",
#     ),
#     MessagesPlaceholder(variable_name="chat_history"),
#     ("user", "{input}"),
#     MessagesPlaceholder(variable_name="agent_scratchpad"),
# ])
prompt = hub.pull("hwchase17/openai-functions-agent")

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chat_history=[]
input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        response = agent_executor.invoke({
            'input': input_text,
            'chat_history': chat_history,
        })
        chat_history.extend([
            HumanMessage(content=input_text),
            AIMessage(content=response["output"]),
        ])
        print(response['output'])
    input_text = input('>>> ')