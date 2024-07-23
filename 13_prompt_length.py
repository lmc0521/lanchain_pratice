from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage

llm = Ollama(model='llama2')
num = llm.get_num_tokens_from_messages([HumanMessage('Hi, there. How are you today?')])
print(f'Tokens: {num}')

num = llm.get_num_tokens('Hi, there. How are you today?')
print(f'Tokens: {num}')