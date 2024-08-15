from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

# llm = Ollama(model='llama3')
llm=OpenAI()
# print(llm.invoke('Hi, how are you today?'))

prompt = ChatPromptTemplate.from_messages([
("system", "You are a content manager with extensive SEO knowledge. Your task is to write an article based on a given title."),
    ("user", "{input}"),
])

chain = prompt | llm

# print(chain.invoke({"input": "How does software change the world?"}))
print(chain.invoke({"input": "How does software change the world? use chinese to reply me"}))