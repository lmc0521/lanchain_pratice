from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import ConfigurableField
from langchain_openai import ChatOpenAI

llm = Ollama(model='llama3').configurable_alternatives(
    ConfigurableField(id="llm"),
    default_key='llama3',
    gpt35=ChatOpenAI(model="gpt-3.5-turbo", temperature=0),
)

# prompt = ChatPromptTemplate.from_messages([
#     ("user", "{input}"),
# ])

prompt = PromptTemplate.from_template(
    "Tell me a joke about {topic}"
).configurable_alternatives(
    ConfigurableField(id="prompt"),
    default_key="joke",
    poem=PromptTemplate.from_template("Write a short poem about {topic}"),
)

# chain.with_config(configurable={"prompt": "poem"}).invoke({"topic": "Earth"})

chain = prompt | llm

# print(chain.with_config(configurable={"llm": "llama3"}).invoke({'input': 'Tell me a joke'}))

print(chain.with_config(configurable={"llm": "llama3","prompt": "joke"}).invoke({"topic": "Earth"}))