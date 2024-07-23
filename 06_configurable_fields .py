from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField

llm = Ollama(model='llama3', temperature=0).configurable_fields(
    temperature=ConfigurableField(
        id="llm_temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    ),
    model=ConfigurableField(
        id="model",
        name="The Model",
        description="The language model",
    ),
)
prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
])
chain = prompt | llm
print(chain.with_config(
        configurable={
            "model": "codellama",
            "llm_temperature": 0.9
        }
    ).invoke({'input': 'Tell me a joke'}))