from fastapi import FastAPI
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableField
from langserve import add_routes

llm = Ollama(model='llama3').configurable_fields(
    temperature=ConfigurableField(
        id="temperature",
        name="LLM Temperature",
        description="The temperature of the LLM",
    ),
)

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a powerful assistant.'),
    ('user', '{input}'),
])

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    prompt | llm,
    path="/llama2",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=9000)