from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Depends
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

chain=prompt | llm

async def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # assuming the token is provided as a Bearer token
    api_key = authorization.split(" ")[1] if len(authorization.split(" ")) == 2 else None
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    if api_key != "valid_api_key":
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return {"user_name": "John"}

add_routes(
    app,
    chain,
    path="/llama2",
    dependencies=[Depends(verify_api_key)],
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=9000)