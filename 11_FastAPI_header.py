from typing import Optional

from fastapi import FastAPI, Header, HTTPException

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A simple api server using Langchain's Runnable interfaces",
)

def get_user_from_api_key(api_key: str) -> Optional[dict]:
    if api_key == "valid_api_key":
        return {"user_id": "useridx", "user_name": "John"}
    return None

@app.get("/user")
async def get_user(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")

    # assuming the token is provided as a Bearer token
    api_key = authorization.split(" ")[1] if len(authorization.split(" ")) == 2 else None
    if api_key is None:
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    user_data = get_user_from_api_key(api_key)
    if user_data is None:
        raise HTTPException(status_code=403, detail="Invalid API Key")

    return {"user_name": user_data["user_name"],"user_id": user_data["user_id"]}

if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="localhost", port=9000)