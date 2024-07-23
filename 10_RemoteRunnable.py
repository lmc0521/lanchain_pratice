from langserve import RemoteRunnable

llama2 = RemoteRunnable("http://localhost:9000/llama2/")
response = llama2.invoke({"input": "Hi there"})
print(response)