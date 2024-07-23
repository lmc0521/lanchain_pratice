from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("user", "{input}"),
])

chain = RunnablePassthrough.assign(input=lambda x: x['input'] + ' this is important to me.') | prompt
print(chain.invoke({"input": "python is the best."}))