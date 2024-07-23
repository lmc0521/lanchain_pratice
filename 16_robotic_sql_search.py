from langchain_community.llms import Ollama
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from sqlalchemy.exc import OperationalError

llm = Ollama(model='llama2')


db = SQLDatabase.from_uri("sqlite:///./northwind.db")


def get_db_schema(_):
    return db.get_table_info()


def run_query(query):
    try:
        return db.run(query)
    except (OperationalError, Exception) as e:
        return str(e)


gen_sql_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Based on the table schema below, write a SQL query that would answer the user\'s question: {db_schema}'),
    ('user', 'Please generate a SQL query for the following question: "{input}". \
     The query should be formatted as follows without any additional explanation: \
     SQL> <sql_query>\
    '),
])


class SqlQueryParser(StrOutputParser):
    def parse(self, s):
        # print(s)
        r = s.split('SQL> ')
        # print(r)
        if len(r) > 0:
            return r[1]
        return s


gen_query_chain = (
    RunnablePassthrough.assign(db_schema=get_db_schema)
    | gen_sql_prompt
    | llm
    | SqlQueryParser()
)


gen_answer_prompt = ChatPromptTemplate.from_template("""
Based on the provided question, SQL query, and query result, write a natural language response.
No additional explanations should be included.

Question: {input}
SQL Query: {query}
Query Result: {result}

The response should be formatted as follows:
'''
Executed: {query}
Answer: <answer>
'''
""")


chain = (
    RunnablePassthrough.assign(query=gen_query_chain).assign(
        result=lambda x: run_query(x["query"]),
    )
    | gen_answer_prompt
    | llm
)

input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        response = chain.invoke({
            'input': input_text,
        })
        print(response)
    input_text = input('>>> ')