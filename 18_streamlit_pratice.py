import streamlit as st
from langchain_community.llms.ollama import Ollama
from langchain_core.prompts import ChatPromptTemplate

ROLE_PYTHON_DEVELOPER = 'Senior Python Developer'
TMPL_NAMING = 'Naming'
TMPL_CODE_REVIEW = 'Code Review'

ROLE_ENGLISH_TEACHER = 'Professional English Teacher'
TMPL_CORRECT_GRAMMAR = 'Correct English Grammar'
TMPL_CORRECT_GRAMMAR_WITH_DESCR = 'Correct English Grammar (with additional description)'

TMPL_NONE = 'None'

llm = Ollama(model='llama2')

st.title('Ask Me Anything')

role = st.selectbox(
   "Which AI role would you like to ask?",
   (ROLE_PYTHON_DEVELOPER, ROLE_ENGLISH_TEACHER),
   index=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a {role}, you job is answering user's question.",
    ),
    ("user", "{input}"),
])

chain= prompt | llm

def generate_response(text):
    # return llm.invoke(text)
    for r in chain.stream({'input':text,'role':role}):
        yield r

template = TMPL_NONE
if role == ROLE_PYTHON_DEVELOPER:
    template = st.radio(
        "Templates",
        [
            TMPL_NAMING,
            TMPL_CODE_REVIEW,
            TMPL_NONE,
        ],
        horizontal=True
    )
elif role == ROLE_ENGLISH_TEACHER:
    template = st.radio(
        "Templates",
        [
            TMPL_CORRECT_GRAMMAR_WITH_DESCR,
            TMPL_CORRECT_GRAMMAR,
            TMPL_NONE,
        ],
        horizontal=True
    )

def use_template(text):
    if template == TMPL_NONE:
        return text
    if template == TMPL_NAMING:
        return f'Is "{text}" a good variable name in Python?'
    if template == TMPL_CODE_REVIEW:
        return (
            "Please assist me in reviewing the following code snippet. "
            "Two hard rules apply: "
            "1. Function names and variables must be clear. "
            "2. There should be no performance issues. "
            "If the code appears clear and efficient, simply respond with 'LGTM'. "
            "Otherwise, please identify any issues and provide additional explanation. "
            "The code snippet is:\n"
            f"'''\n{text}\n'''"
        )
    if template == TMPL_CORRECT_GRAMMAR_WITH_DESCR:
        return (
            "Please assist me in correcting the grammar issues. "
            "The text is:\n"
            f"'''\n{text}\n'''"
        )
    if template == TMPL_CORRECT_GRAMMAR:
        return (
            "Please assist me in correcting the grammar issues. "
            "You don't need to provide additional explanation. "
            "The text is:\n"
            f"'''\n{text}\n'''"
        )

with st.form('form'):
    text = st.text_area('Enter text:', '')
    submitted = st.form_submit_button('Submit')
    if submitted:
        # st.info(generate_response(text))
        resp = ""
        st.write_stream(generate_response(use_template(text)))