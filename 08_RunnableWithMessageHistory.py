from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.llms.ollama import Ollama
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableWithMessageHistory, ConfigurableFieldSpec

chat001 = ChatMessageHistory()
chat001.add_user_message('My name is Amo.')

chat002 = ChatMessageHistory()
chat002.add_user_message('My name is Jeff.')

store = {
    ('amo', 'chat001'): chat001,
    ('jeff', 'chat002'): chat002,
}


def get_chat_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    key = (user_id, conversation_id, )
    if key not in store:
        store[key] = ChatMessageHistory()
    return store[key]

llm = Ollama(model='llama3')

prompt = ChatPromptTemplate.from_messages([
    ('system', 'You are a good assistant.'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('user', '{input}'),
])

chain = prompt | llm

with_message_history = RunnableWithMessageHistory(
    chain,
    get_chat_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User Id",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation Id",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

user_id = 'amo'
conversation_id = 'chat001'
input_text = input('>>> ')
while input_text.lower() != 'bye':
    if input_text:
        response = with_message_history.with_config(
            configurable={
                'user_id': user_id,
                'conversation_id': conversation_id
            }
        ).invoke({'input': input_text})
        print(response)
    input_text = input('>>> ')