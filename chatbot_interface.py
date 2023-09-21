import streamlit as st
from streamlit_chat import message
import llm

st.set_page_config(page_title="Welcome to the World ChatDPO!!", page_icon="💬")

# Page title
st.markdown("# ChatBot - How can I help you?")

# Sidebar
st.sidebar.header("ChatDPO")


def clear_text_input():
    st.session_state["question"] = st.session_state["input"]
    st.session_state["input"] = ""


def clear_chat_data():
    st.session_state["input"] = ""
    st.session_state["chat_history"] = []


# Initialize chat_history
if "question" not in st.session_state:
    st.session_state["question"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

llm_helper = llm.LLMHelper()

# Chat
st.text_input(
    "Digite sua pergunta aqui: ",
    placeholder="digite sua pergunta",
    key="input",
    on_change=clear_text_input,
)

clear_chat = st.button("Limpar o chat", key="clear_chat", on_click=clear_chat_data)

if st.session_state['question']:
    question, result = llm_helper.get_semantic_search_conversational_chain(st.session_state['question'], st.session_state['chat_history'])
    st.session_state['chat_history'].append((question, result))

if st.session_state['chat_history']:
    for i in range(len(st.session_state['chat_history'])-1, -1, -1):
        message(st.session_state['chat_history'][i][1], key=str(i))
        # st.markdown(f'\n\nSources: {st.session_state["source_documents"][i]}')
        message(st.session_state['chat_history'][i][0], is_user=True, key=str(i) + '_user')
