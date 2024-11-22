import argparse
# from dataclasses import dataclass
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
# from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from dotenv import load_dotenv
import streamlit as st
import os


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# Load environment variables from .env file
load_dotenv()

# Initiate LLM => Default 'gpt-3.5-turbo' will be used
model = ChatOpenAI()
# Use GPT-4 model
# model = ChatOpenAI(model="gpt-4")


# Set Streamlit pages config
st.set_page_config(page_title="Chat with Preloaded index", page_icon="📚")

# CSS to inject into the app
page_bg_color = """
    <style>
    .stApp {
        background-color: rgb(240, 242, 246);
    }
    </style>
    """

# Inject the CSS into the Streamlit app
st.markdown(page_bg_color, unsafe_allow_html=True)

st.title("RAG using LangChain and chroma")
# query_text = st.text_input("Ask questions about Alice in Wonderland")
# st.write(f"Your Question is: {query_text}")

# Now you can access the API key in your code
openai_api_key = os.getenv('OPENAI_API_KEY')


class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.status = container.status("**Context Retrieval**")

    def on_retriever_start(self, serialized: dict, query: str, **kwargs):
        self.status.write(f"**Question:** {query}")
        self.status.update(label=f"**Context Retrieval:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = os.path.basename(doc.metadata["source"])
            self.status.write(f"**Document {idx} from {source}**")
            self.status.markdown(doc.page_content)
        self.status.update(state="complete")

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container: st.delta_generator.DeltaGenerator, initial_text: str = ""):
        self.container = container
        self.text = initial_text
        self.run_id_ignore_token = None

    def on_llm_start(self, serialized: dict, prompts: list, **kwargs):
        # Workaround to prevent showing the rephrased question as output
        if prompts[0].startswith("Human"):
            self.run_id_ignore_token = kwargs.get("run_id")

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.run_id_ignore_token == kwargs.get("run_id", False):
            return
        self.text += token
        self.container.markdown(self.text)        

if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")
    st.stop()

username = st.sidebar.text_input(label="Enter your username")

if not username:
    st.info("Please enter your username to continue.")
    st.stop()

# Prepare the DB.
embedding_function = OpenAIEmbeddings()
vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

db_path = 'local_sqlite_db.db'
msgs = SQLChatMessageHistory(
    session_id=username,
    connection_string="sqlite:///" + db_path  # This is the SQLite connection string
)

memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

# Create the Conversational Retrieval Chain
retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vector_store.as_retriever(),
    memory=memory  # Optional: Add memory for multi-turn conversations
)     

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)    

if user_query := st.chat_input(placeholder="Ask me anything!"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = retrieval_chain.run(user_query, callbacks=[retrieval_handler, stream_handler])
        st.write(response)




