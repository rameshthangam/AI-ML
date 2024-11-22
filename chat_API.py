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
from typing import Dict

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
    def __init__(self, initial_text: str = ""):
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

 


   
    # Generate a response using the chatbot logic
    # response_message = generate_response(user_message)
    
    # return {"response": response_message}
   
def generate_response(message: str) -> str:
    # print('request_body :', request_body)
    user_message = message
    if not user_message:
        return {"error": "No message provided"}, 400  # Bad Request
    
    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

    # Load environment variables from .env file
    load_dotenv()

    
    CHROMA_PATH = "chroma"
    
    # Initiate LLM => Default 'gpt-3.5-turbo' will be used
    model = ChatOpenAI()
    # Use GPT-4 model
    # model = ChatOpenAI(model="gpt-4")

    # Now you can access the API key in your code
    openai_api_key = os.getenv('OPENAI_API_KEY')

       # Prepare the DB.
    embedding_function = OpenAIEmbeddings()
    vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    db_path = 'local_sqlite_db.db'
    msgs = SQLChatMessageHistory(
        session_id="ramesh",
        connection_string="sqlite:///" + db_path  # This is the SQLite connection string
    )

    memory = ConversationBufferMemory(memory_key="chat_history", chat_memory=msgs, return_messages=True)

    # Create the Conversational Retrieval Chain
    retrieval_chain = ConversationalRetrievalChain.from_llm(
    llm=model,
    retriever=vector_store.as_retriever(),
    memory=memory  # Optional: Add memory for multi-turn conversations
    )  

    # retrieval_handler = PrintRetrievalHandler()
    # stream_handler = StreamHandler()

    # response = retrieval_chain.run(user_message, callbacks=[retrieval_handler, stream_handler])
    print(f"User message: {user_message}")
    response = retrieval_chain.run(user_message)
    
    return response