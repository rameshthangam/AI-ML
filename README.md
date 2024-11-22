1. Environment Setup

Create a .env file in the project directory and add:
OPENAI_API_KEY: Your OpenAI API key
Install dependencies: pip install chroma

2. Creating Embeddings

Run create_database.py to read the .md files under data/books, split the same embed PDF documents in files/ and store them in a Pinecone index:
python create_embeddings.py

3. Running the Chat App

Run chat.py to start the Streamlit chat interface:
streamlit run chat.py
4. Using the Chat App

Enter your username in the sidebar.
Ask questions or make statements in the chat box.
The app retrieves relevant information and generates responses using a large language model.
Clear the chat history with the sidebar button.
5. SQL History and Database

Chat history is stored in local_sqlite_db.db for persistence and context.
This allows for more personalized and consistent conversations.
Additional Notes

Ensure PDFs exist in files/ before running embeddings.py.
The app uses gpt-3.5-turbo from OpenAI, requiring an OpenAI API key.
Adjust embeddings model and retrieval parameters for fine-tuning.