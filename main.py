from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from chat_API import generate_response

app = FastAPI()

# Load the tokenizer, retriever, and model
# tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
# retriever = RagRetriever.from_pretrained("facebook/rag-token-nq")
# model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq")

class ChatRequest(BaseModel):
    message:str

@app.get("/chat")    
def get_sample():
    return "hello API"

@app.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

@app.post("/chat")
def generate(request: ChatRequest):

    # Generate Output
    # outputs = model.generate(input_ids, context_input_ids=retrieved_docs)
    # generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

    print('request :', request.message)
    response = generate_response(request.message)
    return {"response": response}

# Run the API with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)