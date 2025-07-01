from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
import time
import chromadb
from chromadb.utils import embedding_functions
import json
import uvicorn
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Class Definition
class PythonKidsBot:
    """
    A chatbot for teaching Python to children with a FastAPI interface.
    """

    def __init__(self, persist_dir="web_scraped_data__/chroma_db", 
                 collection_name="python_for_kids", 
                 api_key=None):
        """Initialize the chatbot with required components"""
        self.api_key = api_key or "22C686MeYEWCtJZlh0rGqdQSnhPGPN9J"
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.model_name = "all-MiniLM-L6-v2"
        self.collection = None  # always define this!

        # Initialize embedding function
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.model_name
        )

        # Connect to ChromaDB
        self.setup_vector_store()
        if self.collection is None:
            raise RuntimeError("Failed to initialize ChromaDB collection. See previous error message.")

    def setup_vector_store(self):
        """Connect to the ChromaDB vector store"""
        try:
            self.client = chromadb.PersistentClient(path=self.persist_dir)
            # List all collections and check existence
            collection_names = [col.name for col in self.client.list_collections()]
            if self.collection_name in collection_names:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn
                )
                print(f"Connected to collection '{self.collection_name}' with {self.collection.count()} documents")
            else:
                print(f"Collection '{self.collection_name}' not found. Creating a new collection.")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=self.embedding_fn
                )
                print(f"Created a new collection '{self.collection_name}'")
        except Exception as e:
            print(f"Error connecting to ChromaDB: {e}")
            self.collection = None   # explicitly mark as failure

    def retrieve_context(self, question, n_results=3):
        """Retrieve relevant context for a question"""
        if self.collection is None:
            raise RuntimeError("ChromaDB collection is not set up. Cannot retrieve context.")
        results = self.collection.query(
            query_texts=[question],
            n_results=n_results
        )
        retrieved_chunks = []
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            retrieved_chunks.append({
                'content': doc,
                'metadata': metadata,
                'relevance': 1 - distance,
                'rank': i + 1
            })
        
        return retrieved_chunks

    def format_context(self, chunks):
        """Format chunks into context string for the LLM"""
        context_parts = [
            f"Source {i+1} [{chunk['metadata']['title']}]:\n{chunk['content']}\n"
            for i, chunk in enumerate(chunks)
        ]
        return "\n".join(context_parts)

    def generate_answer(self, question, context):
        """Generate an answer using Mistral API"""
        if not self.api_key:
            return {
                "answer": "No API key provided. Please set your Mistral API key.",
                "error": "No API key"
            }

        import requests

        url = "https://api.mistral.ai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        system_prompt = """You are a helpful, friendly AI assistant designed to teach Python programming to children.
Your answers should be:
1. Simple and easy to understand for children aged 8-12
2. Encouraging and positive
3. Accurate and based only on the provided context
4. Include simple examples when appropriate

If you don't know the answer based on the context, say so politely and suggest where they might find more information."""

        user_prompt = f"""Please answer the following question about Python programming for kids.
Use ONLY the information in the context provided below.

CONTEXT:
{context}

QUESTION: {question}"""

        payload = {
            "model": "mistral-small",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.5,
            "max_tokens": 500
        }

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return {
                "answer": result["choices"][0]["message"]["content"],
                "error": None
            }
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error when trying to generate a response: {e}",
                "error": str(e)
            }

    def answer_question(self, question, n_chunks=3):
        """Full RAG pipeline to answer a question"""
        chunks = self.retrieve_context(question, n_results=n_chunks)
        context = self.format_context(chunks)
        response = self.generate_answer(question, context)
        return {
            'question': question,
            'answer': response["answer"],
            'error': response.get("error"),
            'context_chunks': chunks,
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }

# FastAPI Application
app = FastAPI(title="Python Teacher Bot for Kids")

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the bot
bot = None

# Pydantic models for API
class QuestionRequest(BaseModel):
    question: str
    show_context: bool = False

class AnswerResponse(BaseModel):
    answer: str
    context_chunks: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

@app.on_event("startup")
async def startup_event():
    global bot
    api_key = "22C686MeYEWCtJZlh0rGqdQSnhPGPN9J"
    try:
        bot = PythonKidsBot(api_key=api_key)
    except Exception as e:
        print(f"Could not initialize the chatbot: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/ask", response_model=AnswerResponse)
async def ask_question(question_req: QuestionRequest):
    global bot
    if bot is None:
        raise HTTPException(status_code=500, detail="Chatbot not initialized")
    
    try:
        result = bot.answer_question(question_req.question)
        return {
            "answer": result["answer"],
            "context_chunks": result["context_chunks"] if question_req.show_context else None,
            "error": result.get("error")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the application
if __name__ == "__main__":
    uvicorn.run("fastapi_chatbot:app", host="0.0.0.0", port=8000, reload=True)