# Retrieval Augmented Generation RAG Chatbot
## Solution Approach
We created a Retrieval-Augmented Generation (RAG) chatbot to teach kids Python in an engaging, understandable way.

## Our System:
📥 Retrieves relevant educational content from child-friendly programming resources

🧽 Cleans & splits it into meaningful educational chunks

✏️ Generates answers using a language model that explains concepts simply

🎉 Delivers responses in a supportive, kid-friendly tone

## Technical Implementation
## 1. Data Collection

Sources:
* Programiz.com
* Python for Kids (PDF-style materials)
* Official Python documentation

Method:
* requests + BeautifulSoup for web scraping
* Custom PDF readers (if used)

Topics Covered: Variables, data types, conditions, loops, functions

Dataset Size: ~50,000+ words scraped and processed

## Text Preprocessing
1. Cleaning:
* Remove headers, footers, ads
* Normalize whitespace and symbols
 
2. Chunking:
* Tool: RecursiveCharacterTextSplitter (LangChain)
* Chunk size: 500 tokens
* Overlap: 100 tokens
* Ensures semantic continuity

Output:

✅ 1,000+ clean, coherent text chunks

Each chunk has metadata (title, word count, source)

3. Vector Embedding
* Embedding Model: all-MiniLM-L6-v2 via sentence-transformers
* Embedding dimension: 384
* Process: Batched embedding of all chunks
* Normalized for semantic similarity

4. Vector Database
* Backend: ChromaDB (In-Memory Vector Store used for fast prototyping)

Stored Info:
* Text chunk
* Source URL/Title
* Word count
* Vector representation

Functionality: Fast semantic search to retrieve relevant chunks

## Question Answering System (RAG)
Step 1: Query Embedding
Encode the child’s question using the same model (MiniLM)

Step 2: Retrieval
Find top 3–5 matching chunks using vector similarity

Step 3: Context Preparation
Combine the chunks into a readable passage and Format input for the language model

Step 4: Answer Generation
Model: mistral-small via [Mistral AI API](https://docs.mistral.ai/)

Prompt style:

* Encouraging tone with analogies
* Parameters: max_length=100, temperature=0.5
* Fallback Handling: Graceful response when nothing is found
* Encouraging message: "That's a great question! Let's learn about it together."

## Web Interface (FastAPI)
* Frontend: HTML + CSS form (kid-friendly UI)

* Backend: FastAPI app that:
1. Accepts a question from the form
2. Passes it to the RAG logic (retrieve_answer())
3. Returns the answer to the HTML page

* Frameworks Used:
1. FastAPI for backend
2. Jinja2 for templating
3. Uvicorn as the server

## How to run Locally:
* Install dependencies:
```bash
pip install fastapi uvicorn jinja2 beautifulsoup4 requests sentence-transformers chromadb transformers langchain
