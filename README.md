📘 Cognify: Document-Aware Chatbot with RAG + Gemini
Cognify is an intelligent document assistant that lets you upload PDFs and chat with them. Using Retrieval-Augmented Generation (RAG), semantic search, and Google Gemini, Cognify transforms static documents into interactive, conversational knowledge sources.
⚡ Features

📂 Upload PDFs and parse them into structured chunks with GroundX.
🔍 Semantic Search using SentenceTransformers + ChromaDB.
🤖 AI Answers powered by Google Gemini.
💬 Chat-like UI for interactive Q&A.
🎨 Frontend with theme toggle and chat history reset.
🏗️ Architecture
The project is structured into three layers:

1)Pipeline Layer (rag_pipeline/pipeline.py)

Document parsing (GroundX).
Chunking + embedding with SentenceTransformers.
Vector storage in ChromaDB.
Retrieval + answer generation with Gemini.
2)Backend Layer (app.py)

FastAPI backend with two main endpoints:
/upload → handles file ingestion.
/ask → handles Q&A over ingested documents.
CORS enabled for frontend integration.

3)Frontend Layer (index.html, style.css)

Simple, responsive web interface.
PDF upload button and chat interface.
JS fetch calls to FastAPI backend.
Theme toggle (light/dark).

🚀 Getting Started
1. Clone Repo

git clone https://github.com/yourusername/cognify-chatbot.git

cd cognify-chatbot

2. Install Dependencies

pip install -r requirements.txt

Suggested requirements.txt:
fastapi
uvicorn
chromadb
sentence-transformers
groundx
google-generativeai

3. Environment Variables

Set your API keys:
export GEMINI_API_KEY="your_gemini_key"
export GROUNDX_API_KEY="your_groundx_key"
(or create a .env file if you prefer).

4. Run Backend

uvicorn app:app --reload
Backend runs at: http://127.0.0.1:8000

5. Open Frontend

Open index.html in your browser
Or visit the FastAPI root / which serves index.html

📂 Project Structure

.
├── rag_pipeline/
│ └── pipeline.py # PDF parsing, vector DB, retrieval, answer generation
├── app.py # FastAPI backend
├── index.html # Frontend UI
├── style.css # Optional styling
├── chatbot.png # Logo for UI
└── requirements.txt # Python dependencies

🧠 Workflow

Upload a PDF → GroundX parses into JSON → stored in ChromaDB.
Ask a question → system retrieves relevant chunks → Gemini generates an answer.
Answer displayed in a chat-style interface.

🔮 Future Enhancements

Add support for multiple file uploads.
Store chat history in a database.
Deploy backend on cloud (e.g., AWS, GCP).
Package frontend with modern frameworks (React/Next.js).

📜 License

MIT License. Free to use and modify.
