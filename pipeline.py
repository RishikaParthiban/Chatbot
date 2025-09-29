import os
import json
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# ---------------- Embedding & Vector DB ----------------
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHROMA_CLIENT = chromadb.PersistentClient(path="xray_db")
COLLECTION = CHROMA_CLIENT.get_or_create_collection(name="xray_chunks")

# ---------------- Gemini LLM Setup ----------------
GEN_API_KEY = "AIzaSyAPteCJMtCZBCP4QJbfmfksFk3yEoG1Dt0"
genai.configure(api_key=GEN_API_KEY)
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# ---------------- PDF Parsing ----------------
def parse_pdf(file_path: str, json_out="parsed_pdf.json"):
    reader = PdfReader(file_path)
    chunks = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        chunk = {
            "chunkId": f"page_{i+1}",
            "pageNumbers": [i + 1],
            "text": text.strip(),
            "sectionSummary": ""
        }
        chunks.append(chunk)

    data = {"chunks": chunks}
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return json_out

# ---------------- Store in Vector DB ----------------
def store_in_vector_db(json_path="parsed_pdf.json"):
    """Embed chunks and store in ChromaDB after deleting old data"""
    # Delete all existing documents
    try:
        COLLECTION.delete(where={"page": {"$gte": 0}})
    except Exception as e:
        print(f"Warning: could not delete previous data: {e}")

    # Load new PDF chunks
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, metadatas, ids = [], [], []
    for i, chunk in enumerate(data.get("chunks", [])):
        text = chunk.get("text", "")
        if not text.strip():
            continue
        ids.append(chunk.get("chunkId", f"chunk_{i}"))
        metadatas.append({
            "summary": chunk.get("sectionSummary", ""),
            "page": chunk.get("pageNumbers", [None])[0]
        })
        texts.append(text)

    if not texts:
        return 0

    embeddings = EMBED_MODEL.encode(texts, show_progress_bar=False).tolist()
    COLLECTION.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    return len(texts)


# ---------------- Retrieve Relevant Chunks ----------------
def retrieve(query: str, n_results=5):
    embedding = EMBED_MODEL.encode(query).tolist()
    results = COLLECTION.query(query_embeddings=[embedding], n_results=n_results)
    return results

# ---------------- Generate Answer using Gemini ----------------
def generate_answer(query: str, results):
    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return "No relevant context found."

    context_texts = documents[0]
    context = "\n\n".join(context_texts)

    prompt = f"""
You are an AI assistant. Use the following context to answer the question.

Context:
{context}

Question:
{query}

Answer clearly and concisely.
"""

    try:
        response = GEMINI_MODEL.generate_content(prompt)
        answer_text = response.text if hasattr(response, "text") else str(response)
    except Exception as e:
        answer_text = f"❌ LLM Error: {str(e)}"

    return answer_text
