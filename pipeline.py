# rag_pipeline/pipeline.py
import json, time, urllib.request
import chromadb
from sentence_transformers import SentenceTransformer
from groundx import GroundX, Document
import google.generativeai as genai
import os

# --- Global setup ---
EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
CHROMA_CLIENT = chromadb.PersistentClient(path="xray_db")
COLLECTION = CHROMA_CLIENT.get_or_create_collection(name="xray_chunks")

# Configure Gemini client
genai.configure(api_key=os.getenv("GEMINI_API_KEY", "GEMINI_API_KEY"))
GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# GroundX client
GROUNDX_CLIENT = GroundX(api_key=os.getenv("GROUNDX_API_KEY", "16ec4ba5-f166-40d6-9d12-40f144206d35"))


def parse_and_ingest(file_path: str, bucket_name="parsed_documents_bucket"):
    """Parse a PDF with GroundX and save parsed_xray.json"""
    bucket_resp = GROUNDX_CLIENT.buckets.create(name=bucket_name)
    bucket_id = bucket_resp.bucket.bucket_id

    ingest_resp = GROUNDX_CLIENT.ingest(
        documents=[Document(
            bucket_id=bucket_id,
            file_name=os.path.basename(file_path),
            file_path=file_path,
            file_type="pdf"
        )]
    )
    process_id = ingest_resp.ingest.process_id

    # Poll until complete
    while True:
        status_resp = GROUNDX_CLIENT.documents.get_processing_status_by_id(process_id=process_id)
        status = status_resp.ingest.status.lower()
        if status in ("complete", "cancelled", "error"):
            break
        time.sleep(3)

    if status == "error":
        raise RuntimeError("❌ Error ingesting document")

    # Lookup parsed doc
    doc_list = GROUNDX_CLIENT.documents.lookup(id=bucket_id)
    if not doc_list.documents:
        raise RuntimeError("❌ No documents returned by GroundX")

    xray_url = doc_list.documents[0].xray_url
    with urllib.request.urlopen(xray_url) as url:
        data = json.loads(url.read().decode())

    with open("parsed_xray.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return "parsed_xray.json"


def store_in_vector_db(json_path="parsed_xray.json"):
    """Chunk parsed JSON, embed, and store in ChromaDB"""
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


def retrieve(query: str, n_results=5):
    """Search vector DB for relevant chunks"""
    embedding = EMBED_MODEL.encode(query).tolist()
    results = COLLECTION.query(query_embeddings=[embedding], n_results=n_results)
    return results


def generate_answer(query: str, results):
    """Generate final answer with Gemini"""
    documents = results.get("documents", [[]])
    if not documents or not documents[0]:
        return "No relevant context found in documents."

    context_texts = documents[0]
    context = "\n\n".join(context_texts)

    prompt = f"""
    You are an AI assistant. Use the following context to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer in a clear, concise way.
    """

    response = GEMINI_MODEL.generate_content(prompt)
    return response.text
