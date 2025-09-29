# main.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import shutil, os
from RAG.pipeline import parse_pdf, store_in_vector_db, retrieve, generate_answer

app = FastAPI()

# Serve static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------- Serve HTML directly ----------------
@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("index.html")  # HTML is in project root, not templates

# ---------------- Upload PDF ----------------
@app.post("/upload")
async def upload_pdf(file: UploadFile):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # Parse PDF and store embeddings
        json_path = parse_pdf(file_path)
        count = store_in_vector_db(json_path)
    except Exception as e:
        return {"message": f"❌ Error: {str(e)}"}

    return {"message": f"✅ File uploaded and parsed successfully ({count} chunks stored)."}

# ---------------- Ask Question ----------------
@app.post("/ask")
async def ask_question(query: str = Form(...)):
    try:
        results = retrieve(query)
        answer = generate_answer(query, results)
    except Exception as e:
        return {"answer": f"❌ Error: {str(e)}"}
    return {"answer": answer}
