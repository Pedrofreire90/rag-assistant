from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import rag
from .ingest import load_chunks

app = FastAPI(title="Assistente RAG")


class Question(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(5, ge=1, le=15)
    rerank: bool | None = None  # None = usa o padrão do .env (RERANK)


@app.post("/documents")
def upload(files: list[UploadFile] = File(...)):
    results = []
    for f in files:
        try:
            chunks = load_chunks(f.filename, f.file.read())
            if not chunks:
                raise ValueError("Nenhum texto extraído (PDF escaneado?)")
            results.append({"source": f.filename, "chunks": rag.add_chunks(f.filename, chunks)})
        except Exception as e:
            results.append({"source": f.filename, "error": str(e)})
    return results


@app.get("/documents")
def documents():
    return rag.list_sources()


@app.delete("/documents/{name}")
def remove(name: str):
    rag.delete_source(name)
    return {"deleted": name}


@app.post("/ask")
def ask(q: Question):
    try:
        return rag.ask(q.question, q.k, q.rerank)
    except Exception as e:
        raise HTTPException(500, str(e))


app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def index():
    return FileResponse("app/static/index.html")
