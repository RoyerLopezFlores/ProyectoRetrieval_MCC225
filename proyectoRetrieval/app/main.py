from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.retrieval_service import retrieval_service


class RetrievalRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Texto de consulta")
    k: int = Field(3, ge=1, le=10, description="Cantidad de resultados")


@asynccontextmanager
async def lifespan(_: FastAPI):
    retrieval_service.load()
    yield


app = FastAPI(
    title="LiT Retrieval API",
    version="1.0.0",
    description="API de retrieval texto-imagen basada en el modelo entrenado en CuadernoLIT.",
    lifespan=lifespan,
)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=FileResponse)
async def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/retrieve")
async def retrieve(payload: RetrievalRequest) -> dict:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="El query no puede estar vacio.")
    return retrieval_service.retrieve(query, payload.k)