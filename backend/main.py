from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformer_lens import HookedTransformer
import torch

from analysis import analyze_text

model: HookedTransformer | None = None
model_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, model_loaded
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    model_loaded = True
    yield
    model = None
    model_loaded = False


app = FastAPI(title="mech-lens", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    text: str


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": model_loaded}


@app.get("/api/model-info")
def model_info():
    return {"model": "gpt2", "n_layers": 12, "n_heads": 12, "d_model": 768}


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    tokens = model.to_tokens(req.text, prepend_bos=False)
    n_tokens = tokens.shape[1]
    if n_tokens > 50:
        raise HTTPException(
            status_code=400,
            detail=f"Input too long: {n_tokens} tokens. Limit is 50.",
        )

    result = analyze_text(model, req.text)
    return result
