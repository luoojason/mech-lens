# mech-lens

A local visual exploration tool for transformer model internals. Type text, click Analyze, and see attention patterns, layer-by-layer predictions, and token contributions — browser DevTools but for the inside of a transformer.

## UI Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  mech-lens                                                       │
│  browser DevTools for the inside of a transformer               │
├────────────────────────────┬────────────────────────────────────┤
│  INPUT                     │  ATTENTION PATTERNS                │
│  ┌──────────────────────┐  │  Layer [0 ▾]  Head [0 ▾]          │
│  │ The Eiffel Tower is  │  │                                    │
│  │ in                   │  │  ┌──┬──┬──┬──┬──┐  ← to          │
│  └──────────────────────┘  │  ├──┼──┼──┼──┼──┤                │
│  [  Analyze  ]             │  ├──┼██┼──┼──┼──┤  from          │
│                            │  ├──┼──┼██┼──┼──┤    ↓           │
│  The  Eiffel  Tower  is in │  ├──┼──┼──┼──┼──┤                │
│  (token chips — click any) │  └──┴──┴──┴──┴──┘                │
│                            │   white=0.0  deep-blue=1.0        │
│  LOGIT LENS                │                                    │
│  ┌───────┬───────┬───────┐ │                                    │
│  │ Layer │  #1   │  #2   │ │                                    │
│  │   0   │ the   │  a    │ │                                    │
│  │   …   │  …    │  …    │ │                                    │
│  │ 11 ★  │ Paris │France │ │                                    │
│  └───────┴───────┴───────┘ │                                    │
│  Top predictions: Paris 82%│                                    │
│                            │                                    │
│  TOKEN CONTRIBUTION        │                                    │
│  The    ██░░░░░░░░  12%    │                                    │
│  Eiffel ████████░░  31%    │                                    │
│  Tower  ███████░░░  28%    │                                    │
│  is     █████░░░░░  18%    │                                    │
│  in     ███░░░░░░░  11%    │                                    │
└────────────────────────────┴────────────────────────────────────┘
```

On mobile (< 768px) the panels stack vertically: Input → Attention → Logit Lens.

## Setup

### Backend (Python 3.10+)

> **First run:** GPT-2 small (~500 MB) is downloaded automatically from Hugging Face on startup. Subsequent starts are instant.

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

The API is available at `http://localhost:8000`. CPU inference takes roughly 2-5 seconds per request.

### Frontend

```bash
cd frontend
npm install
npm run dev   # http://localhost:5173
```

Both must be running at the same time.

## What each panel shows

**Attention patterns** — For a selected layer and head, shows which tokens each token "looks at." The heatmap rows are source tokens (attending *from*), columns are destination tokens (attending *to*). Cell color encodes weight: white = 0.0, deep blue = 1.0. Click a token chip to highlight its entire attention row in amber.

**Logit lens** — At each of the 12 transformer layers, the residual stream at the final token position is projected directly through the unembedding matrix (bypassing later layers). This shows how the model's "best guess" evolves layer by layer from mostly noise to a confident prediction.

**Token contribution** — Direct logit attribution: each input token's residual stream at the final layer is projected onto the unembedding direction of the top predicted token. The bar chart shows which input tokens contributed most to that prediction.

## Endpoints (backend)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | `{ status, model_loaded }` |
| GET | `/api/model-info` | Model config (GPT-2, 12 layers, 12 heads, 768 d_model) |
| POST | `/api/analyze` | Full analysis — returns tokens, attention, logit lens, attribution |

Input is capped at 50 tokens. Empty input returns HTTP 400.

## Running tests

```bash
cd backend
pip install pytest
pytest tests/
```

> Tests load GPT-2 and run a forward pass — expect ~30s on first run while the model downloads.
