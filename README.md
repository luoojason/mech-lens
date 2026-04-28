# mech-lens

**Browser DevTools for the inside of a transformer.**

Type a sentence. Click Analyze. See every attention head, watch the model's prediction sharpen layer by layer, and find which tokens drive the final output — no Python notebooks, no boilerplate.

> Type `"The Eiffel Tower is in"` → layer 0 guesses `"the"` → layer 11 is 82% confident: `"Paris"`

<!-- Add a screen recording GIF here — it's the single biggest driver of stars -->

## Quickstart

```bash
# Backend (Python 3.10+) — downloads GPT-2 small (~500 MB) on first run
cd backend && pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend (new terminal)
cd frontend && npm install && npm run dev
# → http://localhost:5173
```

Everything runs locally. No API keys. No data sent anywhere.

## What you see

```
┌────────────────────────┬──────────────────────────────────────────┐
│  INPUT                 │  ATTENTION PATTERNS                      │
│  ┌──────────────────┐  │  Layer [0 ▾]  Head [0 ▾]                 │
│  │ The Eiffel Tower │  │                                           │
│  │ is in            │  │  ┌──┬──┬──┬──┬──┐  ← to                 │
│  └──────────────────┘  │  ├──┼──┼──┼──┼──┤                       │
│  [ Analyze ]           │  ├──┼██┼──┼──┼──┤  from ↓               │
│                        │  ├──┼──┼██┼──┼──┤                       │
│  The Eiffel Tower is in│  └──┴──┴──┴──┴──┘                       │
│  (click any token)     │   white=0.0 → deep-blue=1.0              │
│                        │                                           │
│  LOGIT LENS            │                                           │
│  Layer │  #1    │  #2  │                                           │
│    0   │  the   │  a   │                                           │
│    …   │  …     │  …   │                                           │
│   11★  │  Paris │France│                                           │
│                        │                                           │
│  TOKEN CONTRIBUTION    │                                           │
│  Eiffel ████████  31%  │                                           │
│  Tower  ███████   28%  │                                           │
│  is     █████     18%  │                                           │
└────────────────────────┴──────────────────────────────────────────┘
```

### Three panels

**Attention heatmap** — pick any layer (0–11) and head (0–11). Rows = attending *from*, columns = attending *to*. Click a token chip to highlight its full attention row in amber.

**Logit lens** — at each layer the residual stream is projected through the unembedding matrix, showing how the model's best guess evolves from noise to a confident prediction.

**Token contribution** — direct logit attribution: which input tokens pushed the model toward its top prediction, normalized to 100%.

## Stack

| | |
|---|---|
| Backend | Python · FastAPI · TransformerLens · GPT-2 small (124M) |
| Frontend | SvelteKit · TypeScript · D3.js |
| Transport | REST/JSON · runs fully on CPU |

## Running tests

```bash
cd backend && pip install pytest && pytest tests/
```

4 tests: tokenization, attention shape, row-sum invariant, logit lens depth.

## Endpoints

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/api/health` | `{ status, model_loaded }` |
| `GET` | `/api/model-info` | model config |
| `POST` | `/api/analyze` | tokens, attention `[12][12][seq][seq]`, logit lens, attribution |

Input capped at 50 tokens. Empty input → 400.

## Contributing

The model is swappable — any `HookedTransformer`-compatible checkpoint works. PRs welcome for additional analysis methods (activation patching, neuron attribution, etc.).
