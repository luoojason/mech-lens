# mech-lens

**Browser DevTools for the inside of a transformer.**

Type a sentence. Click Analyze. See every attention head, watch the model's prediction sharpen layer by layer, and read the final-layer logit lens position by position — no Python notebooks, no boilerplate.

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
│  FINAL-LAYER READOUT   │                                           │
│  Eiffel  ▌██     +1.42 │                                           │
│  Tower   ▌█      +0.87 │                                           │
│  in    ★ ▌████   +9.10 │  ← the only causal one                    │
└────────────────────────┴──────────────────────────────────────────┘
```

### Three panels

**Attention heatmap** — pick any layer (0–11) and head (0–11). Rows = attending *from*, columns = attending *to*. Click a token chip to highlight its full attention row in amber.

**Logit lens** — at each layer the residual stream is projected through the unembedding matrix, showing how the model's best guess evolves from noise to a confident prediction.

**Final-layer readout** — for every position, the logit its final-layer residual stream assigns to the model's top token (`unembed(ln_final(resid_post[-1]))[:, top_token]`), in logits.

This is the logit lens applied across positions, **not** an attribution. Only the last position is unembedded to make the prediction, so only the last bar is the logit the model actually used — it equals `logits[0, -1, top_token]` exactly. The other bars say what each earlier position *would* have predicted for that token; those positions cannot influence the final logit, and the values are not shares of it and do not sum to it. The panel says so in the UI. Per-token *attribution* (e.g. decomposing the final logit over heads and MLPs, or over source tokens through the OV circuits) is not implemented — see Contributing.

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

5 tests: tokenization, attention shape, row-sum invariant, logit lens depth, and the final-layer readout invariant (last position must equal the model's real logit, and must not be normalized).

## Endpoints

| Method | Path | Returns |
|--------|------|---------|
| `GET` | `/api/health` | `{ status, model_loaded }` |
| `GET` | `/api/model-info` | model config |
| `POST` | `/api/analyze` | tokens, attention `[12][12][seq][seq]`, logit lens, `final_layer_readout` (logits, not an attribution) |

Input capped at 50 tokens. Empty input → 400.

## Contributing

The model is swappable — any `HookedTransformer`-compatible checkpoint works. PRs welcome for additional analysis methods (activation patching, neuron attribution, etc.).

Wanted: **real direct logit attribution.** Decomposing `logits[0, -1, top_token]` over the residual-stream writers at the final position (embedding, each attention head's output, each MLP's output) is the standard, exactly-summing decomposition and is what the "Token contribution" panel used to claim to be. It is not implemented today.
