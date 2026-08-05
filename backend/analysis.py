import torch
from transformer_lens import HookedTransformer


def extract_attention(model: HookedTransformer, cache) -> list:
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    attention = []
    for layer in range(n_layers):
        layer_attn = []
        for head in range(n_heads):
            pattern = cache["pattern", layer, "attn"][0, head]  # [seq, seq]
            layer_attn.append(pattern.tolist())
        attention.append(layer_attn)
    return attention


@torch.no_grad()
def compute_logit_lens(model: HookedTransformer, cache) -> dict:
    n_layers = model.cfg.n_layers
    logit_lens = {}
    for layer in range(n_layers):
        resid = cache["resid_post", layer][0, -1, :]  # [d_model]
        resid_2d = resid.unsqueeze(0).unsqueeze(0)  # [1, 1, d_model]
        logits_at_layer = model.unembed(model.ln_final(resid_2d))[0, 0]  # [d_vocab]
        probs = torch.softmax(logits_at_layer, dim=-1)
        top5_indices = torch.topk(probs, k=5).indices
        top5 = [[model.to_single_str_token(int(idx)), float(probs[idx])] for idx in top5_indices]
        logit_lens[f"layer_{layer}"] = top5
    return logit_lens


@torch.no_grad()
def compute_final_layer_readout(model: HookedTransformer, cache, logits: torch.Tensor) -> list:
    """Per-position logit-lens readout of the final layer against the top token.

    This is NOT an attribution. For each sequence position p it returns the logit
    that position's final-layer residual stream assigns to the model's top token:

        readout[p] = unembed(ln_final(resid_post[-1][p]))[top_token_id]

    Only the last position is ever unembedded to produce the model's actual
    prediction, so readout[-1] is exactly logits[0, -1, top_token_id]. Every other
    entry is a diagnostic "what would this position have predicted" number; those
    positions are causally incapable of contributing to the final logit, and the
    values do not decompose it. Deliberately unnormalized: the raw logits carry
    their unit and sign so nothing reads as a share-of-contribution.
    """
    top_token_id = int(torch.argmax(logits[0, -1]))
    resid_final = cache["resid_post", model.cfg.n_layers - 1]  # [1, seq, d_model]
    # ln_final normalizes each position independently, so this vectorizes over seq.
    normed = model.ln_final(resid_final)  # [1, seq, d_model]
    readout = model.unembed(normed)[0, :, top_token_id]  # [seq]
    return readout.tolist()


def analyze_text(model: HookedTransformer, text: str) -> dict:
    tokens = model.to_tokens(text, prepend_bos=False)
    token_strs = model.to_str_tokens(tokens[0])
    token_ids = tokens[0].tolist()

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        attention = extract_attention(model, cache)
        logit_lens = compute_logit_lens(model, cache)
        final_layer_readout = compute_final_layer_readout(model, cache, logits)

    final_logits = logits[0, -1]
    probs = torch.softmax(final_logits, dim=-1)
    top5_indices = torch.topk(probs, k=5).indices
    top_predictions = [[model.to_single_str_token(int(idx)), float(probs[idx])] for idx in top5_indices]

    return {
        "tokens": token_strs,
        "token_ids": token_ids,
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "attention": attention,
        "logit_lens": logit_lens,
        "top_predictions": top_predictions,
        # Per-position logit-lens readout, NOT an attribution. See
        # compute_final_layer_readout. Only the last entry is the model's real logit.
        "final_layer_readout": final_layer_readout,
        "final_layer_readout_unit": "logits",
    }
