import pytest
import torch
from transformer_lens import HookedTransformer
from analysis import extract_attention, compute_logit_lens, compute_logit_attribution, analyze_text

TEST_TEXT = "The Eiffel Tower is in"


@pytest.fixture(scope="module")
def model_and_cache():
    model = HookedTransformer.from_pretrained("gpt2")
    model.eval()
    tokens = model.to_tokens(TEST_TEXT, prepend_bos=False)
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
    return model, tokens, logits, cache


def test_tokenization(model_and_cache):
    model, tokens, logits, cache = model_and_cache
    token_strs = model.to_str_tokens(tokens[0])
    assert len(token_strs) > 0
    assert isinstance(token_strs[0], str)
    # "The Eiffel Tower is in" should tokenize to at least 5 tokens
    assert len(token_strs) >= 5


def test_attention_shape(model_and_cache):
    model, tokens, logits, cache = model_and_cache
    attention = extract_attention(model, cache)
    seq_len = tokens.shape[1]
    assert len(attention) == 12, f"Expected 12 layers, got {len(attention)}"
    assert len(attention[0]) == 12, f"Expected 12 heads, got {len(attention[0])}"
    assert len(attention[0][0]) == seq_len, f"Expected {seq_len} rows"
    assert len(attention[0][0][0]) == seq_len, f"Expected {seq_len} cols"


def test_attention_row_sum(model_and_cache):
    model, tokens, logits, cache = model_and_cache
    attention = extract_attention(model, cache)
    for layer_idx, layer in enumerate(attention):
        for head_idx, head in enumerate(layer):
            for row_idx, row in enumerate(head):
                row_sum = sum(row)
                assert abs(row_sum - 1.0) < 1e-5, (
                    f"Layer {layer_idx}, head {head_idx}, row {row_idx}: sum={row_sum}"
                )


def test_logit_lens_length(model_and_cache):
    model, tokens, logits, cache = model_and_cache
    logit_lens = compute_logit_lens(model, cache)
    assert len(logit_lens) == 12, f"Expected 12 layers in logit lens, got {len(logit_lens)}"
    for layer_idx in range(12):
        key = f"layer_{layer_idx}"
        assert key in logit_lens, f"Missing key {key}"
        entries = logit_lens[key]
        assert isinstance(entries, list)
        assert len(entries) > 0
        for token, prob in entries:
            assert isinstance(token, str)
            assert 0.0 <= prob <= 1.0
