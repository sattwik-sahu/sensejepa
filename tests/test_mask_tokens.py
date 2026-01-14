import torch
from sensejepa.utils.masking import mask_tokens as mask_tokens


def test_mask_tokens_logic():
    batch, n, d = 4, 10, 32
    p = 0.3  # Should keep 7, mask 3
    tokens = torch.randn(batch, n, d)

    masked_out, mask = mask_tokens(tokens, p)

    # Check shapes
    assert masked_out.shape == (batch, 7, d)
    assert mask.shape == (batch, n)

    # Check that mask has correct number of 1s (masked)
    assert torch.all(mask.sum(dim=1) == 3)
