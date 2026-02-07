import torch
from x_transformers import Decoder as XTDecoder
from typing_extensions import override


class Predictor(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        attn_kv_heads: int,
        ff_dropout: float = 0.1,
        layer_dropout: float = 0.1,
        attn_dropout: float = 0.01,
    ) -> None:
        super().__init__()

        self._decoder = XTDecoder(
            dim=dim,
            depth=n_layers,
            heads=n_heads,
            attn_kv_heads=attn_kv_heads,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            layer_dropout=layer_dropout,
            ff_glu=True,
            ff_swish=True,
            cross_attend=True,  # Enables memory interaction
            use_rmsnorm=True,
        )

    @override
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Ensure z matches batch size for queries
        batch_size = x.shape[0]
        if z.ndim < x.ndim:
            z = z.unsqueeze(0)

        # Expand queries to batch size: (B, N_sensors+1, dim)
        z_batched = z.expand(batch_size, *z.shape[1:])

        # In x-transformers Decoder, cross-attention happens automatically
        # when 'context' is provided
        reconst_output = self._decoder(z_batched, context=x)
        return reconst_output
