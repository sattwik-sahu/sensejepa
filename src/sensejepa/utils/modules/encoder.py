import torch
import torch.nn as nn
from typing_extensions import override
from x_transformers import Encoder as Encoder_


class Encoder(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        n_layers: int,
        n_heads: int,
        attn_kv_heads: int,
        ff_dropout: float = 0.1,
        layer_dropout: float = 0.1,
        attn_dropout: float = 0.01,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._cls_token = nn.Parameter(torch.randn(dim))

        self._model = Encoder_(
            dim=dim,
            depth=n_layers,
            heads=n_heads,
            attn_kv_heads=attn_kv_heads,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
            layer_dropout=layer_dropout,
            ff_glu=True,
            ff_swish=True,
            use_rmsnorm=True,  # For SIGReg stability
            attn_flash=True,
        )

        self._norm = nn.LayerNorm(normalized_shape=dim)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = x.shape
        x_input = torch.cat(
            [self._cls_token.view(1, 1, -1).expand(batch_size, 1, -1), x], dim=1
        )

        encoder_output = self._model(x_input)
        return self._norm(encoder_output)
