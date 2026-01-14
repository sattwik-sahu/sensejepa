import torch
from x_transformers import ContinuousTransformerWrapper, Encoder as XTEncoder
from typing_extensions import override


class Encoder(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        n_sensors: int,
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

        self._cls_token: torch.nn.Parameter = torch.nn.Parameter(torch.randn(dim))

        # self._model: ContinuousTransformerWrapper = ContinuousTransformerWrapper(
        #     dim_in=dim,
        #     dim_out=dim,
        #     max_seq_len=n_sensors + 1,
        #     use_abs_pos_emb=False,
        #     attn_layers=XTEncoder(
        #         dim=dim,
        #         depth=n_layers,
        #         heads=n_heads,
        #         ff_swish=True,
        #         ff_glu=True,
        #         attn_kv_heads=attn_kv_heads,
        #         ff_dropout=ff_dropout,
        #         attn_dropout=attn_dropout,
        #         layer_dropout=layer_dropout,
        #         use_scalenorm=True,
        #         rotary_pos_emb=False,
        #     ),
        # )
        self._model = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=n_heads,
                dropout=ff_dropout,
                activation=torch.nn.GELU(),
                batch_first=True,
                # norm_first=True,
            ),
            num_layers=n_layers,
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = x.shape
        return self._model(
            torch.cat(
                [self._cls_token.view(1, 1, -1).expand(batch_size, 1, -1), x], dim=1
            )
        )
