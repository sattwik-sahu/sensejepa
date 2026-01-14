import torch
from x_transformers import ContinuousTransformerWrapper, Decoder
from typing_extensions import override


class Predictor(torch.nn.Module):
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
    ) -> None:
        super().__init__()

        # The transformer decoder
        # self._decoder: ContinuousTransformerWrapper = ContinuousTransformerWrapper(
        #     dim_in=dim,
        #     dim_out=dim,
        #     max_seq_len=n_sensors + 1,
        #     use_abs_pos_emb=False,
        #     attn_layers=Decoder(
        #         dim=dim,
        #         depth=n_layers,
        #         heads=n_heads,
        #         ff_swish=True,
        #         ff_glu=True,
        #         attn_kv_heads=attn_kv_heads,
        #         use_scalenorm=True,
        #         ff_dropout=ff_dropout,
        #         attn_dropout=attn_dropout,
        #         layer_dropout=layer_dropout,
        #         cross_attend=True,
        #         rotary_pos_emb=False,
        #     ),
        # )
        self._decoder = torch.nn.TransformerDecoder(
            decoder_layer=torch.nn.TransformerDecoderLayer(
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
    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = x.shape
        if z.ndim < x.ndim:
            z = z.unsqueeze(0)
        z_batched = z.expand(batch_size, *z.shape[1:])

        reconst_output: torch.Tensor = self._decoder(tgt=z_batched, memory=x)
        return reconst_output
