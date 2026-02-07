import torch
from sensejepa.utils.modules.sigreg import SIGReg
from typing_extensions import override
from dataclasses import dataclass


@dataclass
class JEPAOutput:
    s_x: torch.Tensor
    shat_y: torch.Tensor
    s_y: torch.Tensor
    loss_sigreg: torch.Tensor
    loss_jepa: torch.Tensor
    loss: torch.Tensor


class JointEmbeddingPredictiveArchitecture(torch.nn.Module):
    """
    The JEPA Module.
    """

    def __init__(
        self,
        dim: int,
        context_encoder: torch.nn.Module,
        target_encoder: torch.nn.Module,
        predictor: torch.nn.Module,
        jepa_loss_fn: torch.nn.Module,
        sigreg_lambda: float,
        sigreg_knots: int = 17,
        eta: float = 1e-3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._dim: int = dim
        self._sigreg: SIGReg = SIGReg(knots=sigreg_knots)
        self._sigreg_lambda: float = sigreg_lambda
        self._jepa_loss_fn: torch.nn.Module = jepa_loss_fn
        self._eta = eta

        self._context_encoder: torch.nn.Module = context_encoder
        self._target_encoder: torch.nn.Module = target_encoder
        self._predictor: torch.nn.Module = predictor

        self._z_cls: torch.nn.Parameter = torch.nn.Parameter(torch.randn(dim))

    @property
    def sigreg_lambda(self) -> float:
        return self._sigreg_lambda

    @sigreg_lambda.setter
    def sigreg_lambda(self, v_lambda: float) -> float:
        if 0.0 <= v_lambda <= 1.0:
            self._sigreg_lambda = v_lambda
        return self._sigreg_lambda

    @override
    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> JEPAOutput:
        # Encode the context
        s_x: torch.Tensor = self._context_encoder(x)

        # Encode the target
        with torch.no_grad():
            s_y = self._target_encoder(y).detach()

        # Extract the CLS tokens
        cls_token = s_x[:, 0]

        # Predict the target
        z_full = torch.cat([self._z_cls.unsqueeze(0), z])
        shat_y = self._predictor(s_x, z_full)

        # Calculate loss
        loss_jepa = self._jepa_loss_fn(shat_y, s_y)
        loss_sigreg = self._sigreg(cls_token)
        loss = (
            1 - self._sigreg_lambda
        ) * loss_jepa + self._eta * self._sigreg_lambda * loss_sigreg

        return JEPAOutput(
            s_x=s_x,
            s_y=s_y,
            shat_y=shat_y,
            loss_jepa=loss_jepa,
            loss_sigreg=loss_sigreg,
            loss=loss,
        )
