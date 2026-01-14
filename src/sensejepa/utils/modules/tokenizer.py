import torch
from typing_extensions import override


class SensorTokenizer(torch.nn.Module):
    """
    Tokenizer for sensor reading values.
    """

    def __init__(self, n_sensors: int, dim: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._n_sensors: int = n_sensors
        self._dim: int = dim

        self._token_proj: torch.nn.Conv1d = torch.nn.Conv1d(
            in_channels=self._n_sensors,
            out_channels=self._n_sensors * self._dim,
            kernel_size=1,
            groups=self._n_sensors,
        )

        self._sensor_embeddings: torch.nn.Parameter = torch.nn.Parameter(
            torch.randn(self._n_sensors, self._dim)
        )

    @property
    def sensor_embeddings(self) -> torch.Tensor:
        return self._sensor_embeddings

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        tokens = self._token_proj(x.unsqueeze(-1)).view(
            batch_size, self._n_sensors, self._dim
        )
        se_tokens = tokens + self._sensor_embeddings.unsqueeze(0)
        return se_tokens
