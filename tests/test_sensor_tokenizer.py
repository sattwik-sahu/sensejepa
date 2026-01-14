import torch
import pytest
from sensejepa.utils.modules.tokenizer import SensorTokenizer as SensorTokenizer


class TestSensorTokenizer:
    @pytest.fixture
    def params(self):
        return {"n_sensors": 10, "dim": 512, "batch_size": 10}

    def test_output_shape(self, params):
        """Verifies that the output matches (Batch, N, Dim)."""
        tokenizer = SensorTokenizer(n_sensors=params["n_sensors"], dim=params["dim"])
        x = torch.randn(params["batch_size"], params["n_sensors"])

        output = tokenizer(x)

        expected_shape = (params["batch_size"], params["n_sensors"], params["dim"])
        assert output.shape == expected_shape, (
            f"Expected {expected_shape}, got {output.shape}"
        )

    def test_gradient_isolation(self, params):
        """
        Verifies that weights for Sensor A are not affected by gradients
        calculated from Sensor B's output.
        """
        n = params["n_sensors"]
        d = params["dim"]
        tokenizer = SensorTokenizer(n_sensors=n, dim=d)

        # Enable gradient tracking on input for extra verification
        x = torch.randn(params["batch_size"], n, requires_grad=True)

        # Forward pass
        output = tokenizer(x)

        # 1. Calculate loss ONLY on the FIRST sensor's output (index 0)
        loss = output[:, 0, :].sum()
        loss.backward()

        # 2. Check gradients of the projection layer
        # Weights in Conv1d are stored as (out_channels, in_channels/groups, kernel_size)
        # For groups=10, weights 0-511 belong to sensor 0, 512-1023 to sensor 1, etc.
        weights_grad = tokenizer._token_proj.weight.grad

        # Gradients for the first sensor's weights should be non-zero
        first_sensor_grad_chunk = weights_grad[0:d, :, :]
        assert torch.any(first_sensor_grad_chunk != 0), (
            "Sensor 0 weights should have gradients."
        )

        # Gradients for all other sensors should be EXACTLY zero
        other_sensors_grad_chunk = weights_grad[d:, :, :]
        assert torch.all(other_sensors_grad_chunk == 0), (
            "Other sensor weights leaked information!"
        )

        # 3. Check gradients of the sensor embeddings
        assert torch.any(tokenizer._sensor_embeddings.grad[0] != 0), (
            "Sensor 0 embedding should have grad."
        )
        assert torch.all(tokenizer._sensor_embeddings.grad[1:] == 0), (
            "Other sensor embeddings leaked info!"
        )

    def test_learnable_embeddings_added(self, params):
        """Verifies that sensor embeddings are actually added to the projection."""
        tokenizer = SensorTokenizer(n_sensors=params["n_sensors"], dim=params["dim"])

        # Zero out weights so output is only the bias + embedding
        torch.nn.init.zeros_(tokenizer._token_proj.weight)
        torch.nn.init.zeros_(tokenizer._token_proj.bias)

        x = torch.zeros(params["batch_size"], params["n_sensors"])
        output = tokenizer(x)

        # Output should equal the learnable sensor embeddings (broadcasted)
        for i in range(params["n_sensors"]):
            # Check if output for sensor i equals embedding i
            assert torch.allclose(output[0, i, :], tokenizer._sensor_embeddings[i]), (
                f"Embedding {i} not correctly added."
            )
