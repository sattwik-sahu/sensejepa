import torch
from torch.utils.data import Dataset
import cudf
from typing import TypedDict


class GestrSample(TypedDict):
    label: str
    label_ohe: torch.Tensor
    values: torch.Tensor


class GestrDataset(Dataset[GestrSample]):
    """
    Dataset for Gestr glove using CuPy for GPU-accelerated processing.
    """

    def __init__(self, file_path: str) -> None:
        super().__init__()

        # 1. Load data with cudf (GPU DataFrame)
        df = cudf.read_csv(file_path)

        # 2. Extract labels and values as CuPy arrays
        # We use .values to get a cupy-backed array directly
        self._labels_series = df["label"]
        raw_values = df.iloc[:, 1:].values  # This is already a CuPy array

        self._mu: torch.Tensor = torch.as_tensor(raw_values.mean(axis=0))
        self._sigma: torch.Tensor = torch.as_tensor(raw_values.std(axis=0))

        # Convert values to torch directly (zero-copy)
        self._values = torch.as_tensor(raw_values, device="cuda")

        # 3. Create One-Hot Encoding using CuPy/Torch instead of sklearn
        # This keeps the logic on the GPU
        unique_labels = self._labels_series.unique().sort_values()
        label_map = {
            label: i for i, label in enumerate(unique_labels.to_arrow().to_pylist())
        }

        # Map string labels to integers on GPU
        label_indices = self._labels_series.map(label_map).values
        num_classes = len(unique_labels)

        # Create OHE matrix on GPU
        self._labels_ohe = torch.nn.functional.one_hot(
            torch.as_tensor(label_indices, device="cuda").long(),
            num_classes=num_classes,
        )

    @property
    def mu(self) -> torch.Tensor:
        return self._mu

    @property
    def sigma(self) -> torch.Tensor:
        return self._sigma

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> GestrSample:
        # Everything here is already a torch.Tensor on the GPU
        return GestrSample(
            label=str(self._labels_series.iloc[index]),
            label_ohe=self._labels_ohe[index],
            values=self._values[index],
        )
