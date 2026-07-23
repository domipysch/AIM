import torch
from .base import SpotStateMapper


class ReferenceMapper(SpotStateMapper):

    name = "reference"

    def __init__(
        self,
        reference_method: str = "tangram",
    ) -> None:
        self.reference_method = reference_method

    def map(self, Z_shared: torch.Tensor, M_shared: torch.Tensor) -> torch.Tensor:
        # todo.
        return torch.tensor(1)

    def config(self) -> dict:
        return {
            "mapping": self.name,
            "reference_method": self.reference_method,
        }
