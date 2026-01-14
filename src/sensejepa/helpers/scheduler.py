from abc import ABC, abstractmethod

import torch
from typing_extensions import override


class Scheduler(ABC):
    def __init__(self, v0: float) -> None:
        self._v = v0

    @property
    def value(self) -> float:
        return self._v

    @abstractmethod
    def _update(self) -> None:
        pass

    def step(self) -> float:
        self._update()
        return self._v


class LinearScheduler(Scheduler):
    def __init__(self, v0: float, v_final: float, t_final: int) -> None:
        super().__init__(v0=v0)
        self._v_max = v_final
        self._t = 0
        self._t_max = t_final
        self._delta_v = (v_final - v0) / t_final

    @override
    def _update(self) -> None:
        if self._t < self._t_max:
            self._t += 1
            self._v += self._delta_v


def linear_warmup_cosine_annealing_scheduler(
    n_warmup_steps: int,
    n_total_steps: int,
    optimizer: torch.optim.Optimizer,
    eta_min: float,
) -> torch.optim.lr_scheduler.LRScheduler:
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer, total_iters=n_warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=n_total_steps - n_warmup_steps, eta_min=eta_min
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer, schedulers=[linear, cosine], milestones=[n_warmup_steps]
    )
