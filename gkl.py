from __future__ import annotations

import numpy as np
from typing import List


class GKLRule:
    """
    Simple benchmark implementation of a GKL-style rule
    for the 1D density classification task.
    """

    def __init__(self, n_cells: int = 149) -> None:
        self.n_cells = n_cells

    def step(self, state: np.ndarray) -> np.ndarray:
        new_state = np.zeros_like(state)

        for i in range(self.n_cells):
            left1 = state[(i - 1) % self.n_cells]
            left3 = state[(i - 3) % self.n_cells]
            center = state[i]
            right1 = state[(i + 1) % self.n_cells]
            right3 = state[(i + 3) % self.n_cells]

            if center == 1:
                new_state[i] = 1 if (right1 + right3) >= 1 else 0
            else:
                new_state[i] = 1 if (left1 + left3) == 2 else 0

        return new_state

    def run(
        self,
        initial_state: np.ndarray,
        steps: int = 200,
        stop_on_uniform: bool = True,
    ) -> List[np.ndarray]:
        history = [initial_state.copy()]
        current = initial_state.copy()

        for _ in range(steps):
            current = self.step(current)
            history.append(current.copy())

            if stop_on_uniform and (np.all(current == 0) or np.all(current == 1)):
                break

        return history