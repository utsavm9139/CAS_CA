from __future__ import annotations

import numpy as np
from typing import List, Tuple


class CellularAutomaton1D:
    """
    1D binary cellular automaton on a periodic ring.
    Supports:
      - generic radius-3 lookup-table rules (128 bits)
      - repeated simulation
      - density-classification checking
    """

    def __init__(self, n_cells: int = 149, radius: int = 3) -> None:
        if n_cells <= 0:
            raise ValueError("n_cells must be positive.")
        if radius != 3:
            raise ValueError("This project currently assumes radius=3 for 128-bit rules.")
        self.n_cells = n_cells
        self.radius = radius
        self.rule_size = 2 ** (2 * radius + 1)  # 2^7 = 128

    def random_state(self, rng: np.random.Generator | None = None) -> np.ndarray:
        """Generate a random binary state."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.integers(0, 2, size=self.n_cells, dtype=np.int8)

    def majority_label(self, state: np.ndarray) -> int | None:
        """
        Returns:
            1 if density(state) > 0.5
            0 if density(state) < 0.5
            None if exactly tied
        """
        ones = int(np.sum(state))
        zeros = len(state) - ones
        if ones > zeros:
            return 1
        if zeros > ones:
            return 0
        return None

    def is_uniform(self, state: np.ndarray) -> bool:
        """Check whether state is all 0s or all 1s."""
        return bool(np.all(state == state[0]))

    def classify_result(self, final_state: np.ndarray) -> int | None:
        """
        Returns:
            0 if final state is all 0s
            1 if final state is all 1s
            None otherwise
        """
        if np.all(final_state == 0):
            return 0
        if np.all(final_state == 1):
            return 1
        return None

    def neighborhood_to_index(self, neighborhood: np.ndarray) -> int:
        """
        Convert a 7-bit neighborhood to an integer in [0, 127].
        Example neighborhood [1,0,1,1,0,0,1] -> integer index.
        """
        idx = 0
        for bit in neighborhood:
            idx = (idx << 1) | int(bit)
        return idx

    def step_lookup_rule(self, state: np.ndarray, rule_bits: np.ndarray) -> np.ndarray:
        """
        Apply one synchronous update using a 128-bit lookup rule.
        rule_bits[index] gives the output bit for the 7-bit neighborhood encoded by index.
        """
        if len(rule_bits) != self.rule_size:
            raise ValueError(f"rule_bits must have length {self.rule_size}.")
        new_state = np.zeros_like(state)

        for i in range(self.n_cells):
            neighborhood = np.array(
                [state[(i + offset) % self.n_cells] for offset in range(-3, 4)],
                dtype=np.int8
            )
            idx = self.neighborhood_to_index(neighborhood)
            new_state[i] = rule_bits[idx]

        return new_state

    def run_lookup_rule(
        self,
        initial_state: np.ndarray,
        rule_bits: np.ndarray,
        steps: int = 200,
        stop_on_uniform: bool = True,
    ) -> List[np.ndarray]:
        """Run the CA under a lookup rule and return all states."""
        history = [initial_state.copy()]
        current = initial_state.copy()

        for _ in range(steps):
            current = self.step_lookup_rule(current, rule_bits)
            history.append(current.copy())

            if stop_on_uniform and self.is_uniform(current):
                break

        return history