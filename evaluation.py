from __future__ import annotations

import numpy as np

from ca_1d import CellularAutomaton1D
from gkl import GKLRule


def evaluate_lookup_rule_accuracy(
    ca: CellularAutomaton1D,
    rule_bits: np.ndarray,
    n_samples: int = 1000,
    max_steps: int = 200,
    seed: int = 42,
) -> float:
    """
    Evaluate a 128-bit lookup-table rule on random initial states.

    Accuracy is measured only on non-tied initial states.
    """
    rng = np.random.default_rng(seed)
    correct = 0
    valid = 0

    for _ in range(n_samples):
        init_state = ca.random_state(rng)
        target = ca.majority_label(init_state)

        if target is None:
            continue

        history = ca.run_lookup_rule(
            initial_state=init_state,
            rule_bits=rule_bits,
            steps=max_steps,
            stop_on_uniform=True,
        )
        final_state = history[-1]
        pred = ca.classify_result(final_state)

        valid += 1
        if pred == target:
            correct += 1

    return correct / valid if valid > 0 else 0.0


def evaluate_gkl_accuracy(
    n_cells: int = 149,
    n_samples: int = 1000,
    max_steps: int = 200,
    seed: int = 42,
) -> float:
    """
    Evaluate the GKL benchmark rule on random initial states.
    """
    rng = np.random.default_rng(seed)
    ca = CellularAutomaton1D(n_cells=n_cells, radius=3)
    gkl = GKLRule(n_cells=n_cells)

    correct = 0
    valid = 0

    for _ in range(n_samples):
        init_state = ca.random_state(rng)
        target = ca.majority_label(init_state)

        if target is None:
            continue

        history = gkl.run(
            initial_state=init_state,
            steps=max_steps,
            stop_on_uniform=True,
        )
        final_state = history[-1]

        if np.all(final_state == 0):
            pred = 0
        elif np.all(final_state == 1):
            pred = 1
        else:
            pred = None

        valid += 1
        if pred == target:
            correct += 1

    return correct / valid if valid > 0 else 0.0