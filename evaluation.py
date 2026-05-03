from __future__ import annotations

import numpy as np

from ca_1d import CellularAutomaton1D
from gkl import GKLRule


def make_state_with_density(
    n_cells: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Create an initial condition with controlled percent of 1s.
    rho = fraction of 1s.
    """
    n_ones = int(round(rho * n_cells))
    state = np.zeros(n_cells, dtype=np.int8)
    state[:n_ones] = 1
    rng.shuffle(state)
    return state


def proportional_score(final_state: np.ndarray, target: int) -> float:
    """
    Mitchell-style proportional fitness.
    If target is 1, score is fraction of 1s in final state.
    If target is 0, score is fraction of 0s in final state.
    """
    if target == 1:
        return float(np.mean(final_state == 1))
    return float(np.mean(final_state == 0))


def strict_prediction(final_state: np.ndarray) -> int | None:
    """
    Strict density classification result.
    Only returns 0 or 1 if the CA fully converged.
    """
    if np.all(final_state == 0):
        return 0
    if np.all(final_state == 1):
        return 1
    return None


def evaluate_lookup_rule_accuracy(
    ca: CellularAutomaton1D,
    rule_bits: np.ndarray,
    n_samples: int = 300,
    max_steps: int = 320,
    seed: int = 42,
) -> float:
    """
    This is the GA fitness function.

    Instead of requiring full convergence, this gives partial credit based on
    the fraction of final cells that match the correct answer.
    """
    rng = np.random.default_rng(seed)

    # Uniform density values, balanced below and above 0.5
    low_rhos = rng.uniform(0.05, 0.49, size=n_samples // 2)
    high_rhos = rng.uniform(0.51, 0.95, size=n_samples - n_samples // 2)
    rhos = np.concatenate([low_rhos, high_rhos])
    rng.shuffle(rhos)

    scores = []

    for rho in rhos:
        init_state = make_state_with_density(ca.n_cells, rho, rng)
        target = 1 if rho > 0.5 else 0

        history = ca.run_lookup_rule(
            initial_state=init_state,
            rule_bits=rule_bits,
            steps=max_steps,
            stop_on_uniform=False,
        )

        final_state = history[-1]
        scores.append(proportional_score(final_state, target))

    return float(np.mean(scores))


def evaluate_lookup_rule_strict_convergence(
    ca: CellularAutomaton1D,
    rule_bits: np.ndarray,
    n_samples: int = 300,
    max_steps: int = 1000,
    seed: int = 42,
) -> float:
    """
    Strict performance evaluation.
    This is not used for GA training.
    It checks whether the rule fully converges to the correct fixed point.
    """
    rng = np.random.default_rng(seed)

    low_rhos = rng.uniform(0.05, 0.49, size=n_samples // 2)
    high_rhos = rng.uniform(0.51, 0.95, size=n_samples - n_samples // 2)
    rhos = np.concatenate([low_rhos, high_rhos])
    rng.shuffle(rhos)

    correct = 0

    for rho in rhos:
        init_state = make_state_with_density(ca.n_cells, rho, rng)
        target = 1 if rho > 0.5 else 0

        history = ca.run_lookup_rule(
            initial_state=init_state,
            rule_bits=rule_bits,
            steps=max_steps,
            stop_on_uniform=True,
        )

        pred = strict_prediction(history[-1])

        if pred == target:
            correct += 1

    return correct / n_samples


def evaluate_gkl_accuracy(
    n_cells: int = 149,
    n_samples: int = 300,
    max_steps: int = 320,
    seed: int = 42,
) -> float:
    """
    Proportional score for GKL benchmark.
    """
    rng = np.random.default_rng(seed)
    ca = CellularAutomaton1D(n_cells=n_cells, radius=3)
    gkl = GKLRule(n_cells=n_cells)

    low_rhos = rng.uniform(0.05, 0.49, size=n_samples // 2)
    high_rhos = rng.uniform(0.51, 0.95, size=n_samples - n_samples // 2)
    rhos = np.concatenate([low_rhos, high_rhos])
    rng.shuffle(rhos)

    scores = []

    for rho in rhos:
        init_state = make_state_with_density(n_cells, rho, rng)
        target = 1 if rho > 0.5 else 0

        history = gkl.run(
            initial_state=init_state,
            steps=max_steps,
            stop_on_uniform=False,
        )

        scores.append(proportional_score(history[-1], target))

    return float(np.mean(scores))


def evaluate_gkl_strict_accuracy(
    n_cells: int = 149,
    n_samples: int = 300,
    max_steps: int = 1000,
    seed: int = 42,
) -> float:
    """
    Strict GKL performance.
    """
    rng = np.random.default_rng(seed)
    gkl = GKLRule(n_cells=n_cells)

    low_rhos = rng.uniform(0.05, 0.49, size=n_samples // 2)
    high_rhos = rng.uniform(0.51, 0.95, size=n_samples - n_samples // 2)
    rhos = np.concatenate([low_rhos, high_rhos])
    rng.shuffle(rhos)

    correct = 0

    for rho in rhos:
        init_state = make_state_with_density(n_cells, rho, rng)
        target = 1 if rho > 0.5 else 0

        history = gkl.run(
            initial_state=init_state,
            steps=max_steps,
            stop_on_uniform=True,
        )

        pred = strict_prediction(history[-1])

        if pred == target:
            correct += 1

    return correct / n_samples


def density_sweep_lookup_rule(
    ca: CellularAutomaton1D,
    rule_bits: np.ndarray,
    rho_values: list[float],
    samples_per_rho: int = 50,
    max_steps: int = 1000,
    seed: int = 42,
) -> list[float]:
    """
    Strict accuracy as a function of density rho.
    """
    rng = np.random.default_rng(seed)
    accuracies = []

    for rho in rho_values:
        correct = 0
        target = 1 if rho > 0.5 else 0

        for _ in range(samples_per_rho):
            init_state = make_state_with_density(ca.n_cells, rho, rng)

            history = ca.run_lookup_rule(
                initial_state=init_state,
                rule_bits=rule_bits,
                steps=max_steps,
                stop_on_uniform=True,
            )

            pred = strict_prediction(history[-1])

            if pred == target:
                correct += 1

        accuracies.append(correct / samples_per_rho)

    return accuracies


def density_sweep_gkl(
    n_cells: int,
    rho_values: list[float],
    samples_per_rho: int = 50,
    max_steps: int = 1000,
    seed: int = 42,
) -> list[float]:
    """
    GKL strict accuracy as a function of density rho.
    """
    rng = np.random.default_rng(seed)
    gkl = GKLRule(n_cells=n_cells)
    accuracies = []

    for rho in rho_values:
        correct = 0
        target = 1 if rho > 0.5 else 0

        for _ in range(samples_per_rho):
            init_state = make_state_with_density(n_cells, rho, rng)

            history = gkl.run(
                initial_state=init_state,
                steps=max_steps,
                stop_on_uniform=True,
            )

            pred = strict_prediction(history[-1])

            if pred == target:
                correct += 1

        accuracies.append(correct / samples_per_rho)

    return accuracies