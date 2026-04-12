from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_ca_history(history: List[np.ndarray], title: str = "CA Space-Time Diagram") -> None:
    """
    Plot the CA states over time as a space-time diagram.
    """
    grid = np.array(history)

    plt.figure(figsize=(10, 6))
    plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.xlabel("Cell index")
    plt.ylabel("Time step")
    plt.title(title)
    plt.colorbar(label="State")
    plt.tight_layout()
    plt.show()


def boundary_counts(history: List[np.ndarray]) -> np.ndarray:
    """
    Count the number of domain boundaries in each state.
    A boundary occurs when adjacent cells differ.
    """
    counts = []

    for state in history:
        count = np.sum(state != np.roll(state, -1))
        counts.append(int(count))

    return np.array(counts)


def plot_boundary_counts(history: List[np.ndarray], title: str = "Boundary Count Over Time") -> None:
    """
    Plot the number of boundaries over time.
    """
    counts = boundary_counts(history)

    plt.figure(figsize=(8, 4))
    plt.plot(counts)
    plt.xlabel("Time step")
    plt.ylabel("Number of boundaries")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_fitness_history(fitness_history: List[float], title: str = "Best Fitness Over Generations") -> None:
    """
    Plot GA best-fitness progression over generations.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.tight_layout()
    plt.show()