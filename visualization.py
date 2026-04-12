from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from typing import List


def plot_ca_history(
    history: List[np.ndarray],
    title: str = "CA Space-Time Diagram",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot CA states over time as a space-time diagram.
    """
    grid = np.array(history)

    plt.figure(figsize=(10, 6))
    plt.imshow(grid, aspect="auto", interpolation="nearest")
    plt.xlabel("Cell index")
    plt.ylabel("Time step")
    plt.title(title)
    plt.colorbar(label="State")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(block=False)
        plt.pause(0.5)

    plt.close()


def boundary_counts(history: List[np.ndarray]) -> np.ndarray:
    """
    Count the number of domain boundaries in each time step.
    A boundary occurs when adjacent cells differ.
    """
    counts = []

    for state in history:
        count = np.sum(state != np.roll(state, -1))
        counts.append(int(count))

    return np.array(counts)


def plot_boundary_counts(
    history: List[np.ndarray],
    title: str = "Boundary Count Over Time",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot number of boundaries over time.
    """
    counts = boundary_counts(history)

    plt.figure(figsize=(8, 4))
    plt.plot(counts)
    plt.xlabel("Time step")
    plt.ylabel("Number of boundaries")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(block=False)
        plt.pause(0.5)

    plt.close()


def plot_fitness_history(
    fitness_history: List[float],
    title: str = "Best Fitness Over Generations",
    save_path: str | None = None,
    show: bool = False,
) -> None:
    """
    Plot GA best fitness over generations.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(title)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show(block=False)
        plt.pause(0.5)

    plt.close()