from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ca_1d import CellularAutomaton1D
from gkl import GKLRule
from ga import GeneticAlgorithmCA, GAConfig
from evaluation import evaluate_gkl_accuracy
from visualization import plot_ca_history, plot_boundary_counts, plot_fitness_history
from networks import make_erdos_renyi, make_watts_strogatz, make_barabasi_albert
from graph_ca import GraphThresholdCA


def demo_gkl() -> None:
    ca = CellularAutomaton1D(n_cells=149, radius=3)
    gkl = GKLRule(n_cells=149)

    init_state = ca.random_state(np.random.default_rng(0))
    history = gkl.run(init_state, steps=100)

    print("GKL demo final classification:", ca.classify_result(history[-1]), flush=True)

    plot_ca_history(
        history,
        title="GKL Benchmark: Space-Time Diagram",
        save_path="Figure_1_GKL_SpaceTime.png",
        show=False,
    )

    plot_boundary_counts(
        history,
        title="GKL Benchmark: Boundary Count Over Time",
        save_path="Figure_2_GKL_Boundaries.png",
        show=False,
    )

    acc = evaluate_gkl_accuracy(n_cells=149, n_samples=100, max_steps=100, seed=0)
    print(f"GKL benchmark accuracy (sampled): {acc:.4f}", flush=True)


def demo_ga() -> np.ndarray:
    ca = CellularAutomaton1D(n_cells=149, radius=3)

    # Smaller settings for testing first
    config = GAConfig(
        population_size=20,
        generations=5,
        mutation_rate=0.01,
        crossover_rate=0.9,
        tournament_size=5,
        eval_samples=50,
        max_steps=100,
        seed=1,
    )

    ga = GeneticAlgorithmCA(ca, config)
    best_rule, best_fitness, fitness_history = ga.evolve()

    print(f"Best evolved rule fitness: {best_fitness:.4f}", flush=True)

    plot_fitness_history(
        fitness_history,
        title="GA Fitness Over Generations",
        save_path="Figure_3_GA_Fitness.png",
        show=False,
    )

    init_state = ca.random_state(np.random.default_rng(5))
    history = ca.run_lookup_rule(init_state, best_rule, steps=100)

    plot_ca_history(
        history,
        title="Best Evolved Rule: Space-Time Diagram",
        save_path="Figure_4_Evolved_SpaceTime.png",
        show=False,
    )

    plot_boundary_counts(
        history,
        title="Best Evolved Rule: Boundary Count Over Time",
        save_path="Figure_5_Evolved_Boundaries.png",
        show=False,
    )

    return best_rule


def demo_networks() -> None:
    er = make_erdos_renyi()
    ws = make_watts_strogatz()
    ba = make_barabasi_albert()

    print("Erdos-Renyi:", er.number_of_nodes(), er.number_of_edges(), flush=True)
    print("Watts-Strogatz:", ws.number_of_nodes(), ws.number_of_edges(), flush=True)
    print("Barabasi-Albert:", ba.number_of_nodes(), ba.number_of_edges(), flush=True)

    gca = GraphThresholdCA(ws, threshold=0.5, include_self=True, seed=7)
    init_state = gca.random_state()
    history = gca.run(init_state, steps=20, synchronous=True)

    final_active = sum(history[-1].values())
    print("Final active nodes on WS graph:", final_active, flush=True)

    pos = nx.spring_layout(ws, seed=0)
    node_colors = [history[-1][node] for node in ws.nodes()]

    plt.figure(figsize=(6, 6))
    nx.draw(ws, pos=pos, node_color=node_colors, with_labels=False, node_size=80)
    plt.title("Early Network Experiment: Watts-Strogatz Threshold CA")
    plt.tight_layout()
    plt.savefig("Figure_6_WS_Graph.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    print("=== GKL benchmark ===", flush=True)
    demo_gkl()

    print("\n=== GA evolution ===", flush=True)
    best_rule = demo_ga()

    print("\n=== Early network experiments ===", flush=True)
    demo_networks()

    print("\nAll figures saved successfully.", flush=True)