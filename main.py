from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from ca_1d import CellularAutomaton1D
from gkl import GKLRule
from ga import GeneticAlgorithmCA, GAConfig
from evaluation import (
    make_state_with_density,
    evaluate_gkl_accuracy,
    evaluate_gkl_strict_accuracy,
    evaluate_lookup_rule_strict_convergence,
    density_sweep_gkl,
    density_sweep_lookup_rule,
)
from visualization import (
    plot_ca_history,
    plot_boundary_counts,
    plot_fitness_history,
)
from networks import make_erdos_renyi, make_watts_strogatz, make_barabasi_albert
from graph_ca import GraphThresholdCA
from network_ga import (
    NetworkGAConfig,
    NetworkGeneticAlgorithm,
    evaluate_network_rule_strict,
)


def demo_gkl() -> None:
    ca = CellularAutomaton1D(n_cells=149, radius=3)
    gkl = GKLRule(n_cells=149)

    rng = np.random.default_rng(0)
    init_state = make_state_with_density(n_cells=149, rho=0.35, rng=rng)
    history = gkl.run(init_state, steps=149)

    print("=== GKL benchmark ===", flush=True)

    prop_acc = evaluate_gkl_accuracy(
        n_cells=149,
        n_samples=300,
        max_steps=320,
        seed=0,
    )

    strict_acc = evaluate_gkl_strict_accuracy(
        n_cells=149,
        n_samples=300,
        max_steps=1000,
        seed=1,
    )

    print(f"GKL proportional fitness: {prop_acc:.4f}", flush=True)
    print(f"GKL strict accuracy: {strict_acc:.4f}", flush=True)

    plot_ca_history(
        history,
        title="GKL Benchmark: Space-Time Diagram (rho = 0.35)",
        save_path="Figure_1_GKL_SpaceTime.png",
        show=False,
    )

    plot_boundary_counts(
        history,
        title="GKL Benchmark: Boundary Count Over Time",
        save_path="Figure_2_GKL_Boundaries.png",
        show=False,
    )


def demo_ga() -> np.ndarray:
    print("\n=== GA evolution on 1D ring ===", flush=True)

    ca = CellularAutomaton1D(n_cells=149, radius=3)

    config = GAConfig(
        population_size=30,
        generations=10,
        mutation_rate=0.03,
        crossover_rate=0.9,
        tournament_size=5,
        eval_samples=60,
        max_steps=149,
        seed=1,
    )

    ga = GeneticAlgorithmCA(ca, config)
    best_rule, best_fitness, fitness_history = ga.evolve()

    print(f"Best evolved rule proportional fitness: {best_fitness:.4f}", flush=True)

    strict_acc = evaluate_lookup_rule_strict_convergence(
        ca=ca,
        rule_bits=best_rule,
        n_samples=100,
        max_steps=500,
        seed=4,
    )

    print(f"Best evolved rule strict accuracy: {strict_acc:.4f}", flush=True)

    plot_fitness_history(
        fitness_history,
        title="GA Fitness Over Generations on 1D Ring",
        save_path="Figure_3_GA_Fitness.png",
        show=False,
    )

    rng = np.random.default_rng(5)
    init_state = make_state_with_density(n_cells=149, rho=0.35, rng=rng)

    history = ca.run_lookup_rule(
        initial_state=init_state,
        rule_bits=best_rule,
        steps=149,
        stop_on_uniform=False,
    )

    plot_ca_history(
        history,
        title="Best Evolved Ring Rule: Space-Time Diagram (rho = 0.35)",
        save_path="Figure_4_Evolved_SpaceTime.png",
        show=False,
    )

    plot_boundary_counts(
        history,
        title="Best Evolved Ring Rule: Boundary Count Over Time",
        save_path="Figure_5_Evolved_Boundaries.png",
        show=False,
    )

    return best_rule


def density_sweep_experiment(best_rule: np.ndarray) -> None:
    print("\n=== Density sweep experiment ===", flush=True)

    ca = CellularAutomaton1D(n_cells=149, radius=3)

    rho_values = [0.1, 0.2, 0.3, 0.4, 0.45, 0.55, 0.6, 0.7, 0.8, 0.9]

    gkl_acc = density_sweep_gkl(
        n_cells=149,
        rho_values=rho_values,
        samples_per_rho=20,
        max_steps=500,
        seed=10,
    )

    evolved_acc = density_sweep_lookup_rule(
        ca=ca,
        rule_bits=best_rule,
        rho_values=rho_values,
        samples_per_rho=20,
        max_steps=500,
        seed=11,
    )

    print("rho values:", rho_values, flush=True)
    print("GKL density accuracy:", gkl_acc, flush=True)
    print("Evolved ring-rule density accuracy:", evolved_acc, flush=True)

    plt.figure(figsize=(8, 5))
    plt.plot(rho_values, gkl_acc, marker="o", label="GKL Rule")
    plt.plot(rho_values, evolved_acc, marker="s", label="Best Evolved Ring Rule")
    plt.axvline(0.5, linestyle="--", label="rho = 0.5")
    plt.xlabel("Initial density of 1s, rho(0)")
    plt.ylabel("Strict classification accuracy")
    plt.title("Strict Classification Accuracy vs Initial Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure_6_Density_Sweep.png", dpi=200, bbox_inches="tight")
    plt.close()


def graph_state_with_density(
    graph: nx.Graph,
    rho: float,
    rng: np.random.Generator,
) -> dict[int, int]:
    nodes = list(graph.nodes())
    n_ones = int(round(rho * len(nodes)))

    values = np.zeros(len(nodes), dtype=np.int8)
    values[:n_ones] = 1
    rng.shuffle(values)

    return {node: int(values[i]) for i, node in enumerate(nodes)}


def run_graph_ca_once(
    graph: nx.Graph,
    rho: float,
    threshold: float,
    seed: int,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed)

    gca = GraphThresholdCA(
        graph=graph,
        threshold=threshold,
        include_self=True,
        seed=seed,
    )

    init_state = graph_state_with_density(graph, rho, rng)
    history = gca.run(init_state, steps=50, synchronous=True)

    final_active = sum(history[-1].values())
    final_inactive = graph.number_of_nodes() - final_active

    return final_active, final_inactive


def demo_networks() -> None:
    print("\n=== Threshold network experiments ===", flush=True)

    er = make_erdos_renyi(n=149, avg_degree=6, seed=42)
    ws = make_watts_strogatz(n=149, k=6, beta=0.1, seed=42)
    ba = make_barabasi_albert(n=149, m=3, seed=42)

    graphs = {
        "Erdos-Renyi": er,
        "Watts-Strogatz": ws,
        "Barabasi-Albert": ba,
    }

    threshold = 0.5
    rho = 0.35

    names = []
    active_counts = []

    for idx, (name, graph) in enumerate(graphs.items()):
        active, inactive = run_graph_ca_once(
            graph=graph,
            rho=rho,
            threshold=threshold,
            seed=100 + idx,
        )

        print(
            f"{name}: nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}, "
            f"final active={active}, final inactive={inactive}",
            flush=True,
        )

        names.append(name)
        active_counts.append(active)

    plt.figure(figsize=(8, 5))
    plt.bar(names, active_counts)
    plt.ylabel("Final active nodes")
    plt.title("Threshold Network CA Final Active Nodes (rho = 0.35, threshold = 0.5)")
    plt.tight_layout()
    plt.savefig("Figure_7_Network_Comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    rng = np.random.default_rng(7)
    gca = GraphThresholdCA(ws, threshold=0.5, include_self=True, seed=7)
    init_state = graph_state_with_density(ws, rho=0.35, rng=rng)
    history = gca.run(init_state, steps=50, synchronous=True)

    pos = nx.spring_layout(ws, seed=0)
    node_colors = [history[-1][node] for node in ws.nodes()]

    plt.figure(figsize=(7, 7))
    nx.draw(ws, pos=pos, node_color=node_colors, with_labels=False, node_size=80)
    plt.title("Watts-Strogatz Threshold Network CA Final State")
    plt.savefig("Figure_8_WS_Graph_Final_State.png", dpi=200, bbox_inches="tight")
    plt.close()


def evolve_rules_directly_on_networks() -> None:
    print("\n=== Direct network rule evolution ===", flush=True)

    graphs = {
        "Erdos-Renyi": make_erdos_renyi(n=149, avg_degree=6, seed=42),
        "Watts-Strogatz": make_watts_strogatz(n=149, k=6, beta=0.1, seed=42),
        "Barabasi-Albert": make_barabasi_albert(n=149, m=3, seed=42),
    }

    sync_results = []
    async_results = []
    sync_strict_results = []
    async_strict_results = []
    names = []

    for graph_name, graph in graphs.items():
        print(f"\n--- Evolving directly on {graph_name} network ---", flush=True)

        sync_config = NetworkGAConfig(
            population_size=30,
            generations=10,
            mutation_rate=0.05,
            crossover_rate=0.9,
            tournament_size=5,
            eval_samples=60,
            steps=50,
            seed=10,
            synchronous=True,
        )

        sync_ga = NetworkGeneticAlgorithm(graph=graph, config=sync_config)
        sync_rule, sync_fit, sync_history = sync_ga.evolve()

        sync_strict = evaluate_network_rule_strict(
            graph=graph,
            rule_bits=sync_rule,
            n_samples=60,
            steps=100,
            seed=100,
            synchronous=True,
        )

        print(
            f"{graph_name} synchronous best fitness={sync_fit:.4f}, "
            f"strict accuracy={sync_strict:.4f}, rule={sync_rule.tolist()}",
            flush=True,
        )

        async_config = NetworkGAConfig(
            population_size=30,
            generations=10,
            mutation_rate=0.05,
            crossover_rate=0.9,
            tournament_size=5,
            eval_samples=60,
            steps=50,
            seed=20,
            synchronous=False,
        )

        async_ga = NetworkGeneticAlgorithm(graph=graph, config=async_config)
        async_rule, async_fit, async_history = async_ga.evolve()

        async_strict = evaluate_network_rule_strict(
            graph=graph,
            rule_bits=async_rule,
            n_samples=60,
            steps=100,
            seed=200,
            synchronous=False,
        )

        print(
            f"{graph_name} asynchronous best fitness={async_fit:.4f}, "
            f"strict accuracy={async_strict:.4f}, rule={async_rule.tolist()}",
            flush=True,
        )

        names.append(graph_name)
        sync_results.append(sync_fit)
        async_results.append(async_fit)
        sync_strict_results.append(sync_strict)
        async_strict_results.append(async_strict)

        plt.figure(figsize=(8, 5))
        plt.plot(sync_history, marker="o", label="Synchronous")
        plt.plot(async_history, marker="s", label="Asynchronous")
        plt.xlabel("Generation")
        plt.ylabel("Best proportional fitness")
        plt.title(f"Direct Network GA Fitness: {graph_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"Figure_Network_GA_{graph_name.replace('-', '_')}.png", dpi=200)
        plt.close()

    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, sync_results, width, label="Synchronous")
    plt.bar(x + width / 2, async_results, width, label="Asynchronous")
    plt.xticks(x, names)
    plt.ylabel("Best proportional fitness")
    plt.title("Direct Network Rule Evolution: Sync vs Async")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure_9_Direct_Network_Evolution_Comparison.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, sync_strict_results, width, label="Synchronous")
    plt.bar(x + width / 2, async_strict_results, width, label="Asynchronous")
    plt.xticks(x, names)
    plt.ylabel("Strict classification accuracy")
    plt.title("Direct Network Rule Evolution: Strict Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Figure_10_Direct_Network_Strict_Accuracy.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    demo_gkl()
    best_rule = demo_ga()
    density_sweep_experiment(best_rule)
    demo_networks()
    evolve_rules_directly_on_networks()

    print("\nAll experiments completed. Figures saved.", flush=True)