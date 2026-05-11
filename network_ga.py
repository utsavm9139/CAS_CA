from __future__ import annotations

import numpy as np
import networkx as nx
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class NetworkGAConfig:
    population_size: int = 30
    generations: int = 10
    mutation_rate: float = 0.05
    crossover_rate: float = 0.9
    tournament_size: int = 5
    eval_samples: int = 60
    steps: int = 50
    seed: int = 42
    synchronous: bool = True


class NetworkRuleCA:
    def __init__(
        self,
        graph: nx.Graph,
        rule_bits: np.ndarray,
        include_self: bool = True,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.rule_bits = rule_bits.astype(np.int8)
        self.include_self = include_self
        self.rng = np.random.default_rng(seed)

        if len(self.rule_bits) != 11:
            raise ValueError("Network rule must have length 11.")

    def fraction_bin(self, values: List[int]) -> int:
        if len(values) == 0:
            return 0
        frac = sum(values) / len(values)
        return int(round(frac * 10))

    def synchronous_step(self, state: Dict[int, int]) -> Dict[int, int]:
        new_state: Dict[int, int] = {}

        for node in self.graph.nodes():
            values = [state[n] for n in self.graph.neighbors(node)]

            if self.include_self:
                values.append(state[node])

            b = self.fraction_bin(values)
            new_state[node] = int(self.rule_bits[b])

        return new_state

    def asynchronous_step(self, state: Dict[int, int]) -> Dict[int, int]:
        new_state = state.copy()
        nodes = list(self.graph.nodes())
        self.rng.shuffle(nodes)

        for node in nodes:
            values = [new_state[n] for n in self.graph.neighbors(node)]

            if self.include_self:
                values.append(new_state[node])

            b = self.fraction_bin(values)
            new_state[node] = int(self.rule_bits[b])

        return new_state

    def run(
        self,
        initial_state: Dict[int, int],
        steps: int = 50,
        synchronous: bool = True,
    ) -> List[Dict[int, int]]:
        history = [initial_state.copy()]
        current = initial_state.copy()

        for _ in range(steps):
            if synchronous:
                current = self.synchronous_step(current)
            else:
                current = self.asynchronous_step(current)

            history.append(current.copy())

        return history


def graph_state_with_density(
    graph: nx.Graph,
    rho: float,
    rng: np.random.Generator,
) -> Dict[int, int]:
    nodes = list(graph.nodes())
    n_ones = int(round(rho * len(nodes)))

    values = np.zeros(len(nodes), dtype=np.int8)
    values[:n_ones] = 1
    rng.shuffle(values)

    return {node: int(values[i]) for i, node in enumerate(nodes)}


def proportional_graph_score(final_state: Dict[int, int], target: int) -> float:
    values = np.array(list(final_state.values()), dtype=np.int8)

    if target == 1:
        return float(np.mean(values == 1))
    return float(np.mean(values == 0))


def strict_graph_prediction(final_state: Dict[int, int]) -> int | None:
    values = np.array(list(final_state.values()), dtype=np.int8)

    if np.all(values == 0):
        return 0
    if np.all(values == 1):
        return 1

    return None


def evaluate_network_rule(
    graph: nx.Graph,
    rule_bits: np.ndarray,
    n_samples: int = 60,
    steps: int = 50,
    seed: int = 42,
    synchronous: bool = True,
) -> float:
    rng = np.random.default_rng(seed)

    low_rhos = rng.uniform(0.05, 0.49, size=n_samples // 2)
    high_rhos = rng.uniform(0.51, 0.95, size=n_samples - n_samples // 2)
    rhos = np.concatenate([low_rhos, high_rhos])
    rng.shuffle(rhos)

    scores = []

    for rho in rhos:
        init_state = graph_state_with_density(graph, rho, rng)
        target = 1 if rho > 0.5 else 0

        ca = NetworkRuleCA(
            graph=graph,
            rule_bits=rule_bits,
            include_self=True,
            seed=seed,
        )

        history = ca.run(
            initial_state=init_state,
            steps=steps,
            synchronous=synchronous,
        )

        scores.append(proportional_graph_score(history[-1], target))

    return float(np.mean(scores))


def evaluate_network_rule_strict(
    graph: nx.Graph,
    rule_bits: np.ndarray,
    n_samples: int = 60,
    steps: int = 100,
    seed: int = 42,
    synchronous: bool = True,
) -> float:
    rng = np.random.default_rng(seed)

    low_rhos = rng.uniform(0.05, 0.49, size=n_samples // 2)
    high_rhos = rng.uniform(0.51, 0.95, size=n_samples - n_samples // 2)
    rhos = np.concatenate([low_rhos, high_rhos])
    rng.shuffle(rhos)

    correct = 0

    for rho in rhos:
        init_state = graph_state_with_density(graph, rho, rng)
        target = 1 if rho > 0.5 else 0

        ca = NetworkRuleCA(
            graph=graph,
            rule_bits=rule_bits,
            include_self=True,
            seed=seed,
        )

        history = ca.run(
            initial_state=init_state,
            steps=steps,
            synchronous=synchronous,
        )

        pred = strict_graph_prediction(history[-1])

        if pred == target:
            correct += 1

    return correct / n_samples


class NetworkGeneticAlgorithm:
    def __init__(self, graph: nx.Graph, config: NetworkGAConfig) -> None:
        self.graph = graph
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.rule_length = 11

    def initialize_population(self) -> np.ndarray:
        return self.rng.integers(
            0,
            2,
            size=(self.config.population_size, self.rule_length),
            dtype=np.int8,
        )

    def fitness_population(self, population: np.ndarray) -> np.ndarray:
        fitnesses = np.zeros(len(population), dtype=float)

        for i, rule_bits in enumerate(population):
            fitnesses[i] = evaluate_network_rule(
                graph=self.graph,
                rule_bits=rule_bits,
                n_samples=self.config.eval_samples,
                steps=self.config.steps,
                seed=self.config.seed + i,
                synchronous=self.config.synchronous,
            )

        return fitnesses

    def tournament_select(self, population: np.ndarray, fitnesses: np.ndarray) -> np.ndarray:
        indices = self.rng.choice(
            len(population),
            size=self.config.tournament_size,
            replace=False,
        )
        best_idx = indices[np.argmax(fitnesses[indices])]
        return population[best_idx].copy()

    def crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.config.crossover_rate:
            return parent1.copy(), parent2.copy()

        point = self.rng.integers(1, self.rule_length)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])

        return child1.astype(np.int8), child2.astype(np.int8)

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        mask = self.rng.random(self.rule_length) < self.config.mutation_rate
        mutated = individual.copy()
        mutated[mask] = 1 - mutated[mask]
        return mutated.astype(np.int8)

    def evolve(self) -> Tuple[np.ndarray, float, List[float]]:
        population = self.initialize_population()

        best_rule: np.ndarray | None = None
        best_fitness = -1.0
        best_history: List[float] = []

        for gen in range(self.config.generations):
            print(
                f"Network GA generation {gen + 1}/{self.config.generations}...",
                flush=True,
            )

            fitnesses = self.fitness_population(population)
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = float(fitnesses[gen_best_idx])

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_rule = population[gen_best_idx].copy()

            best_history.append(best_fitness)

            print(f"Best network fitness so far = {best_fitness:.4f}", flush=True)

            new_population = [population[gen_best_idx].copy()]

            while len(new_population) < self.config.population_size:
                parent1 = self.tournament_select(population, fitnesses)
                parent2 = self.tournament_select(population, fitnesses)

                child1, child2 = self.crossover(parent1, parent2)
                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)

                if len(new_population) < self.config.population_size:
                    new_population.append(child2)

            population = np.array(new_population, dtype=np.int8)

        if best_rule is None:
            raise RuntimeError("Network GA failed to produce a rule.")

        return best_rule, best_fitness, best_history