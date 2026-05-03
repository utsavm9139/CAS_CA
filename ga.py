from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

from ca_1d import CellularAutomaton1D
from evaluation import evaluate_lookup_rule_accuracy


@dataclass
class GAConfig:
    population_size: int = 100
    generations: int = 100
    mutation_rate: float = 0.03
    crossover_rate: float = 0.9
    tournament_size: int = 5
    eval_samples: int = 300
    max_steps: int = 320
    seed: int = 42


class GeneticAlgorithmCA:
    """
    Genetic Algorithm for evolving 128-bit cellular automaton rules.
    """

    def __init__(self, ca: CellularAutomaton1D, config: GAConfig) -> None:
        self.ca = ca
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.rule_length = 128

    def initialize_population(self) -> np.ndarray:
        return self.rng.integers(
            0,
            2,
            size=(self.config.population_size, self.rule_length),
            dtype=np.int8,
        )

    def fitness_population(self, population: np.ndarray) -> np.ndarray:
        fitnesses = np.zeros(len(population), dtype=float)

        for i, individual in enumerate(population):
            fitnesses[i] = evaluate_lookup_rule_accuracy(
                ca=self.ca,
                rule_bits=individual,
                n_samples=self.config.eval_samples,
                max_steps=self.config.max_steps,
                seed=self.config.seed + i,
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

    def single_point_crossover(
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
        best_fitness_history: List[float] = []

        best_rule: np.ndarray | None = None
        best_fitness = -1.0

        for gen in range(self.config.generations):
            print(f"Evaluating generation {gen + 1}/{self.config.generations}...", flush=True)

            fitnesses = self.fitness_population(population)

            gen_best_idx = int(np.argmax(fitnesses))
            gen_best_fitness = float(fitnesses[gen_best_idx])

            if gen_best_fitness > best_fitness:
                best_fitness = gen_best_fitness
                best_rule = population[gen_best_idx].copy()

            best_fitness_history.append(best_fitness)

            print(
                f"Generation {gen + 1:3d} | best fitness so far = {best_fitness:.4f}",
                flush=True,
            )

            new_population = []

            # Elitism: keep best rule from current generation.
            new_population.append(population[gen_best_idx].copy())

            while len(new_population) < self.config.population_size:
                parent1 = self.tournament_select(population, fitnesses)
                parent2 = self.tournament_select(population, fitnesses)

                child1, child2 = self.single_point_crossover(parent1, parent2)

                child1 = self.mutate(child1)
                child2 = self.mutate(child2)

                new_population.append(child1)

                if len(new_population) < self.config.population_size:
                    new_population.append(child2)

            population = np.array(new_population, dtype=np.int8)

        if best_rule is None:
            raise RuntimeError("GA failed to produce a best rule.")

        return best_rule, best_fitness, best_fitness_history