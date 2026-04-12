from __future__ import annotations

import numpy as np
import networkx as nx
from typing import Dict, List


class GraphThresholdCA:
    """
    Graph-based cellular automaton using a threshold rule.

    Later-stage extension for the project:
    - works on irregular graph topologies
    - supports synchronous and asynchronous updates
    """

    def __init__(
        self,
        graph: nx.Graph,
        threshold: float = 0.5,
        include_self: bool = True,
        seed: int = 42,
    ) -> None:
        self.graph = graph
        self.threshold = threshold
        self.include_self = include_self
        self.rng = np.random.default_rng(seed)

    def random_state(self) -> Dict[int, int]:
        """Generate a random binary state for all graph nodes."""
        return {node: int(self.rng.integers(0, 2)) for node in self.graph.nodes()}

    def synchronous_step(self, state: Dict[int, int]) -> Dict[int, int]:
        """
        Update all nodes at once using the threshold rule.
        """
        new_state: Dict[int, int] = {}

        for node in self.graph.nodes():
            neighbors = list(self.graph.neighbors(node))
            values = [state[n] for n in neighbors]

            if self.include_self:
                values.append(state[node])

            if len(values) == 0:
                avg = state[node]
            else:
                avg = sum(values) / len(values)

            new_state[node] = 1 if avg >= self.threshold else 0

        return new_state

    def asynchronous_step(self, state: Dict[int, int]) -> Dict[int, int]:
        """
        Update nodes one by one in random order.
        """
        new_state = state.copy()
        nodes = list(self.graph.nodes())
        self.rng.shuffle(nodes)

        for node in nodes:
            neighbors = list(self.graph.neighbors(node))
            values = [new_state[n] for n in neighbors]

            if self.include_self:
                values.append(new_state[node])

            if len(values) == 0:
                avg = new_state[node]
            else:
                avg = sum(values) / len(values)

            new_state[node] = 1 if avg >= self.threshold else 0

        return new_state

    def run(
        self,
        initial_state: Dict[int, int],
        steps: int = 50,
        synchronous: bool = True,
    ) -> List[Dict[int, int]]:
        """
        Run the graph CA for a number of steps.
        """
        history = [initial_state.copy()]
        current = initial_state.copy()

        for _ in range(steps):
            if synchronous:
                current = self.synchronous_step(current)
            else:
                current = self.asynchronous_step(current)

            history.append(current.copy())

        return history