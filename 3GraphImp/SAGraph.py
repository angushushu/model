from logging import root
import networkx as nx
import matplotlib.pyplot as plt
from EltType import Type
import Graph


class SAGraph:
    def __init__(self, need: set = set()) -> None:
        self.graph = nx.DiGraph()
        for goal_id in need: # the goal id should be a rep id
            self.graph.add_node(goal_id, type=Type.g, label=goal_id, activation=.0)

    # add state
    def add_s(self, state_id, label: str = None) -> str:
        if not label:
            label = state_id
        self.graph.add_node(state_id, type=Type.s, label=label, activation=.0)
        return state_id

    # state to state path
    def add_ss(self, s1, s2) -> None:
        if not self.graph.has_node(s1):
            self.add_s(s1, s1)
        if not self.graph.has_node(s2):
            self.add_s(s2, s2)
        self.graph.add_edge(s1, s2, type=Type.c3)

    # state to state w/ action path
    def add_sas(self, s1, a, s2) -> None:
        if not self.graph.has_node(s1):
            self.add_s(s1, s1)
        if not self.graph.has_node(s2):
            self.add_s(s2, s2)
        self.graph.add_edge(s1, s2, act=a, type=Type.c2)

    def draw(self):
        print(self.graph.nodes)
        mapping = dict([(Type.g, '#ff4a4a'),(Type.s, '#32a852'), (Type.a, '#000')])
        Graph.draw(self.graph, mapping)
