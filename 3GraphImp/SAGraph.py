from logging import root
import networkx as nx
import matplotlib.pyplot as plt
from utils import Type
import utils

class SAGraph:
    def __init__(self, need: set = None) -> None:
        self.graph = nx.DiGraph()
        if need is not None:
            for goal_id in need: # the goal id should be a rep id
                self.graph.add_node(goal_id, type=Type.g, label=goal_id, activation=.0, position=utils.get_pos(x=0), fire_cnt=0)

    # add state
    def add_s(self, state_id: str, label: str = None) -> str:
        if label is None:
            label = state_id
        self.graph.add_node(state_id, type=Type.s, label=label, activation=.0, position=utils.get_pos(x=1), fire_cnt=0)
        return state_id

    # state to state path (state use id)
    # validation is done by model
    def add_ss(self, s1: str, s2: str, a=None) -> None:
        if not self.graph.has_node(s1):
            self.add_s(s1, s1)
        if not self.graph.has_node(s2):
            self.add_s(s2, s2)
        self.graph.add_edge(s1, s2, act=a, type=Type.c3)

    def draw(self):
        print(self.graph.nodes)
        mapping = dict([(Type.g, '#ff4a4a'),(Type.s, '#32a852'), (Type.a, '#000')])
        utils.draw(self.graph, mapping)
    
    def get_all_s(self):
        return self.graph.nodes
