from cProfile import label
from logging import root
import networkx as nx
import matplotlib.pyplot as plt
from EltType import Type
import Graph


class ActGraph:
    def __init__(self, act_units: set = set()) -> None:
        self.id_cnt = 0
        self.graph = nx.DiGraph()
        for au in act_units:
            self.graph.add_node(self.next_au_id(), type=Type.au, label=au, activation=.0)

    def next_au_id(self) -> int:
        self.id_cnt += 1
        return 'au_' + str(self.id_cnt)  # au_ indicates this is an act unit

    def next_a_id(self) -> int:
        self.id_cnt += 1
        return 'a_' + str(self.id_cnt)  # r_ indicates this is an act

    # set goals
    def add_a(self, label: str, base: nx.DiGraph = None) -> str:  # the base should be a graph
        # a problem: how to distribute input rep to multi sub acts?
        # should act be implemented using continuous NN or using discrete mapping (set of pairs)?
        # what are the basic mental actions? If we can't identify them,
        # can we construct local NN for each mental action?
        if base:
            for elt in base.nodes:  # base only includes r
                if not elt.split('_')[0] in ['a', 'au'] or not self.graph.has_node(elt):  # 这个检查也许应该在model和其他type一起完成
                    return
        else:
            return
        a_id = self.next_a_id()
        self.graph.add_node(a_id, type=Type.a, label=label, activation=.0, base=base)
        for elt in base.nodes:
            if elt.split('_')[0] in ['a', 'au']:
                self.graph.add_edge(elt, a_id)
        return a_id

    def draw(self):
        print(self.graph.nodes)
        mapping = dict([(Type.au, '#45bf00'), (Type.a, '#2599b0')])
        Graph.draw(self.graph, mapping)

