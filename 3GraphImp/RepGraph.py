from cProfile import label
from logging import root
import networkx as nx
import matplotlib.pyplot as plt
from EltType import Type
import Graph

class RepGraph:
    '''
    id_cnt
    graph
    new_id()
    add_rep(new_id(), label, type)
    connect()
    '''

    def __init__(self, rus: set = set()):
        self.id_cnt = 0
        self.graph = nx.DiGraph()
        for ru in rus:
            self.graph.add_node(self.next_ru_id(), type=Type.ru, label=ru, activation=.0)

    def next_ru_id(self) -> int:
        self.id_cnt += 1
        return 'ru_' + str(self.id_cnt)  # ru_ indicates this is a rep unit

    def next_r_id(self) -> int:
        self.id_cnt += 1
        return 'r_' + str(self.id_cnt)  # r_ indicates this is a rep

    def add_r(self, label: str, base: set[str] = set()) -> str:  # base is a set of ids
        # must formed by existed rep_units
        for elt in base:  # base only includes r
            if not elt.split('_')[0] in ['r', 'ru'] or not self.graph.has_node(elt):  # 这个检查也许应该在model和其他type一起完成
                return
        r_id = self.next_r_id()
        self.graph.add_node(r_id, type=Type.r, label=label, activation=.0, base=base)
        for elt in base:
            if str(elt).split('_')[0] in ['r', 'ru']:
                self.graph.add_edge(elt, r_id)
        return r_id

    def add_rr(self, rep1, rep2) -> None:
        print('rep1:', rep1)
        if type(rep1) is dict and type(rep2) is dict:
            rep1 = self.add_r(label=rep1["label"], base=rep1["base"])
            rep2 = self.add_r(label=rep2["label"], base=rep2["base"])
        self.graph.add_edge(rep1, rep2)

    def draw(self):
        print(self.graph.nodes)
        mapping = dict([(Type.ru, '#45bf5f'), (Type.r, '#2599b0')])
        Graph.draw(self.graph, mapping)
