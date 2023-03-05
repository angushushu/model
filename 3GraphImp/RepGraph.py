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

    def __init__(self, rus: set[str] = set(), *, forward_r = 1.0, backward_r = .0, decline_r = .5) -> None:
        self.id_cnt = 0
        self.graph = nx.DiGraph()
        self.nodes = self.graph.nodes
        self.forward_rate = forward_r
        self.backward_rate = backward_r
        self.decline_rate = decline_r
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
        if type(rep1) is dict:
            rep1 = self.add_r(label=rep1["label"], base=rep1["base"])
        if type(rep2) is dict:
            rep2 = self.add_r(label=rep2["label"], base=rep2["base"])
        self.graph.add_edge(rep1, rep2)

    def draw(self):
        print(self.nodes)
        mapping = dict([(Type.ru, '#45ff6d'), (Type.r, '#2edcff')])
        Graph.draw(self.graph, mapping)
    
    # get all ids of all reps that match the label
    def get_id(self, label:str):
        return [x for x,y in self.graph.nodes(data=True) if y['label'] == label]

    # get dict for label:[ids]
    def get_ids(self, labels:list[str]):
        ret = dict()
        for l in labels:
            ret[l] = self.get_id(l)
        return ret
    
    def to_ids(self, labels:set[str]):
        ret = set()
        for l in labels:
            ret.add(self.get_id(l))
        return ret

    def get_label(self, id:str):
        return self.nodes[id]['label']

    # get dict for id:label
    def labels(self, ids:list[str]):
        ret = dict()
        for i in ids:
            ret[i] = self.get_label()
        return ret

    def activate_single(self, id:str, rate:float):
        self.nodes[id]['activation'] += rate # += or =? += indicates no upper bound
        return

    # use ids for uniqueness of each rep
    def activate(self, ids:set[str], rate:float):
        for id in ids:
            self.activate_single(id, rate)
        return

    def decline_single(self, id:str, rate:float):
        self.nodes[id]['activation'] *= rate # += or =? += indicates no upper bound
        return

    # use ids for uniqueness of each rep
    def decline(self, ids:set[str], rate:float):
        for id in ids:
            self.decline_single(id, rate)
        return
    
    def next(self, *, activation = True, decline = True):
        # activation
        if activation:
            activated = [(x,y) for x,y in self.graph.nodes(data=True) if y['activation'] > 0]
            print('activated:', activated)
            for x,y in activated:
                outs = list(self.graph.neighbors(x))
                # print('outs:', outs)
                for out_n in outs:
                    print('test',x,'-',y)
                    print(out_n)
                    self.activate_single(out_n, y['activation']*self.forward_rate)

                ins = list(self.graph.predecessors(x))
                # print('ins:', ins)
                for in_n in ins:
                    self.activate_single(in_n, y['activation']*self.backward_rate)
        # decline
        if decline:
            self.decline(self.graph.nodes, self.decline_rate)
        # NOTE: here the order of activation and decline leads to the fineness issue,
        #       might need to use function on time instead of tick (or at least set into multi stages)
        return
