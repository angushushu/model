from logging import root
import networkx as nx
import matplotlib.pyplot as plt
from utils import Type
import utils
from copy import deepcopy

class RepGraph:
    '''
    id_cnt
    graph
    new_id()
    add_rep(new_id(), label, type)
    connect()
    '''
    # deactivate_r is how many percent will be removed
    # contemporarily added fire_cnt
    def __init__(self, rus: set[str] = set()) -> None:
        self.id_cnt = 0
        self.graph = nx.DiGraph()
        self.nodes = self.graph.nodes
        for ru in rus:
            self.add_ru(ru)

    def next_ru_id(self) -> int:
        self.id_cnt += 1
        return 'ru_' + str(self.id_cnt)  # ru_ indicates this is a rep unit

    def next_r_id(self) -> int:
        self.id_cnt += 1
        return 'r_' + str(self.id_cnt)  # r_ indicates this is a rep

    def add_ru(self, ru:str, bias:float=0):
        self.graph.add_node(self.next_ru_id(), type=Type.ru, label=ru, activation=.0, bias=bias, position=utils.get_pos(x=0))

    def add_r(self, label: str = None, base: set[str] = set(), *, for_w:float=1, back_w:float=0, bias:float = 0, graph:nx.graph = None) -> str:  # base is a set of ids
        # adding graph as r: add_r(label, graph=graph)
        if graph is not None:
            base = graph.nodes
            self.graph.add_node(r_id, type=Type.r, label=label, activation=.0, base=base, bias=bias, position=utils.get_pos(x=depth), graph=graph)
        print('welcome to add_r')
        print(base)
        # must formed by existed rep_units
        for elt in base:  # base only includes r
            if not elt.split('_')[0] in ['r', 'ru'] or not self.graph.has_node(elt):  # 这个检查也许应该在model和其他type一起完成
                return
        r_id = self.next_r_id()
        label = r_id if label is None else label
        depth = max([self.nodes[x]['position'][0] for x in base]) + 1  # get the max depth of base
        self.graph.add_node(r_id, type=Type.r, label=label, activation=.0, base=base, bias=bias, position=utils.get_pos(x=depth))
        for elt in base:
            if str(elt).split('_')[0] in ['r', 'ru']:
                self.graph.add_edge(elt, r_id, for_w=for_w, back_w=back_w)
        return r_id

    def add_rr(self, rep1, rep2, *, for_w:float=1, back_w:float=0) -> None:
        print('rep1:', rep1)
        if type(rep1) is dict:
            rep1 = self.add_r(label=rep1["label"], base=rep1["base"])
        if type(rep2) is dict:
            rep2 = self.add_r(label=rep2["label"], base=rep2["base"])
        self.graph.add_edge(rep1, rep2, for_w=for_w, back_w=back_w)

    def draw(self):
        print(self.nodes)
        mapping = dict([(Type.ru, '#45ff6d'), (Type.r, '#2edcff')])
        utils.draw(self.graph, mapping)
    
    def bokeh_draw(self):
        utils.bokeh_draw(self.graph)
    
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
            ret.update(self.get_id(l))
        return ret

    def get_label(self, id:str):
        return self.nodes[id]['label']

    # get dict for id:label
    def labels(self, ids:list[str]):
        ret = dict()
        for i in ids:
            ret[i] = self.get_label()
        return ret

    def activate_single(self, id:str, *, rate:float=1, value:float=0):
        # should activation funciton used?
        print(id,'--',rate,value)
        self.nodes[id]['activation'] = (self.nodes[id]['activation']*rate + value)
        return

    # use ids for uniqueness of each rep
    def activate_multi(self, ids:set[str], *, rate:float=1, value:float=0):
        for id in ids:
            self.activate_single(id, rate=rate, value=value)
        return

    def deactivate_single(self, id:str, *, rate:float=0, value:float=0):
        self.nodes[id]['activation'] = self.nodes[id]['activation']*(1-rate) - value
        return

    # use ids for uniqueness of each rep
    def deactivate_multi(self, ids:set[str], *, rate:float=0, value:float=0):
        for id in ids:
            self.deactivate_single(id, rate=rate, value=value)
        return
    
    def deactivate_all(self, rate:float=1):
        for node in self.nodes:
            self.deactivate_single(node, rate=rate)

    def activate_func(self, func=None):
        if func is None:
            return
        else:
            for _,data in self.nodes(data=True):
                data['activation'] = func(data['activation'])


    # need to run regular function
    def regular_spread(self, *, deact_r:float=1, self_w:float=0): # self_w for circle, for test
        temp = [deepcopy((x,y)) for x,y in self.nodes(data=True)]
        for node,data in temp:
            # forward
            outs = list(self.graph.neighbors(node))
            for out_n in outs:
                value_trans = data['activation']*self.graph.get_edge_data(node, out_n)['for_w']
                self.activate_single(out_n, value=value_trans)
            # backward
            ins = list(self.graph.predecessors(node))
            for in_n in ins:
                value_trans = data['activation']*self.graph.get_edge_data(in_n, node)['back_w']
                self.activate_single(in_n, value=value_trans)
            # self?
            self.activate_single(node, value=data['activation']*self_w)
            # bias
            self.activate_single(node, value=data['bias'])

    def conserv_spread(self, *, for_w:float=1, back_w:float=0, self_w:float=0):
        # for_w & back_w will be forced set for all edges
        temp = [deepcopy((x,y)) for x,y in self.nodes(data=True)]
        for node,data in temp:
            if data['activation'] > 0:
                # forward
                outs = list(self.graph.neighbors(node))
                value_trans = data['activation']*for_w/max(1,len(outs))
                for out_n in outs:
                    self.activate_single(out_n, value=value_trans)
                # backward
                ins = list(self.graph.predecessors(node))
                value_trans = data['activation']*back_w/max(1,len(ins))
                for in_n in ins:
                    self.activate_single(in_n, value=value_trans)
                #self?
                self.activate_single(node, value=data['activation']*self_w)
    
    def get_activated(self, filter:float=0.5):
        return [(node,data) for node,data in self.graph.nodes(data=True) if data['activation'] > filter]        

    # how two reps merge or gen. new one
    def abstract_intersect():
        pass

    def abstract_consist():
        pass

    def specific_union():
        pass