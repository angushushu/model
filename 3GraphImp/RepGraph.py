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
    # deactivate_r is how many percent will be removed
    # contemporarily added fire_cnt
    def __init__(self, rus: set[str] = set(), *, forward_r = 1.0, backward_r = .0, deactivate_r = .0) -> None:
        self.id_cnt = 0
        self.graph = nx.DiGraph()
        self.nodes = self.graph.nodes
        self.forward_rate = forward_r
        self.backward_rate = backward_r
        self.deactivate_rate = deactivate_r # for activation
        # self.last_activation = []
        for ru in rus:
            self.__add_ru(ru)

    def next_ru_id(self) -> int:
        self.id_cnt += 1
        return 'ru_' + str(self.id_cnt)  # ru_ indicates this is a rep unit

    def next_r_id(self) -> int:
        self.id_cnt += 1
        return 'r_' + str(self.id_cnt)  # r_ indicates this is a rep

    def __add_ru(self, ru:str):
        self.graph.add_node(self.next_ru_id(), type=Type.ru, label=ru, activation=.0, position=Graph.get_pos(x=0), fire_cnt=0)
        print(f"-------------- depth of {self.get_id(ru)} is {nx.get_node_attributes(self.graph, 'position')} -----------------")

    def add_r(self, label: str = None, base: set[str] = set()) -> str:  # base is a set of ids
        print('welcome to add_r')
        print(base)
        # must formed by existed rep_units
        for elt in base:  # base only includes r
            if not elt.split('_')[0] in ['r', 'ru'] or not self.graph.has_node(elt):  # 这个检查也许应该在model和其他type一起完成
                return
        r_id = self.next_r_id()
        label = r_id if label is None else label
        depth = max([self.nodes[x]['position'][0] for x in base]) + 1  # get the max depth of base
        print(f'-------------- depth of {r_id} is {depth} -----------------')
        self.graph.add_node(r_id, type=Type.r, label=label, activation=.0, base=base, position=Graph.get_pos(x=depth), fire_cnt=0)
        print(f"-------------- pos of {r_id} is {self.nodes[r_id]['position']} -----------------")
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
    
    def bokeh_draw(self):
        Graph.bokeh_draw(self.graph)
    
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
            print('to_ids --- label ---', l)
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
        # self.nodes[id]['activation'] = (self.nodes[id]['activation']*rate + value)/max(self.graph.in_degree(id),1) # do i need to do /?
        self.nodes[id]['activation'] = (self.nodes[id]['activation']*rate + value)
        return

    # use ids for uniqueness of each rep
    def activate(self, ids:set[str], *, rate:float=1, value:float=0):
        for id in ids:
            self.activate_single(id, rate=rate, value=value)
        return

    def deactivate_single(self, id:str, *, rate:float=0, value:float=0):
        self.nodes[id]['activation'] *= (1-rate) - value # += or =? += indicates no upper bound
        return

    # use ids for uniqueness of each rep
    def deactivate(self, ids:set[str], *, rate:float=0, value:float=0):
        for id in ids:
            self.deactivate_single(id, rate=rate, value=value)
        return
    
    def next(self, *, activation = True, deactivate = True):
        # activation
        if activation:
            activated = [(x,y) for x,y in self.graph.nodes(data=True) if y['activation'] > 0]
            print('activated:', activated)
            for x,y in activated:
                outs = list(self.graph.neighbors(x))
                '''
                1. activate all out with activation/#outs*forward rate, deactivate w/ deactivate_rate
                2. activate all out with activation*forward rate, deactivate w/ deactivate_rate
                '''
                # forward activation
                for out_n in outs:
                    print('test',x,'-',y)
                    print(out_n)
                    value_trans = y['activation']*self.forward_rate/len(outs)
                    self.activate_single(out_n, rate=self.forward_rate, value=value_trans)
                    # deactivation
                    if deactivate: # so deactivate only happens on those just fired? This keeps the energy kinda balance
                        self.deactivate_single(x, rate=self.deactivate_rate, value=value_trans)
                ins = list(self.graph.predecessors(x))
                # backward activation
                for in_n in ins:
                    value_trans = y['activation']*self.forward_rate/len(in_n)
                    self.activate_single(in_n, rate=self.deactivate_rate, value=value_trans)
                    # deactivation
                    if deactivate: # so deactivate only happens on those just fired? This keeps the energy kinda balance
                        self.deactivate_single(x, rate=self.backward_rate, value=value_trans)
            # for fire_cnt
                self.nodes[x]['fire_cnt'] += 1
                    
            unactivated = [(x,y) for x,y in self.graph.nodes(data=True) if y['activation'] <= 0]
            for x,y in unactivated:
                self.nodes[x]['fire_cnt'] = 0
        # if deactivate:
        #     self.deactivate(self.graph.nodes, self.deactivate_rate)
        # NOTE: here the order of activation and deactivate leads to the fineness issue,
        #       might need to use function on time instead of tick (or at least set into multi stages)
        self.get_cohort()
        return

    # form cohort 1
    # when activation reach the highest level, those activated high level reps form new cohort
    def get_cohort(self):
        out_limit = 1 # for reps with out_degree less than out_limit, consider as elt of new cohort
        activation_limit = 0.5
        fire_cnt_limit = 1
        # temp = [n for n in self.nodes if self.graph.out_degree(n) < out_limit \
        #                      and self.nodes[n]['activation'] > activation_limit]
        # if len(temp) > 1 and set(temp) == set(self.last_activation):
        #     self.add_r(base = set(temp))
        # else:
        #     self.last_activation = temp
        temp = [n for n in self.nodes if self.graph.out_degree(n) < out_limit \
                             and self.nodes[n]['activation'] > activation_limit\
                                and self.nodes[n]['fire_cnt'] > fire_cnt_limit]
        if len(temp) > 1:
            self.add_r(base = set(temp))
        return

    # how two reps merge or gen. new one
    def abstract_intersect():
        pass

    def abstract_consist():
        pass

    def specific_union():
        pass