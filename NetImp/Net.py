from logging import root
import networkx as nx
import matplotlib.pyplot as plt
from utils import Type
import utils
import Model
from copy import deepcopy

'''
type:
    unit(either rep or act): n = n
    inclusion: n = {a1,a2...an}
        rep
        physical_act
        sequence
    sequence: n = (a1,a2)
function:
    add_node(type, elements, label) -> id
    connect1(elements, target, weights)
        build inclusion relation
    connect2(sequence, target, weight)
        build sequence relation
    remove_node(id) ->
    get_node(id) -> node
    change_weight(node1, node2)
    set_weight(node1, node2)
'''
class Net:
    def __init__(self, rus: set[str] = set()) -> None:
        self.id_cnt = 0
        self.graph = nx.DiGraph()
        self.nodes = self.graph.nodes
        for ru in rus:
            self.add_ru(ru)
    def add_node(self, node_type=Type.ru, label=None, activation=0, bias=0, position=utils.get_pos(x=0)):
        if node_type in [Type.ru, Type.pau]:
            pass
        elif node_type in [Type.r, Type.pa, Type.ma]:
            pass
        elif node_type in [Type.seq]:
            pass





