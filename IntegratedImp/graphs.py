from cProfile import label
from logging import root
import networkx as nx
import matplotlib.pyplot as plt

def drawGraph(*, g, mapping):
    print('Shared drawing func')
    pos = nx.spring_layout(g)
    nodes = g.nodes
    
    # print(nodes[40]['type'])
    colors = [mapping[nodes[n]['type']] for n in nodes]
    labels = {}
    # draw labels
    for n in nodes:
        if nodes[n]['label'] != '':
            labels[n] = nodes[n]['label']
        else:
            labels[n] = n
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes,
        node_color=colors
    )
    nx.draw_networkx_labels(
        g,
        pos,
        labels=labels,
        font_color='#000'
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=g.edges,
        width=2,
        alpha=0.5,
        edge_color="#000",
    )
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=nx.get_edge_attributes(g,'act')
    )

class RGraph:
    def __init__(self, *, rep_units:set=set()):
        print('initializing RGraph...')
        self.graph = nx.DiGraph()
        for ru_id in rep_units:
            self.graph.add_node(ru_id, type='ru', label='', activation=.0)
    def addRep(self, *, id, base):
        for r in base:
            if not self.graph.has_node(r):
                return
        self.graph.add_node(id, type='r', label='', activation=.0)
        for r in base:
            self.graph.add_edge(r, id)
    def setLabel(self, *, id, label):
        nx.set_node_attributes(self.graph, {id:label}, name="label")
    def draw(self):
        mapping = dict([('ru','#45bf5f'),('r','#2599b0')])
        drawGraph(g=self.graph,mapping=mapping)
        # nx.draw(self.graph,with_labels=True)


class AGraph: # not sure about the graph
    def __init__(self, *, act_units:set=set()):
        print('initializing AGraph...')
        self.graph = nx.DiGraph()
        for au_id in act_units:
            self.graph.add_node(au_id, type='au', label='', activation=.0)
            # print(self.graph.nodes[au_id])
    def addAct(self, *, id, base:list):
        for a in base:
            if not self.graph.has_node(a):
                return
        nx.add_cycle(self.graph, [id]+base+[id])
        nx.set_node_attributes(self.graph, {id:{'type':'a','label':'','activation':.0}})
    def Cycles(self, *, act):
        print(sorted(nx.simple_cycles(self.graph)))
    def draw(self):
        mapping = dict([('au','red'),('a','yellow')])
        drawGraph(g=self.graph,mapping=mapping)

class Actions: # not in graph form
    def __init__(self, *, act_units:set=set()):
        print('initializing Actions...')
        self.actions = []
        for au_id in act_units: # au
            self.actions.append({'id':au_id,'type':'au','label':'','activation':.0})
    def addAct(self, *, id, base:list): # list of act id
        ids = [a['id'] for a in self.actions]
        for a in base:
            if not a in ids:
                return
        print('adding',{'id':id,'type':'a','label':'','activation':.0})
        self.actions.append({'id':id,'type':'a','label':'','activation':.0,'base':base})
    def allAct(self, *, id_only=False):
        if id_only:
            return [a['id'] for a in self.actions]
        return self.actions
    def getAct(self, *, id):
        for a in self.actions:
            if a['id']==id:
                return a
        return None

class EGraph: # 用于存储任何预期图，需求图为该类的特殊情况
    def __init__(self, *, rep_units:set=set()) -> None:
        print('initializing SAGraph...')
        self.graph = nx.DiGraph()
        self.need = None
        for ru_id in rep_units:
            self.graph.add_node(ru_id, type='ru', label='', activation=.0)
    # set one rep as the ultimate goal
    def setNeed(self, *, need) -> None:
        if type(need) is dict:
            need = self.addRep(id=need['id'], base=need['base'])
        print('*',id)
        self.need = next(r for r in self.graph.nodes if r.id==id)
    # add one rep
    def addRep(self, *, id, base:set=set()) -> None:
        # must formed by existed rep_units
        for r in base:
            if not self.graph.has_node(r):
                return
        self.graph.add_node(id, type='r', label='', activation=.0, base=base)
        return id
    # set one act btw 2 reps
    def setAct(self, *, rep1, act, rep2) -> None:
        if type(rep1) is dict and type(rep2) is dict:
            rep1 = self.addRep(id=rep1['id'], base=rep1['base'])
            rep2 = self.addRep(id=rep2['id'], base=rep2['base'])
        self.graph.add_edge(rep1, rep2, act=act)
    # get rep obj through id
    def getRep(self, id):
        return next((r for r in self.reps if r.id==id),None)
    # draw the graph
    def draw(self) -> None:
        mapping = dict([('ru','#45bf5f'),('r','#2599b0')])
        drawGraph(g=self.graph, mapping=mapping)
