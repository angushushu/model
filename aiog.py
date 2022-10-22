import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from graphs import drawGraph

# all in one graph
class Graph:
    def __init__(self, rep_units:set=set(), act_units:set=set()) -> None:
        print('initializing SAGraph...')
        self.graph = nx.DiGraph()
        self.actions = []
        self.need = None
        for ru_id in rep_units:
            self.graph.add_node(ru_id, type='ru', label='', activation=.0)
        for au_id in act_units:
            self.actions.append({'id':au_id,'type':'au','label':'','activation':.0})
    # set goals
    def setNeedById(self, id) -> None:
        print('*',id)
        self.need = next(r for r in self.graph.nodes if r.id==id)
    # add one rep
    def addRep(self, id, base:set=set()) -> None:
        # must formed by existed rep_units
        for r in base:
            if not self.graph.has_node(r):
                return
        self.graph.add_node(id, type='r', label='', activation=.0, base=base)
        for r in base:
            self.graph.add_edge(r,id)
        return id
    # get rep obj through id
    def getRep(self, id):
        return next((r for r in self.reps if r.id==id),None)
    def allRep(self):
        for x,y in self.graph.nodes(data=True):
            print(x,y)
        return [(id, data) for id,data in self.graph.nodes(data=True) if data['type']=='ru' or data['type']=='r']
    def addAct(self, id, base:list): # list of act id
        ids = [a['id'] for a in self.actions]
        for a in base:
            if not a in ids:
                return
        print('adding',{'id':id,'type':'a','label':'','activation':.0})
        self.actions.append({'id':id,'type':'a','label':'','activation':.0,'base':base})
        return id
    # set one act btw 2 reps
    def addRAR(self, rep1, act, rep2) -> None:
        if type(act) is dict:
            act = self.addAct(id=act['id'],base=act['base'])
        if type(rep1) is dict and type(rep2) is dict:
            rep1 = self.addRep(id=rep1['id'], base=rep1['base'])
            rep2 = self.addRep(id=rep2['id'], base=rep2['base'])
        self.graph.add_edge(rep1, rep2, act=act)
    # set direct connect btw 2 reps
    def addRR(self, rep1, rep2) -> None:
        if type(rep1) is dict and type(rep2) is dict:
            rep1 = self.addRep(id=rep1['id'], base=rep1['base'])
            rep2 = self.addRep(id=rep2['id'], base=rep2['base'])
        self.graph.add_edge(rep1, rep2)
    # draw the graph
    def draw(self) -> None:
        pos = nx.spring_layout(self.graph, dim=3, seed=779)
        node_xyz = np.array([pos[v] for v in sorted(self.graph)])
        edge_xyz = np.array([(pos[u],pos[v]) for u,v in self.graph.edges()])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w")

        # Plot the edges
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")

        # mapping = dict([('ru','#45bf5f'),('r','#2599b0')])
        # drawGraph(g=self.graph, mapping=mapping)
        self._format_axes(ax)
        fig.tight_layout()
        plt.show()

    def _format_axes(self, ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
