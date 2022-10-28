import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from Arrow3D import Arrow3D
from Annotation3D import Annotation3D
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
        return next((r for r in self.graph.nodes(data=True) if r[0]==id),None)
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
        # determine z axis based on type
        node_xyz = np.empty([0,3])
        node_id = np.empty([0])
        for v in sorted(self.graph.nodes(data=True)):
            print('v:', v)
            node_id = np.append(node_id, [v[0]]) # for label
            if v[1]['type'] == 'ru':
                ru_pos = pos[int(v[0])]
                print('ru_pos_1:',ru_pos.shape)
                ru_pos[2] = 0
                print('ru_pos_2:', ru_pos)
                print('ru_pos:',ru_pos)
                node_xyz = np.append(node_xyz, [ru_pos], axis=0)
            elif v[1]['type'] == 'r':
                ru_pos = pos[int(v[0])]
                print('ru_pos_1:',ru_pos.shape)
                ru_pos[2] = self._rep_depth(v)
                print('ru_pos_2:', ru_pos)
                print('ru_pos:',ru_pos)
                node_xyz = np.append(node_xyz, [ru_pos], axis=0)
            else:
                node_xyz = np.append(node_xyz, [pos[v[0]]], axis=0)
        print(node_xyz.shape)
        # determine style based type (RAR, RR)
        # edge_xyz = np.array([(pos[u],pos[v]) for u,v in self.graph.edges()])
        RR_edge_xyz = np.empty([0,2,3])
        RAR_edge_xyz = np.empty([0,2,3])
        print('RR_edge_xyz', RR_edge_xyz)
        for e in self.graph.edges(data=True):
            print('e:',e)
            if 'act' in e[2]:
                print('RAR')
                print('[pos[e[0]], pos[e[1]]]:', [pos[e[0]], pos[e[1]]])
                RAR_edge_xyz = np.append(RAR_edge_xyz, [[pos[e[0]], pos[e[1]]]], axis=0)
            else:
                print('RR')
                RR_edge_xyz = np.append(RR_edge_xyz, [[pos[e[0]], pos[e[1]]]], axis=0)
        print(RR_edge_xyz)
        print(RAR_edge_xyz)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Plot the nodes - alpha is scaled by "depth" automatically
        ax.scatter(*node_xyz.T, s=100, ec="w")
        for i in range(0,len(node_id)):
            ax.text(*node_xyz[i].T, str(node_id[i]))
            # self._annotate3D(ax, str(node_id[i]), node_xyz[i], fontsize=10, xytext=(-3,3), textcoords='offset points', ha='right',va='bottom')
        # Plot the edges
        for vizedge in RR_edge_xyz:
            print('vizedge:',vizedge[0])
            ax.plot(*vizedge.T, color="tab:gray") # for bidirectional
            # edges = Arrow3D(*vizedge.T, mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
            # ax.add_artist(edges)
        for vizedge in RAR_edge_xyz:
            self._arrow3D(ax, *vizedge.T, mutation_scale=20, lw=3, arrowstyle="-|>", color="r")

        # mapping = dict([('ru','#45bf5f'),('r','#2599b0')])
        # drawGraph(g=self.graph, mapping=mapping)
        self._format_axes(ax)
        fig.tight_layout()
        plt.show()

    def _format_axes(self, ax):
        """Visualization options for the 3D axes."""
        # Turn gridlines off
        ax.grid(False)
        ax.set_zlim(bottom=0.)
        # Suppress tick labels
        for dim in (ax.xaxis, ax.yaxis, ax.zaxis):
            dim.set_ticks([])
        # Set axes labels
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
    
    def _rep_depth(self, rep): # using recursion here, but will there be loop?
        print(type(rep))
        if rep[1]['type'] == 'ru':
            return 0
        else:
            return 1+max([self._rep_depth(self.getRep(elt)) for elt in rep[1]['base']])

    def _arrow3D(self, ax, s, xyz, *args, **kwargs):
        edges = Arrow3D(s, xyz, *args, **kwargs)
        ax.add_artist(edges)

    def _annotate3D(self, ax, s, *args, **kwargs):
        tag = Annotation3D(s, *args, **kwargs)
        ax.add_artist(tag)
