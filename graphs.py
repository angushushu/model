import networkx as nx
import matplotlib.pyplot as plt
from units import Rep, Act

class RSAGraph:
    def __init__(self, *, rep_units:set=set(), act_units:set=set()) -> None:
        print('initializing...')
        self.next_rid = 0
        self.next_aid = 0
        self.graph = nx.DiGraph()
        self.rep_units:set[Rep] = set()
        self.act_units:set[Act] = set()
        self.reps:set[Rep] = set()
        self.acts:set[Act] = set()
        self.edges:list = []
        self.need = None
        self.graph.add_nodes_from(self.rep_units)
        for r in rep_units:
            rep = Rep(id=self.next_rid, base={r})
            self.reps.add(rep)
            self.graph.add_node(rep)
            self.next_rid+=1
        self.rep_units = self.reps.copy()
        for a in act_units:
            act = Act(id=self.next_aid, base=[a])
            self.acts.add(act)
            self.graph.add_node(act)
            self.next_aid+=1
        self.act_units = self.acts.copy()
    def setNeedById(self, id) -> None:
        self.need = next(r for r in self.reps if r.id==id)
    def addRep(self, base:set=set()) -> None:
        # must formed by existed rep_units
        rep = Rep(id=self.next_rid, base=base)
        print('adding'+str(self.next_rid))
        print(self.repValid(rep))
        if self.repValid(rep):
            self.reps.add(rep)
            self.graph.add_node(rep)
            # self.edges += list(map(lambda source:(rep,source),rep.base)) # new one to
            for source in self.reps:
                if source.id in base:
                    self.edges.append((source,rep)) # to new one
        self.next_rid += 1
        return self.next_rid-1
    def addReps(self, reps:list) -> None:
        for r in reps:
            self.addRep(r)
    def addAct(self, base:list=[]) -> None:
        # must formed by existed act_units
        act = Act(id=self.next_aid, base=base)
        print('adding a'+str(self.next_aid))
        print(self.actValid(act))
        if self.actValid(act):
            self.acts.add(act)
            self.graph.add_node(act)
        self.next_aid += 1
    def addActs(self, acts:list) -> None:
        for a in acts:
            self.addAct(a)
    def getRepById(self, id):
        return next(r for r in self.reps if r.id==id)
    def repValid(self, rep:Rep) -> bool:
        for r_id in rep.base:
            if not next((r for r in self.reps if r.id==r_id),None):
                return False
        return True
    def actValid(self, act:Act) -> bool:
        for a_id in act.base:
            if not next((a for a in self.acts if a.id==a_id),None):
                return False
        return True
    def rep2rep(self, rep1_id, rep2_id):
        if self.hasRep(rep1_id) and self.hasRep(rep2_id):
            self.graph.add_edge(next((r for r in self.reps if r.id==rep1_id), None),\
                next((r for r in self.reps if r.id==rep2_id), None))
    def rep2act(self, rep_id, act_id):
        if self.hasRep(rep_id) and self.hasAct(act_id):
            self.graph.add_edge(next((r for r in self.reps if r.id==rep_id), None),\
                next((a for a in self.acts if r.id==act_id), None))
    def hasRep(self, rep_id:int)->bool:
        for r in self.reps:
            if r.id==rep_id:
                return True
        return False
    def hasAct(self, act_id:int)->bool:
        for r in self.acts:
            if r.id==act_id:
                return True
        return False
    def getBaseById(self, id:int):
        return next(r for r in self.reps if r.id==id).base
    def Reps(self):
        return set(map(lambda x:x.id,self.reps))
    def RepUnits(self):
        return set(map(lambda x:x.id,self.rep_units))
    def Acts(self):
        return set(map(lambda x:x.id,self.acts))
    def ActUnits(self):
        return set(map(lambda x:x.id,self.act_units))
    def show(self) -> None:
        pos = nx.spring_layout(self.graph, seed=3113794652)
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=self.reps,
            node_color='#42f5ef'
        )
        if self.need:
            nx.draw_networkx_nodes(
                self.graph,
                pos,
                nodelist=[self.need],
                node_color='#fc03db',
                node_shape="D"
            )
        print('drawing rep units...')
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=self.rep_units,
            node_color='#42f566'
        )
        print('drawing acts...')
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=self.acts,
            node_color='#f5d142'
        )
        print('drawing act units...')
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            nodelist=self.act_units,
            node_color='#f54254'
        )
        nx.draw_networkx_edges(
            self.graph,
            pos,
            edgelist=self.edges,
            width=2,
            alpha=0.5,
            edge_color="#000",
        )
        labels = {}
        # draw labels
        for r in self.reps:
            labels[r] = r.id
        for r in self.rep_units:
            labels[r] = list(r.base)[0]
        for a in self.acts:
            labels[a] = a.id
        for a in self.act_units:
            labels[a] = a.base[0]
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10, font_color="black")
        plt.show()
