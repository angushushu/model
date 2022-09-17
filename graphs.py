import networkx as nx
import matplotlib.pyplot as plt
from units import Rep, Act

class RGraph:
    def __init__(self, *, rep_units:set=set()):
        print('initializing RGraph...')
        self.graph = nx.DiGraph()
        for ru_id in rep_units:
            self.graph.add_node(ru_id, type='ru', activation=.0)
    def addRep(self, *, id, base):
        for ru in base:
            if not self.graph.has_node(ru):
                return
        self.graph.add_node(id, type='r', activation=.0)
        for ru in base:
            self.graph.add_edge(ru, id)
    def draw(self):
        nx.draw(self.graph)

class AGraph: # not sure about the graph
    def __init__(self, *, act_units:set=set()):
        print('initializing AGraph...')
        self.graph = nx.DiGraph()
        for au_id in act_units:
            self.graph.add_node(au_id, type='au', activation=.0)
    def addAct(self, *, id, base:list):
        for au in base:
            if not self.graph.has_node(au):
                return
            else:
                
        self.graph.add_node(id, type='r', activation=.0)
        for ru in base:
            self.graph.add_edge(ru, id)
    def draw(self):
        nx.draw(self.graph)

class SAGraph:
    def __init__(self, *, rep_units:set=set(), act_units:set=set()) -> None:
        print('initializing SAGraph...')
        self.next_rid = 0
        self.next_aid = 0
        self.graph = nx.DiGraph()
        self.rep_units:set[Rep] = set()
        self.act_units:set[Act] = set()
        self.reps:set[Rep] = set()
        self.acts:set[Act] = set()
        self.RRedges:list = []
        self.RAedges:list = []
        self.ARedges:list = []
        self.need = None
        self.RRweight = 0.5
        self.RAweight = 0.5
        for r in rep_units:
            rep = Rep(id=self.next_rid, base={r}) # del
            self.reps.add(rep)
            self.graph.add_node(rep, id=self.next_rid, activation=.0)
            self.next_rid+=1
        self.rep_units = self.reps.copy()
        for a in act_units:
            act = Act(id=self.next_aid, base=[a]) # del
            self.acts.add(act)
            self.graph.add_node(act, id=self.next_aid, activation=.0)
            self.next_aid+=1
        self.act_units = self.acts.copy()
    # set one rep as the ultimate goal
    def setNeedById(self, id) -> None:
        print('*',id)
        self.need = next(r for r in self.reps if r.id==id)
    # add one rep
    def addRep(self, base:set=set()) -> None:
        # must formed by existed rep_units
        rep = Rep(id=self.next_rid, base=base)
        print('+ r-'+str(self.next_rid))
        print('?',self.repValid(rep))
        if self.repValid(rep):
            self.reps.add(rep)
            self.graph.add_node(rep)
            # self.edges += list(map(lambda source:(rep,source),rep.base)) # new one to
            for source in self.reps:
                if source.id in base:
                    self.RRedges.append((source,rep)) # to new one
                    self.graph.add_edge(source, rep, weight=self.RRweight, activation=.0)
        self.next_rid += 1
        return self.next_rid-1
    # add multi reps
    def addReps(self, reps:list) -> None:
        for r in reps:
            self.addRep(r)
    # add one act
    def addAct(self, base:list=[]) -> None:
        # must formed by existed act_units
        act = Act(id=self.next_aid, base=base)
        print('+ a-'+str(self.next_aid))
        print('?',self.actValid(act))
        if self.actValid(act):
            self.acts.add(act)
            self.graph.add_node(act)
        self.next_aid += 1
    # add multi acts
    def addActs(self, acts:list) -> None:
        for a in acts:
            self.addAct(a)
    # get rep obj through id
    def getRep(self, id):
        return next((r for r in self.reps if r.id==id),None)
    # get act obj through id
    def getAct(self, id):
        return next((a for a in self.acts if a.id==id),None)
    # check if a rep is valid (based on existing reps)
    def repValid(self, rep:Rep) -> bool:
        for r_id in rep.base:
            if not next((r for r in self.reps if r.id==r_id),None):
                return False
        return True
    # check if a act is valid (based on existing acts)
    def actValid(self, act:Act) -> bool:
        for a_id in act.base:
            if not next((a for a in self.acts if a.id==a_id),None):
                return False
        return True
    # r->r
    def rep2rep(self, rep1_id, rep2_id):
        r1 = self.getRep(rep1_id)
        r2 = self.getAct(rep2_id)
        if r1 and r2:
            self.RRedges.append((r1,r2))
            self.graph.add_edge(r1,r2,weight=self.RRweight,activation=.0)
    # r->a
    def rep2act(self, rep_id, act_id):
        r = self.getRep(rep_id)
        a = self.getAct(act_id)
        if r and a:
            self.RAedges.append((r,a))
            self.graph.add_edge(r,a,weight=self.RAweight,activation=.0)
    # a->r
    def act2rep(self, act_id, rep_id):
        a = self.getAct(act_id)
        r = self.getRep(rep_id)
        if a and r:
            self.ARedges.append((a,r))
            self.graph.add_edge(a,r,weight=self.RAweight,activation=.0)
    # check if a rep exists
    def hasRep(self, rep_id:int)->bool:
        for r in self.reps:
            if r.id==rep_id:
                return True
        return False
    # check if an act exists
    def hasAct(self, act_id:int)->bool:
        for r in self.acts:
            if r.id==act_id:
                return True
        return False
    # get one rep's base by id
    def getBaseById(self, id:int):
        return next(r for r in self.reps if r.id==id).base
    # get all existing reps
    def Reps(self):
        return set(map(lambda x:x.id,self.reps))
    # get all existing rep units
    def RepUnits(self):
        return set(map(lambda x:x.id,self.rep_units))
    # get all existing acts
    def Acts(self):
        return set(map(lambda x:x.id,self.acts))
    # get all existing act units
    def ActUnits(self):
        return set(map(lambda x:x.id,self.act_units))
    # activate one rep
    def activateRep(self, rep_id):
        rep = self.getRep(rep_id)
        if rep:
            rep.activation = 1.0
    # set r->r decline rate
    def setRRWeight(self, weight:float=.5):
        self.RRweight = weight
    # set r<->a decline rate
    def setRRWeight(self, weight:float=.5):
        self.RAweight = weight
    # goes to next tick
    def next(self):
        pass # change to using group
    # match a rep to existing reps
    def matchRep(self, input:set, mode:str='prime'):
        if mode=='prime':
            pass # use group
    # get the shortest paths btw 2 reps
    def shortestPaths(self, input, target, draw=None):
        paths = list(nx.all_shortest_paths(self.graph, self.getRep(input), self.getRep(target)))
        id_paths = list(map(lambda p:list(map(lambda n:n.id,p)),paths))
        if draw:
            self.draw()
            # pos = nx.spring_layout(self.graph, seed=3113794652)
            pos = nx.spring_layout(self.graph) # 要改
            edges = []
            for p in paths:
                for i in range(len(p)-1):
                    edges.append((p[i],p[i+1]))
                nx.draw_networkx_edges(
                    self.graph,
                    pos,
                    edgelist=edges,
                    width=2,
                    alpha=0.5,
                    edge_color="#FC03FC",
                )
        return id_paths
    # draw the graph
    def draw(self, only_reps=None) -> None:
        # pos = nx.spring_layout(self.graph, seed=3113794652)
        pos = nx.spring_layout(self.graph)
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
        if not only_reps:
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
            edgelist=self.RRedges,
            width=2,
            alpha=0.5,
            edge_color="#000",
        )
        if not only_reps:
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=self.RAedges,
                width=2,
                alpha=0.5,
                edge_color="#000",
            )
            nx.draw_networkx_edges(
                self.graph,
                pos,
                edgelist=self.ARedges,
                width=2,
                alpha=0.5,
                edge_color="#000",
            )
        
        labels = {}
        # draw labels
        for r in self.reps:
            labels[r] = r.id
        for r in self.rep_units:
            labels[r] = f'({r.id}){list(r.base)[0]}'
        if not only_reps:
            for a in self.acts:
                labels[a] = a.id
            for a in self.act_units:
                labels[a] = f'({a.id}){list(a.base)[0]}'
        nx.draw_networkx_labels(self.graph, pos, labels, font_size=10, font_color="black")
