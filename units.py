# use this for graphs
class Rep:
    def __init__(self, *, id:int, base:set, activation:float=.0):
        self.id = id
        self.base = base
        self.activation = activation

class Act:
    def __init__(self, *, id:int, base:list, activation:float=.0):
        self.id = id
        self.base = base
        self.activation = activation
class Edge:
    def __init__(self, *, ends:tuple, count:int=0, weight:float=.5) -> None:
        self.ends = ends
        self.count = count
        self.weight = weight
    
