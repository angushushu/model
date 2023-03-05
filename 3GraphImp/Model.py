from RepGraph import RepGraph
from ActGraph import ActGraph
from SAGraph import SAGraph

class Model:
    
    # sensors = rus
    def __init__(self, sensors:set[str]) -> None:
        self.reps = RepGraph(sensors)
        self.acts = ActGraph()
        self.sag = SAGraph()
    
    # when new stimuli comes, input in labels
    def perceive(self, input:set[str]):
        self.reps.activate(input)
