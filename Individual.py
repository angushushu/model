import RepGraph
import ConstSAGraph

class Individual:
    def __init__(self) -> None:
        self.rp = RepGraph()
        self.csap = ConstSAGraph()