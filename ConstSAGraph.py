import networkx as nx
import matplotlib.pyplot as plt

class ConstSAGaph:
    def __init__(self, center_id) -> None:
        self.graph = nx.DiGraph()
        self.graph.add_node(center_id)
    
    def show(self) -> None:
        nx.draw(self.graph, with_labels=True, node_color='#555', \
        font_weight='regular', font_color='#fff')
        plt.show()

    def add_node(self, id, to=None) -> None:
        self.graph.add_node(id)
        if to:
            self.graph.add_edge(id, to)
