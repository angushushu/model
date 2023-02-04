import networkx as nx
from RepGraph import RepGraph
from ActGraph import ActGraph
import matplotlib.pyplot as plt
import random

import networkx as nx
import matplotlib.pyplot as plt

r = RepGraph({1, 2, 3})
r.add_rr({"base": {'ru_1', 'ru_2'}, "label": "rep1"}, {"base": {'ru_1', 'ru_3'}, "label": "rep2"})
r.draw()
plt.show()

a = ActGraph({1, 2})
act1 = nx.DiGraph()
act1.add_nodes_from(['au_1', 'au_1'])
act1.add_edge('au_1', 'au_2')
a.add_a('act1', act1)
a.draw()
plt.show()

