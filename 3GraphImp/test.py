import networkx as nx
from RepGraph import RepGraph
from ActGraph import ActGraph
from SAGraph import SAGraph
import matplotlib.pyplot as plt
import random

import networkx as nx
import matplotlib.pyplot as plt

r = RepGraph({1, 2, 3})
r.add_rr({"base": {'ru_1', 'ru_2'}, "label": "rep1"}, {"base": {'ru_1', 'ru_3'}, "label": "rep2"})
print(r.get_id('rep1'))
print(r.get_ids(['rep1']))
r.activate(r.get_id('rep1'), 1)
for i in range(0,5):
    r.draw()
    plt.show()
    r.next(activation=True,decline=True)
r.draw()
plt.show()

# a = ActGraph({1, 2})
# act1 = nx.DiGraph()
# act1.add_nodes_from(['au_1', 'au_1'])
# act1.add_edge('au_1', 'au_2')
# a.add_a('act1', act1)
# a.draw()
# plt.show()

# sa = SAGraph({'r_1', 'r_2'})
# sa.add_sas('r_1', 'a_1', 'r_3')
# sa.draw()
# plt.show()

