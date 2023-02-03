from RepGraph import RepGraph
import matplotlib.pyplot as plt
import random

import networkx as nx
import matplotlib.pyplot as plt

a = RepGraph({1, 2, 3})
a.add_rr({"base": {'ru_1', 'ru_2'}, "label": "rep1"}, {"base": {'ru_1', 'ru_3'}, "label": "rep2"})
a.draw()
plt.show()
