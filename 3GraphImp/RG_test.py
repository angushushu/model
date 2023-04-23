import networkx as nx
from RepGraph import RepGraph
from ActGraph import ActGraph
from SAGraph import SAGraph
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt
import random
import utils

r = RepGraph({1, 2, 3})

# 5 objects each consists of set of features
# objs = []
# for i in range(0,5):
#     obj = []
#     for j in range(random.randint(0,10)):
#         obj.append(f'{i}.{j}')
#     objs.append(obj)


r.add_rr({"base": {'ru_1', 'ru_2'}, "label": "r_4"}, {"base": {'ru_1', 'ru_3'}, "label": "r_5"})
print(r.get_id('r_4'))
print(r.get_ids(['r_5']))
r.add_r('r_6', r.to_ids({'r_4','r_5'}))
r.add_r('r_7', r.to_ids({'r_2','r_5'}))

r.activate_multi(r.to_ids({1,2}), value=1)
r.draw()
plt.show()
for i in range(0,2):
    # r.regular_spread()
    r.conserv_spread()
    r.draw()
    plt.show()
    r.activate_func(utils.sigmoid)
    r.draw()
    plt.show()
    r.deactivate_all(0.2)
    r.draw()
    plt.show()
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

