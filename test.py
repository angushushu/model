import graphs as g
import units as u
import sqlite3
from units import Rep, Act
import matplotlib.pyplot as plt
import random

import networkx as nx
import matplotlib.pyplot as plt

# example of RGraph
# rep_units = set()
# for i in range(0,30):
#     rep_units.add(i)
# r = g.RGraph(rep_units=rep_units)
# for i in range(30,40):
#     base = set()
#     for j in range(0, random.choice(range(0,30))):
#         base.add(random.choice(range(0,30)))
#     r.addRep(id=i,base=base)
# for i in range(40,45):
#     base = set()
#     for j in range(0, random.choice(range(0,40))):
#         base.add(random.choice(range(0,40)))
#     r.addRep(id=i,base=base)
# r.graph.nodes()
# r.draw()
# example of AGraph
# act_units = set()
# for i in range(0,20):
#     act_units.add(i)
# print(act_units)
# a = g.AGraph(act_units=act_units)
# for i in range(20,30):
#     base = []
#     for j in range(0, random.choice(range(0,20))):
#         base.append(random.choice(range(0,20)))
#     a.addAct(id=i,base=base)
# for i in range(30,35):
#     base = []
#     for j in range(0, random.choice(range(0,30))):
#         base.append(random.choice(range(0,30)))
#     a.addAct(id=i,base=base)
# a.Cycles(act=25)
# a.draw()
a = g.Actions(act_units={1,2,3})
a.addAct(id=4,base=[1,2])
print(a.allAct())
print(a.allAct(id_only=True))
print(a.getAct(id=1))
print(a.getAct(id=4))
print(a.allAct(id_only=True))
plt.show()

# con = sqlite3.connect('example.db')
# con = sqlite3.connect('rep_graph.db')
# con = sqlite3.connect('act_graph.db')
# con = sqlite3.connect('csa_graph.db')