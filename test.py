import graphs as g
import units as u
import sqlite3
from units import Rep, Act
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt

# r = g.RGraph(rep_units={0,1,2})
# r.addRep(id=3,base={1,2})
# r.draw()
a = g.AGraph(act_units={0,1,2})
a.addAct(id=3,base=[1,2])
a.draw()
plt.show()

# con = sqlite3.connect('example.db')
# con = sqlite3.connect('rep_graph.db')
# con = sqlite3.connect('act_graph.db')
# con = sqlite3.connect('csa_graph.db')