import graphs as g
import units as u
import sqlite3
from units import Rep, Act
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt

a = g.RSAGraph(rep_units={'r0','r1','r2'},act_units={'a0','a1'})
print('reps_units:',a.RepUnits())
print('reps:',a.Reps())
a.addRep({0,1})
print('reps:',a.Reps())
a.addReps([{1,2},{0,2}])
print('reps:',a.Reps())
print('acts:',a.Acts())
a.addAct([0,1])
print('acts:',a.Acts())
a.addActs([[1,2],[0,2]])
print('reps:',a.Reps())
print('acts:',a.Acts())
a.setNeedById(3)
a.rep2act(4,2)
a.act2rep(2,3)
print(a.shortestPaths(4,3, draw=True))
# a.draw()
plt.show()

# con = sqlite3.connect('example.db')
# con = sqlite3.connect('rep_graph.db')
# con = sqlite3.connect('act_graph.db')
# con = sqlite3.connect('csa_graph.db')