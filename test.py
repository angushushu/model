import graphs as g
import units as u
import sqlite3
from units import Rep, Act

import networkx as nx
import matplotlib.pyplot as plt

a = g.RSAGraph(rep_units={'r1','r2','r3'},act_units={'a1','a2'})
a.addRep({0,1})
a.addReps([{1,2},{0,2}])
a.addAct([0,1])
# a.addActs([[1,2],[0,2]])
a.setNeedById(3)
print(a.reps)
# print(a.getBaseById(1))
# a.rep2rep({'r5','r6'},{'r1'})
a.show()

# con = sqlite3.connect('example.db')
# con = sqlite3.connect('rep_graph.db')
# con = sqlite3.connect('act_graph.db')
# con = sqlite3.connect('csa_graph.db')