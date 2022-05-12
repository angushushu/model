import ConstSAGraph as cg
import sqlite3

# class Test:
#     def __init__(self, name):
#         self.name = name
#     def __str__(self):
#         return self.name

# node = Test('hi')
# a = cg.ConstSAGaph(node)
# a.add_node('hey',node)
# a.show()

con = sqlite3.connect('example.db')
con = sqlite3.connect('rep_graph.db')
con = sqlite3.connect('act_graph.db')
con = sqlite3.connect('csa_graph.db')