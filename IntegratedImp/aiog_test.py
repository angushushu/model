import aiog
import sys
import matplotlib.pyplot as plt
# from random import randrange
from random import randint

import networkx as nx
import matplotlib.pyplot as plt

sys.setrecursionlimit(10**6)

a = aiog.Graph({1,2,3,4,5,6,7,8,9,10}, {1,2})
rus = set(range(randint(0,10)))
aus = set(range(randint(0,10)))
print(rus)
print(aus)
# a = aiog.Graph(rus, aus)
# for i in range(randint(0,10)):
#     a.addRep(i, {randint(0, len(rus)),randint(0, len(rus))})
# lvl 1
a.addRep(11,{1,2})
a.addRep(12,{5,6})
a.addRep(13,{7,2})
a.addRep(14,{3,9})
a.addRep(15,{1,3})
a.addRep(16,{4,8})
a.addRep(17,{7,6})
a.addRep(18,{4,6})
# lvl 2
a.addRep(19,{12,9})
a.addRep(20,{13,11})
a.addRep(21,{17,16})
a.addRep(22,{13,18})
a.addRep(23,{14,15})
# acts
a.addAct(3,[1,2])
a.addAct(4,[2,1])
a.addRAR(rep1=19,act=1,rep2=21)
a.addRAR(rep1=20,act=1,rep2=21)
a.addRAR(rep1=21,act=1,rep2=22)
a.addRAR(rep1=22,act=3,rep2=23)

print(a.getRep(4))
a.draw()