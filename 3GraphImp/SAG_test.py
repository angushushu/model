import networkx as nx
from RepGraph import RepGraph
from ActGraph import ActGraph
from SAGraph import SAGraph
import matplotlib.pyplot as plt
import random

import networkx as nx
import matplotlib.pyplot as plt
import random

sag = SAGraph({'r_1'})
sag.add_s('r_3')
sag.add_ss('r_2','r_3')
sag.draw()
plt.show()