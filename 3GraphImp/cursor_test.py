# Import necessary libraries
import networkx as nx
from bokeh.io import show
from bokeh.plotting import figure, from_networkx
from bokeh.models import Range1d

# Create a directed graph with string nodes
G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A')])

# Create a mapping from string nodes to integer nodes
node_mapping = {'A': 1, 'B': 2, 'C': 3}

# Relabel the nodes in the graph
G = nx.relabel_nodes(G, node_mapping)

# The rest of the code remains the same


# Create a Bokeh plot with the directed graph
plot = figure(title="Directed Graph with String Nodes", x_range=Range1d(-1.5, 1.5), y_range=Range1d(-1.5, 1.5))
graph = from_networkx(G, nx.spring_layout, scale=1, center=(0, 0))
plot.renderers.append(graph)

# Display the plot
# output_notebook()
show(plot)