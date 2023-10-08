import networkx as nx
import pickle
from Graph import Graph

graph = Graph()
graph.load_graph('saved_graph.pkl')
graph.visualize_graph()