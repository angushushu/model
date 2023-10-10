from Graph_opt4 import Graph

# Example usage focusing on connection1
graph = Graph()

# z = 1
graph.add_rep("Rep1")
graph.add_rep("Rep2")
graph.add_rep("Rep3")
# z = 2
graph.add_rep("Rep4")
graph.add_rep("Rep5")
graph.add_rep("Rep6")
graph.add_rep("Rep9")
# z = 3
graph.add_rep("Rep7")
graph.add_rep("Rep8")

graph.add_edge("Rep1", "Rep4", "1", "Edge1-4")
graph.add_edge("Rep2", "Rep4", "1", "Edge2-4")
graph.add_edge("Rep1", "Rep5", "1", "Edge1-5")
graph.add_edge("Rep3", "Rep5", "1", "Edge5-5")
graph.add_edge("Rep2", "Rep6", "1", "Edge2-6")
graph.add_edge("Rep3", "Rep6", "1", "Edge3-6")

graph.add_edge("Rep4", "Rep7", "1", "Edge4-7")
graph.add_edge("Rep5", "Rep7", "1", "Edge5-7")
graph.add_edge("Rep4", "Rep8", "1", "Edge4-8")
graph.add_edge("Rep4", "Rep8", "1", "Edge4-8")
graph.add_edge("Rep1", "Rep9", "1", "Edge1-9")
graph.add_edge("Rep2", "Rep9", "1", "Edge2-9")
graph.add_edge("Rep3", "Rep9", "1", "Edge3-9")

graph.add_edge("Rep5", "Rep8", "2", "Edge5-8")
graph.add_edge("Rep7", "Rep8", "2", "Edge7-8")
graph.add_edge("Rep4", "Rep5", "2", "Edge4-5")
graph.add_edge("Rep6", "Rep5", "2", "Edge6-5")

#spring, shell, kamada_kawai, fruchterman_reingold, spectral, planar
graph.calculate_coordinates('spring')
# Visualizing the graph with computed coordinates
graph.visualize_graph()
