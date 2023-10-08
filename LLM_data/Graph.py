import networkx as nx
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
import random
import numpy as np
import queue
import pickle
import scipy

class Graph:
    def __init__(self):
        """
        Initialize a Graph object.
        """
        self.graph = nx.DiGraph()
        self.id_generator = self._id_generator()

    def _id_generator(self):
        """
        Generator function to create unique IDs for nodes.
        """
        i = 1
        while True:
            yield i
            i += 1

    def add_rep(self, label, value=None):
        """
        Add a representative node with a given label to the graph.

        Parameters:
            label (str): The label for the node.
            value: The value for the node. If a node with this value already exists,
                   the label of the existing node is returned.

        Returns:
            str: The label of the node with the given value.
        """
        # Check for existing node with the same value
        # print('adding', label, value)
        existing_labels = None
        if value:
            existing_labels = [data['label'] for node, data in self.graph.nodes(data=True) if data.get('value') == value]
            # print('existing labels:', existing_labels)
        if existing_labels:
            return existing_labels[0]
        rep_id = next(self.id_generator)
        if not value:
            value = rep_id
        self.graph.add_node(rep_id, label=label, x=0, y=0, z=0, value=value)
        return label

    def add_edge(self, start_label, end_label, connection_type="1", label="", weight=1):
        """
        Add an edge between nodes with given start and end labels.

        Parameters:
            start_label (str): Label of the start node.
            end_label (str): Label of the end node.
            connection_type (str): Type of the connection ("1" or "2"). Default is "1".
            label (str): Label for the edge. Default is an empty string.
            weight (int): Weight of the edge. Default is 1.
        """
        start_ids = [id_ for id_, data in self.graph.nodes(data=True) if data['label'] == start_label]
        end_ids = [id_ for id_, data in self.graph.nodes(data=True) if data['label'] == end_label]

        if not start_ids:
            raise ValueError(f"No node with label {start_label} found.")
        if not end_ids:
            raise ValueError(f"No node with label {end_label} found.")

        start_id = start_ids[0]
        end_id = end_ids[0]
        # print('adding', start_label, '->', end_label)
        # print('start_id', start_id)
        # print('end_id', end_id)
        self.graph.add_edge(start_id, end_id, type=connection_type, label=label, weight=weight)

    def get_value(self, label):
        """
        Retrieve the 'value' attribute of a node given its label.

        Parameters:
            label (str): The label of the node.

        Returns:
            The 'value' attribute of the node or None if not found.
        """
        for _, data in self.graph.nodes(data=True):
            if data['label'] == label:
                return data['value']
        return None

    def _get_leaf_nodes(self):
        """
        Identify and return the leaf nodes in the graph.

        Returns:
            list: List of leaf nodes.
        """
        return [node for node, in_degree in self.graph.in_degree() if in_degree == 0]

    def _compute_z_coordinates_recursive(self, node, z_coordinates):
        """
        Recursively compute the z-coordinate of a node.

        Parameters:
            node: The node for which the z-coordinate is being computed.
            z_coordinates (dict): A dictionary to store computed z-coordinates.

        Returns:
            int: The computed z-coordinate.
        """
        if node in z_coordinates:
            return z_coordinates[node]
        if self.graph.in_degree(node) == 0:
            z_coordinates[node] = 0
            self.graph.nodes[node]['z'] = 0
            return 0
        max_z = max(self._compute_z_coordinates_recursive(predecessor, z_coordinates)
                    for predecessor in self.graph.predecessors(node)
                    if self.graph.edges[predecessor, node]['type'] == "1")
        z_coordinates[node] = max_z + 2
        self.graph.nodes[node]['z'] = max_z + 2
        if(self.graph.nodes[node]['label']=='State3'):
            print(self.graph.nodes[node]['z'])
        return max_z + 2

    def _compute_xy_coordinates(self, layout):
        max_z = max(data['z'] for _, data in self.graph.nodes(data=True))
        nodes_by_depth = {z: [] for z in range(max_z + 1)}

        for node, data in self.graph.nodes(data=True):
            nodes_by_depth[data['z']].append(node)

        # Handling nodes with the highest z using force-directed graph drawing
        highest_z_nodes = nodes_by_depth[max_z]
        subgraph = self.graph.subgraph(highest_z_nodes).copy()
        subgraph.remove_edges_from([(u, v) for u, v, d in subgraph.edges(data=True) if d['type'] != "2"])

        if nx.number_of_edges(subgraph) > 0:
            if layout == 'shell':
                layout_pos = nx.shell_layout(subgraph)  # shell_layout without considering edge weights
            elif layout == 'spring':
                layout_pos = nx.spring_layout(subgraph, seed=42)  # seed for reproducibility
            elif layout == 'kamada_kawai':
                layout_pos = nx.kamada_kawai_layout(subgraph)
            elif layout == 'fruchterman_reingold':
                layout_pos = nx.spring_layout(subgraph)
            elif layout == 'spectral':
                layout_pos = nx.spectral_layout(subgraph)
            elif layout == 'planar':
                layout_pos = nx.planar_layout(subgraph)
            else:
                raise ValueError(
                    "Invalid layout type. Choose from ['shell', 'spring', 'kamada_kawai', 'fruchterman_reingold', 'spectral', 'planar']")
            for node, pos in layout_pos.items():
                print('drawing ...')
                self.graph.nodes[node]['x'] = pos[0] * 50  # scaling to avoid overlap, adjust if necessary
                self.graph.nodes[node]['y'] = pos[1] * 50  # scaling to avoid overlap, adjust if necessary
        else:
            for node in highest_z_nodes:
                print('drawing ...')
                self.graph.nodes[node]['x'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)
                self.graph.nodes[node]['y'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)
                while any(
                        math.sqrt((self.graph.nodes[node]['x'] - self.graph.nodes[other]['x']) ** 2 +
                                  (self.graph.nodes[node]['y'] - self.graph.nodes[other]['y']) ** 2) < 1
                        for other in highest_z_nodes if other != node
                ):
                    print('    replot...')
                    self.graph.nodes[node]['x'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)
                    self.graph.nodes[node]['y'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)

        # Handling nodes with lower z based on average position of successors and avoiding overlap
        for z in reversed(range(max_z)):
            for node in nodes_by_depth[z]:
                print('drawing ...')
                successors = [succ for succ in self.graph.successors(node) if
                              self.graph.edges[node, succ]['type'] == "1"]
                if successors:
                    avg_x = np.mean([self.graph.nodes[succ]['x'] for succ in successors])
                    avg_y = np.mean([self.graph.nodes[succ]['y'] for succ in successors])
                    self.graph.nodes[node]['x'] = avg_x
                    self.graph.nodes[node]['y'] = avg_y
                else:
                    self.graph.nodes[node]['x'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)
                    self.graph.nodes[node]['y'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)
                    while any(
                            math.sqrt((self.graph.nodes[node]['x'] - self.graph.nodes[other]['x']) ** 2 +
                                      (self.graph.nodes[node]['y'] - self.graph.nodes[other]['y']) ** 2) < 1
                            for other in nodes_by_depth[z] if other != node
                    ):
                        print('    replot...')
                        self.graph.nodes[node]['x'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)
                        self.graph.nodes[node]['y'] = np.random.uniform(-50, 50) + random.uniform(-1, 1)

    def calculate_coordinates(self, xy_layout='spring'):
        z_coordinates = {}
        for node in self.graph.nodes():
            self._compute_z_coordinates_recursive(node, z_coordinates)
        print('ok')
        self._compute_xy_coordinates(xy_layout)

    def visualize_graph(self):
        cone_size = 25
        labels = nx.get_node_attributes(self.graph, 'label')
        coords = {(id_): (data['x'], data['y'], data['z']) for id_, data in self.graph.nodes(data=True)}

        edge_x_conn1 = []
        edge_y_conn1 = []
        edge_z_conn1 = []
        edge_x_conn2 = []
        edge_y_conn2 = []
        edge_z_conn2 = []
        cone_x = []
        cone_y = []
        cone_z = []
        cone_u = []
        cone_v = []
        cone_w = []
        loop_x = []
        loop_y = []
        loop_z = []

        for (start_id, end_id, data) in self.graph.edges(data=True):
            x_start, y_start, z_start = coords[start_id]
            x_end, y_end, z_end = coords[end_id]

            if data['type'] == "1":
                edge_x_conn1.extend([x_start, x_end, None])
                edge_y_conn1.extend([y_start, y_end, None])
                edge_z_conn1.extend([z_start, z_end, None])
            elif data['type'] == "2":
                edge_x_conn2.extend([x_start, x_end, None])
                edge_y_conn2.extend([y_start, y_end, None])
                edge_z_conn2.extend([z_start, z_end, None])
                if start_id == end_id:  # Detect loops
                    loop_x.append(x_end)
                    loop_y.append(y_end)
                    loop_z.append(z_end)
                else:
                    u = x_end - x_start
                    v = y_end - y_start
                    w = z_end - z_start
                    length = (u ** 2 + v ** 2 + w ** 2) ** 0.5
                    if length > 0:  # Avoid division by zero
                        u /= length
                        v /= length
                        w /= length
                        cone_u.append(u * cone_size)
                        cone_v.append(v * cone_size)
                        cone_w.append(w * cone_size)
                        cone_x.append(x_end)
                        cone_y.append(y_end)
                        cone_z.append(z_end)

        edge_trace_conn1 = go.Scatter3d(x=edge_x_conn1, y=edge_y_conn1, z=edge_z_conn1,
                                        line=dict(width=2, color='grey'), mode='lines', opacity=0.1)

        edge_trace_conn2 = go.Scatter3d(x=edge_x_conn2, y=edge_y_conn2, z=edge_z_conn2,
                                        line=dict(width=2, color='#32a850'), mode='lines', opacity=0.5)

        cone_trace = go.Cone(x=cone_x, y=cone_y, z=cone_z, u=cone_u, v=cone_v, w=cone_w,
                             sizemode="absolute", sizeref=10, anchor="tip", colorscale=[[0, '#32a850'], [1, '#32a850']], opacity=0.5)

        loop_trace = go.Scatter3d(x=loop_x, y=loop_y, z=loop_z, mode='markers',
                                  marker=dict(size=20, color='red', opacity=0.5),
                                  name='Loops')

        node_x = []
        node_y = []
        node_z = []
        node_color = []

        max_in_degree_conn2 = max((data for node, data in self.graph.in_degree() if
                                   any(d['type'] == "2" for _, _, d in self.graph.in_edges(node, data=True))),
                                  default=1)
        max_in_degree_conn1 = max((data for node, data in self.graph.in_degree() if
                                   any(d['type'] == "1" for _, _, d in self.graph.in_edges(node, data=True))),
                                  default=1)

        for id_, (x, y, z) in coords.items():
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

            in_degree_conn1 = sum(1 for _, _, data in self.graph.in_edges(id_, data=True) if data['type'] == "1")
            in_degree_conn2 = sum(1 for _, _, data in self.graph.in_edges(id_, data=True) if data['type'] == "2")

            r = int((in_degree_conn2 / max_in_degree_conn2) * 255)
            g = 0
            b = int((in_degree_conn1 / max_in_degree_conn1) * 255)

            node_color.append(f'rgb({r},{g},{b})')

        node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
                                  marker=dict(size=2, color=node_color, opacity=0.8),
                                  text=list(labels.values()))

        fig = go.Figure(data=[edge_trace_conn1, edge_trace_conn2, cone_trace, loop_trace, node_trace],
                        layout=go.Layout(scene=dict(aspectmode="cube"),
                                         margin=dict(t=0, b=0, l=0, r=0),
                                         ))

        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))

        fig.update_layout(modebar_orientation="v")
        fig.update_layout(title="3D Visualization of the Graph")
        fig.show()

    def save_graph(self, filename='saved_graph.pkl'):
        with open(filename, 'wb') as f:
            pickle.dump(self.graph, f)

    def load_graph(self, filename='saved_graph.pkl'):
        with open(filename, 'rb') as f:
            self.graph = pickle.load(f)

# Example usage focusing on connection1
graph = Graph()
print('hi')

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

print('hi2')
#spring, shell, kamada_kawai, fruchterman_reingold, spectral, planar
graph.calculate_coordinates('kamada_kawai')
print('hi3')
# Visualizing the graph with computed coordinates
graph.visualize_graph()
