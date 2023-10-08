import networkx as nx
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
import random
import numpy as np
import queue

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

    def _distribute_nodes(self):
        max_z = max(data['z'] for _, data in self.graph.nodes(data=True))
        nodes_by_depth = {z: [] for z in range(max_z + 1)}

        for node, data in self.graph.nodes(data=True):
            nodes_by_depth[data['z']].append(node)

        for z in reversed(range(max_z + 1)):
            nodes_at_depth = nodes_by_depth[z]
            centers = [node for node in nodes_at_depth if
                       self.graph.out_degree(node) == 0 and self.graph.in_degree(node) > 0]

            # Distribute center nodes randomly
            for center in centers:
                while True:
                    x = random.uniform(-50, 50)
                    y = random.uniform(-50, 50)
                    if all(math.sqrt((x - data['x']) ** 2 + (y - data['y']) ** 2) > 1
                           for other_node, data in self.graph.nodes(data=True) if
                           data['z'] == z and other_node != center):
                        self.graph.nodes[center]['x'] = x
                        self.graph.nodes[center]['y'] = y
                        break

            # BFS for distributing other nodes
            for center in centers:
                q = queue.Queue()
                q.put((center, 0))  # (node, distance)
                visited = set([center])

                while not q.empty():
                    current_node, dist = q.get()
                    x_current, y_current = self.graph.nodes[current_node]['x'], self.graph.nodes[current_node]['y']

                    for pred in self.graph.predecessors(current_node):
                        if self.graph.edges[pred, current_node]['type'] == "2" and pred not in visited:
                            # Calculate the average position of successors
                            x_avg = x_current
                            y_avg = y_current
                            count = 1
                            for succ in self.graph.successors(pred):
                                if self.graph.edges[pred, succ]['type'] == "2":
                                    x_avg += self.graph.nodes[succ]['x']
                                    y_avg += self.graph.nodes[succ]['y']
                                    count += 1
                            x_avg /= count
                            y_avg /= count

                            # Find a position in a circle around the average position
                            while True:
                                angle = random.uniform(0, 2 * math.pi)  # Random angle
                                x_new = x_avg + (dist + 1) * 5 * math.cos(angle)
                                y_new = y_avg + (dist + 1) * 5 * math.sin(angle)

                                # Check if the new position is valid
                                if all(math.sqrt((x_new - data['x']) ** 2 + (y_new - data['y']) ** 2) > 1
                                       for other_node, data in self.graph.nodes(data=True) if
                                       data['z'] == z and other_node != pred):
                                    self.graph.nodes[pred]['x'] = x_new
                                    self.graph.nodes[pred]['y'] = y_new
                                    visited.add(pred)
                                    q.put((pred, dist + 1))
                                    break

            # Distribute nodes that are not connected by connection2
            for node in nodes_at_depth:
                if node not in visited:
                    while True:
                        x = random.uniform(-50, 50)
                        y = random.uniform(-50, 50)
                        if all(math.sqrt((x - data['x']) ** 2 + (y - data['y']) ** 2) > 1
                               for other_node, data in self.graph.nodes(data=True) if
                               data['z'] == z and other_node != node):
                            self.graph.nodes[node]['x'] = x
                            self.graph.nodes[node]['y'] = y
                            break

    def calculate_coordinates(self):
        z_coordinates = {}
        for node in self.graph.nodes():
            self._compute_z_coordinates_recursive(node, z_coordinates)
        print('ok')
        # Call the _distribute_nodes method here to calculate x, y coordinates
        self._distribute_nodes()

    def visualize_graph(self):
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
                cone_x.append(x_end)
                cone_y.append(y_end)
                cone_z.append(z_end)
                # Compute the vector from start to end
                u = x_end - x_start
                v = y_end - y_start
                w = z_end - z_start
                # Normalize the vector
                length = (u ** 2 + v ** 2 + w ** 2) ** 0.5
                u /= length
                v /= length
                w /= length
                # Set the cone direction to be a vector of length `cone_length` in the direction of (u, v, w)
                cone_u.append(u * 25)
                cone_v.append(v * 25)
                cone_w.append(w * 25)

        edge_trace_conn1 = go.Scatter3d(x=edge_x_conn1, y=edge_y_conn1, z=edge_z_conn1,
                                        line=dict(width=2, color='grey'), mode='lines', opacity=0.5)

        edge_trace_conn2 = go.Scatter3d(x=edge_x_conn2, y=edge_y_conn2, z=edge_z_conn2,
                                        line=dict(width=2, color='#32a850'), mode='lines')
        cone_trace = go.Cone(x=cone_x, y=cone_y, z=cone_z, u=cone_u, v=cone_v, w=cone_w,
                             sizemode="scaled", sizeref=1, anchor="tip", colorscale=[[0, '#32a850'], [1, '#32a850']])

        node_x = []
        node_y = []
        node_z = []
        for id_, (x, y, z) in coords.items():
            node_x.append(x)
            node_y.append(y)
            node_z.append(z)

        node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
                                  marker=dict(size=10, color='blue', colorscale='Viridis', opacity=0.8),
                                  text=list(labels.values()))

        fig = go.Figure(data=[edge_trace_conn1, edge_trace_conn2, cone_trace, node_trace],
                        layout=go.Layout(scene=dict(aspectmode="cube"),
                                         margin=dict(t=0, b=0, l=0, r=0),
                                         ))

        fig.update_layout(scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ))

        fig.update_layout(title="3D Visualization of the Graph")
        fig.show()


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
graph.calculate_coordinates()
print('hi3')
# Visualizing the graph with computed coordinates
graph.visualize_graph()
