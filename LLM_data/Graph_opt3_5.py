import networkx as nx
import plotly.graph_objects as go
import numpy as np
import pickle
from rtree import index
import concurrent.futures
import threading


# ADDED PARALLEL OPERATION

class Graph:
    def __init__(self):
        """
        Initialize a Graph object.
        """
        self.graph = nx.DiGraph()
        self.id_generator = self._id_generator()
        # 使用字典来存储 value 到 label 和 id 的映射
        self.value_to_label_map = {}
        self.label_to_id_map = {}

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
        # Validate parameters
        if not isinstance(label, str):
            raise ValueError("Label must be a string.")

        # Convert value to a string for hashing
        value_key = str(value)

        # Use a dictionary to map values to labels for O(1) lookup time
        if value is not None and value_key in self.value_to_label_map:
            return self.value_to_label_map[value_key]

        # Generate unique ID
        rep_id = next(self.id_generator)

        # If value is not provided, use the generated ID
        value = value if value is not None else rep_id

        # Add the node to the graph
        self.graph.add_node(rep_id, label=label, x=0, y=0, z=0, value=value)

        # Update the value-to-label map and label-to-id map
        self.value_to_label_map[value_key] = label
        self.label_to_id_map[label] = rep_id

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
        # Validate parameters
        if not isinstance(start_label, str) or not isinstance(end_label, str):
            raise ValueError("Start and end labels must be strings.")
        if connection_type not in ["1", "2"]:
            raise ValueError("Connection type must be '1' or '2'.")
        if not isinstance(label, str):
            raise ValueError("Label must be a string.")
        if not isinstance(weight, int):
            raise ValueError("Weight must be an integer.")

        # Use a dictionary to map labels to node IDs for O(1) lookup time
        # Assuming self.label_to_id_map is a dictionary attribute of the Graph class
        start_id = self.label_to_id_map.get(start_label)
        end_id = self.label_to_id_map.get(end_label)

        if start_id is None:
            raise ValueError(f"No node with label {start_label} found.")
        if end_id is None:
            raise ValueError(f"No node with label {end_label} found.")

        # Add the edge to the graph
        self.graph.add_edge(start_id, end_id, type=connection_type, label=label, weight=weight)

    def get_value(self, label):
        """
        Retrieve the 'value' attribute of a node given its label.

        Parameters:
            label (str): The label of the node.

        Returns:
            The 'value' attribute of the node or None if not found.
        """
        # Validate parameters
        if not isinstance(label, str):
            raise ValueError("Label must be a string.")

        # Use NumPy to get the node ID
        node_id = np.array(self.node_info['id'])[np.where(np.array(self.node_info['label']) == label)[0]]

        # If node ID is found, return its 'value' attribute, otherwise return None
        return self.graph.nodes[node_id[0]]['value'] if node_id.size > 0 else None

    def _get_leaf_nodes(self):
        """
        Identify and return the leaf nodes in the graph.

        Returns:
            list: List of leaf nodes.
        """
        return [node for node, in_degree in self.graph.in_degree() if in_degree == 0]

    def _compute_z_coordinates_recursive(self, node, z_coordinates):
        if node in z_coordinates:
            return z_coordinates[node]
        if self.graph.in_degree(node) == 0:
            z_coordinates[node] = 0
            self.graph.nodes[node]['z'] = 0
            return 0

        # 使用并行计算来获取前驱节点的z坐标
        with concurrent.futures.ThreadPoolExecutor() as executor:
            predecessor_z_coords = np.array(list(executor.map(
                lambda pred: self._compute_z_coordinates_recursive(pred, z_coordinates),
                [predecessor for predecessor in self.graph.predecessors(node)
                 if self.graph.edges[predecessor, node]['type'] == "1"]
            )))

        # 使用 NumPy 的 max 函数来计算最大值
        current_z = np.max(predecessor_z_coords) + 1 if predecessor_z_coords.size > 0 else 0
        z_coordinates[node] = current_z
        self.graph.nodes[node]['z'] = current_z

        return current_z

    def _compute_layout(self, subgraph, layout):
        """
        Compute the layout of a subgraph using the specified layout algorithm.
        """
        if layout == 'shell':
            return nx.shell_layout(subgraph)  # shell_layout without considering edge weights
        elif layout == 'spring':
            return nx.spring_layout(subgraph, seed=42)  # seed for reproducibility
        elif layout == 'kamada_kawai':
            return nx.kamada_kawai_layout(subgraph)
        elif layout == 'fruchterman_reingold':
            return nx.spring_layout(subgraph)
        elif layout == 'spectral':
            return nx.spectral_layout(subgraph)
        elif layout == 'planar':
            return nx.planar_layout(subgraph)
        elif layout == 'random':
            return nx.random_layout(subgraph)
        elif layout == 'circular':
            return nx.circular_layout(subgraph)
        elif layout == 'spiral':
            return nx.spiral_layout(subgraph)
        # elif layout == 'graphviz':
        #     return nx.nx_agraph.graphviz_layout(subgraph)
        elif layout == 'pydot':
            return nx.nx_pydot.pydot_layout(subgraph)
        else:
            raise ValueError(
                "Invalid layout type. Choose from ['shell', 'spring', 'kamada_kawai', 'fruchterman_reingold', 'spectral', 'planar']")

    def _assign_random_coordinates(self, node, rtree_idx, lock):
        """
        Assign random coordinates to a node while avoiding overlap with existing nodes.
        """
        while True:
            x = np.random.uniform(-50, 50)
            y = np.random.uniform(-50, 50)
            bbox = (x - 0.1, y - 0.1, x + 0.1, y + 0.1)
            with lock:
                if not any(True for _ in rtree_idx.intersection(bbox)):
                    self.graph.nodes[node]['x'] = x
                    self.graph.nodes[node]['y'] = y
                    rtree_idx.insert(node, (x, y, x, y))
                    break

    def _compute_xy_coordinates(self, layout):
        node_data = np.array([(id_, data['x'], data['y'], data['z'])
                              for id_, data in self.graph.nodes(data=True)],
                             dtype=[('id', int), ('x', float), ('y', float), ('z', int)])
        max_z = np.max(node_data['z'])

        for z in np.unique(node_data['z']):
            highest_z_nodes = node_data[node_data['z'] == z]['id']
            subgraph = self.graph.subgraph(highest_z_nodes).copy()
            edges_to_remove = [(u, v) for u, v, d in subgraph.edges(data=True) if d['type'] != "2"]
            subgraph.remove_edges_from(edges_to_remove)

            if nx.number_of_edges(subgraph) > 0:
                layout_pos = self._compute_layout(subgraph, layout)
                for node, pos in layout_pos.items():
                    x, y = pos[0] * 50, pos[1] * 50
                    node_data['x'][node_data['id'] == node] = x
                    node_data['y'][node_data['id'] == node] = y

        rtree_idx = index.Index()
        lock = threading.Lock()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            for z in reversed(range(max_z)):
                for node in node_data[node_data['z'] == z]['id']:
                    successors = [succ for succ in self.graph.successors(node) if
                                  self.graph.edges[node, succ]['type'] == "1"]
                    if successors:
                        succ_coords = node_data[np.isin(node_data['id'], successors)][['x', 'y']]
                        mean_x = np.mean(succ_coords['x'])
                        mean_y = np.mean(succ_coords['y'])
                        node_data['x'][node_data['id'] == node] = mean_x
                        node_data['y'][node_data['id'] == node] = mean_y
                    else:
                        executor.submit(self._assign_random_coordinates, node, rtree_idx, lock)

        for node_id, x, y, _ in node_data:
            self.graph.nodes[node_id]['x'] = x
            self.graph.nodes[node_id]['y'] = y

    def calculate_coordinates(self, xy_layout='spring'):
        z_coordinates = {}

        # 避免重复计算：只计算一次每个节点的 z 坐标
        for node in self.graph.nodes():
            if node not in z_coordinates:
                self._compute_z_coordinates_recursive(node, z_coordinates)

        self._compute_xy_coordinates(xy_layout)

    def _compute_color(self, r, g, b):
        return f"rgb({int(r)},{int(g)},{int(b)})"

    def visualize_graph(self):
        arrow_length = 1
        labels = nx.get_node_attributes(self.graph, 'label')

        edge_data = np.array([(start_id, end_id,
                               self.graph.nodes[start_id]['x'],
                               self.graph.nodes[start_id]['y'],
                               self.graph.nodes[start_id]['z'],
                               self.graph.nodes[end_id]['x'],
                               self.graph.nodes[end_id]['y'],
                               self.graph.nodes[end_id]['z'],
                               data['type'])
                             for start_id, end_id, data in self.graph.edges(data=True)],
                             dtype=[('start_id', int), ('end_id', int),
                                    ('x_start', float), ('y_start', float), ('z_start', float),
                                    ('x_end', float), ('y_end', float), ('z_end', float),
                                    ('edge_type', 'U1')])

        # Extract relevant data into ndarrays for vectorized operations
        start_coords = np.array(edge_data[['x_start', 'y_start', 'z_start']].tolist())
        end_coords = np.array(edge_data[['x_end', 'y_end', 'z_end']].tolist())
        # Compute direction vectors
        direction_vectors = end_coords - start_coords
        edge_lengths = np.linalg.norm(direction_vectors.view(float).reshape(len(edge_data), -1), axis=1)
        edge_lengths[edge_lengths == 0] = 1
        unit_direction_vectors = direction_vectors.view(float).reshape(len(edge_data), -1) / edge_lengths[:, None]
        # Extract relevant data into ndarrays for vectorized operations
        end_coords = np.array(edge_data[['x_end', 'y_end', 'z_end']].tolist())
        # Ensure arrow_length does not exceed half the edge_length
        arrow_length = np.where(arrow_length > edge_lengths / 2, edge_lengths / 2, arrow_length)
        # Compute arrow start points
        arrow_start_points = end_coords - arrow_length[:, None] * unit_direction_vectors

        is_conn1 = edge_data['edge_type'] == '1'
        is_conn2 = edge_data['edge_type'] == '2'
        is_loop = (edge_data['start_id'] == edge_data['end_id']) & is_conn2

        edge_x_conn1 = np.vstack((edge_data['x_start'][is_conn1], edge_data['x_end'][is_conn1], np.full(np.sum(is_conn1), np.nan))).T.flatten()
        edge_y_conn1 = np.vstack((edge_data['y_start'][is_conn1], edge_data['y_end'][is_conn1], np.full(np.sum(is_conn1), np.nan))).T.flatten()
        edge_z_conn1 = np.vstack((edge_data['z_start'][is_conn1], edge_data['z_end'][is_conn1], np.full(np.sum(is_conn1), np.nan))).T.flatten()

        edge_x_conn2 = np.vstack((edge_data['x_start'][is_conn2 & ~is_loop], edge_data['x_end'][is_conn2 & ~is_loop], np.full(np.sum(is_conn2 & ~is_loop), np.nan))).T.flatten()
        edge_y_conn2 = np.vstack((edge_data['y_start'][is_conn2 & ~is_loop], edge_data['y_end'][is_conn2 & ~is_loop], np.full(np.sum(is_conn2 & ~is_loop), np.nan))).T.flatten()
        edge_z_conn2 = np.vstack((edge_data['z_start'][is_conn2 & ~is_loop], edge_data['z_end'][is_conn2 & ~is_loop], np.full(np.sum(is_conn2 & ~is_loop), np.nan))).T.flatten()

        arrow_x = np.vstack((arrow_start_points[is_conn2 & ~is_loop, 0], edge_data['x_end'][is_conn2 & ~is_loop], np.full(np.sum(is_conn2 & ~is_loop), np.nan))).T.flatten()
        arrow_y = np.vstack((arrow_start_points[is_conn2 & ~is_loop, 1], edge_data['y_end'][is_conn2 & ~is_loop], np.full(np.sum(is_conn2 & ~is_loop), np.nan))).T.flatten()
        arrow_z = np.vstack((arrow_start_points[is_conn2 & ~is_loop, 2], edge_data['z_end'][is_conn2 & ~is_loop], np.full(np.sum(is_conn2 & ~is_loop), np.nan))).T.flatten()

        loop_x = edge_data['x_end'][is_loop]
        loop_y = edge_data['y_end'][is_loop]
        loop_z = edge_data['z_end'][is_loop]

        max_in_degree_conn2 = max((sum(1 for _, _, d in self.graph.in_edges(node, data=True) if d['type'] == "2")
                                for node in self.graph.nodes()), default=1)

        max_in_degree_conn1 = max((sum(1 for _, _, d in self.graph.in_edges(node, data=True) if d['type'] == "1")
                                for node in self.graph.nodes()), default=1)
        # Parallel computation of node colors
        with concurrent.futures.ThreadPoolExecutor() as executor:
            node_results = list(executor.map(
                lambda node_data: (
                    node_data[1]['x'],
                    node_data[1]['y'],
                    node_data[1]['z'],
                    (
                        255 * (sum(1 for _, _, d in self.graph.in_edges(node_data[0], data=True) if
                                   d['type'] == "2") / max_in_degree_conn2),
                        255 * (1 - sum(1 for _, _, d in self.graph.in_edges(node_data[0], data=True) if
                                       d['type'] == "2") / max_in_degree_conn2),
                        200 * (1 - sum(1 for _, _, d in self.graph.in_edges(node_data[0], data=True) if
                                       d['type'] == "2") / max_in_degree_conn2)
                    ) if sum(
                        1 for _, _, d in self.graph.in_edges(node_data[0], data=True) if d['type'] == "2") > 0 else (
                        255 * (sum(1 for _, _, d in self.graph.in_edges(node_data[0], data=True) if
                                   d['type'] == "1") / max_in_degree_conn1),
                        255 * (sum(1 for _, _, d in self.graph.in_edges(node_data[0], data=True) if
                                   d['type'] == "1") / max_in_degree_conn1),
                        200 * (1 - sum(1 for _, _, d in self.graph.in_edges(node_data[0], data=True) if
                                       d['type'] == "1") / max_in_degree_conn1)
                    )
                ),
                self.graph.nodes(data=True)
            ))

        # Extract node coordinates and colors
        node_x, node_y, node_z, raw_colors = zip(*node_results)
        node_color = [self._compute_color(r, g, b) for r, g, b in raw_colors]

        edge_trace_conn1 = go.Scatter3d(x=edge_x_conn1, y=edge_y_conn1, z=edge_z_conn1,
                                        line=dict(width=2, color='#5f8c94'), mode='lines', opacity=0.1,
                                        name='Connection 1')

        edge_trace_conn2 = go.Scatter3d(x=edge_x_conn2, y=edge_y_conn2, z=edge_z_conn2,
                                        line=dict(width=2, color='#E225F9'), mode='lines', opacity=0.3,
                                        name='Connection 2')

        arrow_trace = go.Scatter3d(x=arrow_x, y=arrow_y, z=arrow_z,
                                line=dict(width=2.5, color='#FF013E'), mode='lines', opacity=0.6,
                                name='Arrows')

        loop_trace = go.Scatter3d(x=loop_x, y=loop_y, z=loop_z, mode='markers',
                                marker=dict(size=10, color='#E225F9', opacity=0.5),
                                name='Loops')  # 添加name属性

        node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
                                  marker=dict(size=3, color=node_color, opacity=0.8),
                                  text=list(labels.values()), name='Nodes')  # 添加name属性

        fig = go.Figure(data=[edge_trace_conn1, edge_trace_conn2, arrow_trace, loop_trace, node_trace],
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
