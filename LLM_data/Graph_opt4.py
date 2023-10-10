import graph_tool.all as gt
import plotly.graph_objects as go
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt

# USING GRAPH-TOOL INSTEAD OF NETWORKX

class Graph:
    def __init__(self):
        self.graph = gt.Graph()
        
        # Define vertex properties
        self.graph.vp["label"] = self.graph.new_vertex_property("string")
        self.graph.vp["value"] = self.graph.new_vertex_property("string") 
        self.graph.vp["x"] = self.graph.new_vertex_property("float")
        self.graph.vp["y"] = self.graph.new_vertex_property("float")
        self.graph.vp["z"] = self.graph.new_vertex_property("int")
        
        # Define edge properties
        self.graph.ep["type"] = self.graph.new_edge_property("string")
        self.graph.ep["label"] = self.graph.new_edge_property("string")
        self.graph.ep["weight"] = self.graph.new_edge_property("int")
        self.id_generator = self._id_generator()  
        self.value_to_label_map = {}
        self.label_to_id_map = {}

    def _id_generator(self):
        i = 1
        while True:
            yield i
            i += 1
            
    def add_rep(self, label, value=None):
        if not isinstance(label, str):
            raise ValueError("Label must be a string.")

        # Convert value to a string for hashing
        value_key = str(value)

        # Use a dictionary to map values to labels for O(1) lookup time
        if value is not None and value_key in self.value_to_label_map:
            return self.value_to_label_map[value_key]
        
        rep_id = next(self.id_generator)
        node = self.graph.add_vertex()
        self.graph.vp['label'][node] = label
        self.graph.vp['value'][node] = value
        
        self.value_to_label_map[str(value)] = label
        self.label_to_id_map[label] = rep_id
        
        return label

    def add_edge(self, start_label, end_label, connection_type="1", label="", weight=1):
        start_id = self.label_to_id_map[start_label] 
        end_id = self.label_to_id_map[end_label]
        
        edge = self.graph.add_edge(start_id, end_id)
        self.graph.ep['type'][edge] = connection_type
        self.graph.ep['label'][edge] = label
        self.graph.ep['weight'][edge] = weight

    def get_value(self, label):
        node_id = [v for v in self.graph.vertices() if self.graph.vp['label'][v] == label][0]
        return self.graph.vp['value'][node_id]

    def _get_leaf_nodes(self):
        leaf_nodes = []
        
        for v in self.graph.vertices():
            if self.graph.get_in_degrees([v])[0] == 0:
                leaf_nodes.append(v)
                
        return leaf_nodes
    
    def _compute_z_coordinates_recursive(self, node, z_coordinates):
        if node in z_coordinates:
            return z_coordinates[node]

        if self.graph.get_in_degrees([node])[0] == 0:
            z_coordinates[node] = 0
            self.graph.vp['z'][node] = 0
            return 0

        pred_nodes = [edge[0] for edge in self.graph.get_in_edges(node)]
        pred_z_coords = [self._compute_z_coordinates_recursive(pred, z_coordinates) 
                        for pred in pred_nodes if 
                        self.graph.ep['type'][self.graph.edge(pred, node)] == "1"]

        current_z = max(pred_z_coords) + 1 if pred_z_coords else 0
        z_coordinates[node] = current_z
        self.graph.vp['z'][node] = current_z

        return current_z

    def _compute_layout(self, subgraph, layout):
        if layout == 'shell':
            return gt.sfdp_layout(subgraph)
        elif layout == 'spring':
            return gt.arf_layout(subgraph)
        elif layout == 'fruchterman_reingold':
            return gt.fruchterman_reingold_layout(subgraph)
        elif layout == 'spectral':
            return gt.spectral_layout(subgraph)
        else:
            raise ValueError("Invalid layout type...")

    def _assign_random_coordinates(self, node):
        max_attempts = 100  # Maximum number of attempts to find a non-overlapping position
        for _ in range(max_attempts):
            x = random.uniform(-50, 50)
            y = random.uniform(-50, 50)
            
            is_overlapping = False
            for v in self.graph.vertices():
                if v == node:
                    continue
                    
                if abs(x - self.graph.vp['x'][v]) < 0.5 and abs(y - self.graph.vp['y'][v]) < 0.5:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                break
        else:  # This block runs if the for loop completes without a break
            print(f"Warning: Couldn't find a non-overlapping position for vertex {node} after {max_attempts} attempts. Assigning the last generated coordinates.")
            
        self.graph.vp['x'][node] = x
        self.graph.vp['y'][node] = y

    def _compute_xy_coordinates(self, layout):

        z_values = np.array([self.graph.vp['z'][v] for v in self.graph.vertices()])
        unique_zs = np.unique(z_values)

        for z in unique_zs:
            z_vertices = [v for v in self.graph.vertices() if self.graph.vp['z'][v] == z]
            v_filt = self.graph.new_vertex_property("bool")
            v_filt.a = False  # set all vertices to False
            for v in z_vertices:
                v_filt[v] = True  # set selected vertices to True

            # Create a subgraph using vertex filter
            z_subgraph = gt.GraphView(self.graph, vfilt=v_filt)

            # Remove type 1 edges
            edges_to_remove = []
            for e in z_subgraph.edges():
                if self.graph.ep['type'][e] == "1":
                    edges_to_remove.append(e)
            for e in edges_to_remove:
                z_subgraph.remove_edge(e)

            if z_subgraph.num_edges() > 0:
                pos = self._compute_layout(z_subgraph, layout)
                for v in z_subgraph.vertices():
                    self.graph.vp['x'][v] = pos[v][0]
                    self.graph.vp['y'][v] = pos[v][1]

        for z in reversed(unique_zs):
            z_vertices = [v for v in self.graph.vertices() if self.graph.vp['z'][v] == z]
            for v in z_vertices:
                if self.graph.get_in_degrees([v])[0] > 0:
                    pred_nodes = self.graph.get_in_edges(v)[0]
                    pred_pos = np.vstack([self.graph.vp['x'][p] for p in pred_nodes])
                    x = np.mean(pred_pos)
                    pred_pos = np.vstack([self.graph.vp['y'][p] for p in pred_nodes])
                    y = np.mean(pred_pos)
                    self.graph.vp['x'][v] = x
                    self.graph.vp['y'][v] = y
                else:
                    self._assign_random_coordinates(v)
    
    def calculate_coordinates(self, layout='sfdp'):    
        z_map = {}
        for v in self.graph.vertices():
            z_map[v] = self._compute_z_coordinates_recursive(v, z_map)

        self._compute_xy_coordinates(layout)
    
    def _compute_color(self, r, g, b):
        return f"rgb({int(r)},{int(g)},{int(b)})"
    
    def visualize_graph(self):
        arrow_length = 1
        max_in_degree_conn2 = max((sum(1 for e in self.graph.get_in_edges(v) if self.graph.ep['type'][e] == "2") for v in self.graph.vertices()), default=1)
        max_in_degree_conn1 = max((sum(1 for e in self.graph.get_in_edges(v) if self.graph.ep['type'][e] == "1") for v in self.graph.vertices()), default=1)

        edge_x_conn1 = []
        edge_y_conn1 = []
        edge_z_conn1 = []
        edge_x_conn2 = []
        edge_y_conn2 = []
        edge_z_conn2 = []
        arrow_x = []
        arrow_y = []
        arrow_z = []
        loop_x = []
        loop_y = []
        loop_z = []

        for e in self.graph.edges():
            start, end = e
            x_start, y_start, z_start = self.graph.vp['x'][start], self.graph.vp['y'][start], self.graph.vp['z'][start]
            x_end, y_end, z_end = self.graph.vp['x'][end], self.graph.vp['y'][end], self.graph.vp['z'][end]

            if self.graph.ep['type'][e] == "1":
                edge_x_conn1.extend([x_start, x_end, None])
                edge_y_conn1.extend([y_start, y_end, None])
                edge_z_conn1.extend([z_start, z_end, None])
            elif self.graph.ep['type'][e] == "2":
                edge_x_conn2.extend([x_start, x_end, None])
                edge_y_conn2.extend([y_start, y_end, None])
                edge_z_conn2.extend([z_start, z_end, None])
                if start == end:  # Detect loops
                    loop_x.append(x_end)
                    loop_y.append(y_end)
                    loop_z.append(z_end)
                else:
                    edge_length = ((x_end - x_start)**2 + (y_end - y_start)**2 + (z_end - z_start)**2) ** 0.5
                    if arrow_length > edge_length:
                        arrow_x_start = x_end - 0.5 * (x_end - x_start)
                        arrow_y_start = y_end - 0.5 * (y_end - y_start)
                        arrow_z_start = z_end - 0.5 * (z_end - z_start)
                    else:
                        arrow_x_start = x_end - (arrow_length / edge_length) * (x_end - x_start)
                        arrow_y_start = y_end - (arrow_length / edge_length) * (y_end - y_start)
                        arrow_z_start = z_end - (arrow_length / edge_length) * (z_end - z_start)

                    arrow_x.extend([arrow_x_start, x_end, None])
                    arrow_y.extend([arrow_y_start, y_end, None])
                    arrow_z.extend([arrow_z_start, z_end, None])

        node_x = []
        node_y = []
        node_z = []
        raw_colors = []
        labels = []

        for v in self.graph.vertices():
            node_x.append(self.graph.vp['x'][v])
            node_y.append(self.graph.vp['y'][v])
            node_z.append(self.graph.vp['z'][v])
            labels.append(self.graph.vp['label'][v])

            in_degree_conn1 = np.sum([1 for e in self.graph.get_in_edges(v) if self.graph.ep['type'][e] == "1"])
            in_degree_conn2 = np.sum([1 for e in self.graph.get_in_edges(v) if self.graph.ep['type'][e] == "2"])

            if in_degree_conn2 > 0:
                r = 255 * (in_degree_conn2 / max_in_degree_conn2)
                g = 255 * (1 - in_degree_conn2 / max_in_degree_conn2)
                b = 200 * (1 - in_degree_conn2 / max_in_degree_conn2)
            else:
                r = 255 * (in_degree_conn1 / max_in_degree_conn1)
                g = 255 * (in_degree_conn1 / max_in_degree_conn1)
                b = 200 * (1 - in_degree_conn1 / max_in_degree_conn1)

            raw_colors.append((r, g, b))

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
                                  name='Loops')

        node_trace = go.Scatter3d(x=node_x, y=node_y, z=node_z, mode='markers',
                                  marker=dict(size=3, color=node_color, opacity=0.8),
                                  text=labels, name='Nodes')

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
    
    def save_graph(self, filename='saved_graph.gt'):
        gt.save_graph(self.graph, filename)

    def load_graph(self, filename='saved_graph.gt'):
        self.graph = gt.load_graph(filename)