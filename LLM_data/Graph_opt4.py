import graph_tools as gt
import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import Image

class Graph:
    def __init__(self):
        self.graph = gt.Graph()
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
        node_id = list(self.graph.vertex(self.graph.vp['label'] == label))[0]
        return self.graph.vp['value'][node_id]

    def _get_leaf_nodes(self):
        leaf_nodes = []
        
        for v in self.graph.vertices():
            if self.graph.in_degree(v) == 0:
                leaf_nodes.append(v)
                
        return leaf_nodes
    
    def _compute_z_coordinates_recursive(self, node, z_coordinates):
        if node in z_coordinates:
            return z_coordinates[node]

        if self.graph.in_degree(node) == 0:
            z_coordinates[node] = 0
            self.graph.vp['z'][node] = 0
            return 0

        pred_nodes = self.graph.get_in_edges(node)[0] 
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
        elif layout == 'kamada_kawai':
            return gt.kk_layout(subgraph)
        elif layout == 'fruchterman_reingold':
            return gt.fruchterman_reingold_layout(subgraph)
        elif layout == 'spectral':
            return gt.spectral_layout(subgraph)
        else:
            raise ValueError("Invalid layout type...")

    def _assign_random_coordinates(self, node):
        while True:
            x = random.uniform(-50, 50)
            y = random.uniform(-50, 50)
            
            is_overlapping = False
            for v in self.graph.vertices():
                if v == node:
                    continue
                    
                if abs(x - self.graph.vp['x'][v]) < 2 and abs(y - self.graph.vp['y'][v]) < 2:
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                break
                
        self.graph.vp['x'][node] = x
        self.graph.vp['y'][node] = y

    def _compute_xy_coordinates(self, layout):

        z_values = np.array([self.graph.vp['z'][v] for v in self.graph.vertices()])
        unique_zs = np.unique(z_values)

        for z in unique_zs:
            z_vertices = self.graph.vertex(self.graph.vp['z'] == z)
            z_subgraph = self.graph.subgraph(z_vertices)

            # Remove type 1 edges
            edges_to_remove = []
            for e in z_subgraph.edges():
                if self.graph.ep['type'][e] == "1":
                    edges_to_remove.append(e)
            z_subgraph.remove_edge(edges_to_remove)

            if z_subgraph.num_edges() > 0:
                pos = self._compute_layout(z_subgraph, layout)
                for v in z_subgraph.vertices():
                    self.graph.vp['x'][v] = pos[v][0]
                    self.graph.vp['y'][v] = pos[v][1]

        for z in reversed(unique_zs):
            z_vertices = self.graph.vertex(self.graph.vp['z'] == z)
            for v in z_vertices:
                if self.graph.in_degree(v) > 0:
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
        pos = np.vstack([self.graph.vp['x'], self.graph.vp['y']]).T
        plt.figure(figsize=(12,8))
        gt.graph_draw(self.graph, pos=pos, vertex_text=self.graph.vp['label'], vertex_font_size=8, vertex_shape='o', vertex_fill_color='grey', edge_pen_width=1.2, output_size=(800, 600), output="graph.png")
        # Image('graph.png')
        plt.close()
    
    def save_graph(self, filename='saved_graph.gt'):
        gt.save_graph(filename, self.graph)

    def load_graph(self, filename='saved_graph.gt'):
        self.graph = gt.load_graph(filename)