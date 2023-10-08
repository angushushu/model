from Graph_opt import Graph
import json
import os

with open(os.getcwd()+'/sequences.json', 'r') as file:
    sequences = json.load(file)

# Initialize Graph
graph = Graph()


# Generate node IDs
def generate_id():
    i = 1
    while True:
        yield i
        i += 1


id_gen = generate_id()

# Add nodes and edges to the graph
for j, sequence in enumerate(sequences):  # Using all sequences
    if sequence is None:
        continue
    # print('sequence', sequence)
    label_seq = []
    for i, state in enumerate(sequence):
        # print('i', i)
        # print('state', state)
        state_id = next(id_gen)
        state_label = str(state)
        # print('state_label', state_label)
        state_label = graph.add_rep(state_label, value=sorted(state))  # Storing id as value for future use
        # print(state_label)
        label_seq.append(state_label)

        # Add elements as nodes and connect them to the state node with connection1
        for element in state:
            element_label = graph.add_rep(label=element, value=element)
            # print('element_label', element_label)
            graph.add_edge(element_label, state_label, connection_type="1")

        # Connect state nodes with connection2 (if not the first state)
        if i > 0:
            graph.add_edge(label_seq[i-1], state_label, connection_type="2")

# Calculate coordinates and visualize the graph
#spring, shell, kamada_kawai, fruchterman_reingold, spectral, planar
graph.calculate_coordinates('kamada_kawai')
print('hi')
graph.visualize_graph()
print('hi2')
for node, data in graph.graph.nodes(data=True):
    if data['label'] == 'Seq1-S3':
        print(graph.graph.nodes[node]['x'],graph.graph.nodes[node]['y'],graph.graph.nodes[node]['z'])
    if data['label'] == 'Seq1-S4':
        print(graph.graph.nodes[node]['x'],graph.graph.nodes[node]['y'],graph.graph.nodes[node]['z'])
graph.save_graph('saved_graph.pkl')