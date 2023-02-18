from cProfile import label
from logging import root
import networkx as nx
import matplotlib.pyplot as plt


def draw(g, mapping):
    print('Shared drawing func')
    pos = nx.spring_layout(g)
    nodes = g.nodes

    #
    # for i in nodes:
    #     print(i)
    #     print(nodes[i])

    # print(nodes[40]['type'])
    colors = [mapping[nodes[n]['type']] for n in nodes]
    labels = {}
    # draw labels
    for n in nodes:
        if nodes[n]['label'] != '':
            labels[n] = nodes[n]['label']
        else:
            labels[n] = n
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes,
        node_color=colors
    )
    nx.draw_networkx_labels(
        g,
        pos,
        labels=labels,
        font_color='#000'
    )
    nx.draw_networkx_edges(
        g,
        pos,
        edgelist=g.edges,
        width=2,
        alpha=0.5,
        edge_color="#aaa",
    )
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=nx.get_edge_attributes(g, 'act')
    )

