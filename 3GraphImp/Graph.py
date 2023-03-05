from cProfile import label
from logging import root
import networkx as nx
import matplotlib.pyplot as plt


def draw(g, mapping):
    print('Shared drawing func')
    pos = nx.spring_layout(g)
    nodes = g.nodes

    # trans hex base on activation
    def adjust_color(hex, perc):
        perc = min(1.0, perc)*.5+.5
        rgb = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple([int(perc*x) for x in rgb])
        return '%02x%02x%02x' % rgb
    colors = ['#'+adjust_color(mapping[nodes[n]['type']].lstrip('#'),nodes[n]['activation']) for n in nodes]
    labels = {}
    # draw labels
    for n in nodes:
        if nodes[n]['label'] != '':
            labels[n] = str(nodes[n]['label']) + '-' + str(round(nodes[n]['activation'],2))
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

