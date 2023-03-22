from cProfile import label
from logging import root
import networkx as nx
import matplotlib.pyplot as plt
import random
from EltType import Type

# bokeh
from bokeh.palettes import Category20_20
from bokeh.plotting import figure, from_networkx, show

# untested
def str_to_int_graph(g):
    mapping = map()
    mapping = {n:int(Type.str_to_int(n.split('_')[0])+n.split('_')[1]) for n in g.nodes}
    return nx.relabel_nodes(g, mapping)
        
def int_to_str_graph(g):
    mapping = map()
    mapping = {n:(Type.int_to_str(int(str(n)[0])+str(n)[1:])) for n in g.nodes}
    return nx.relabel_nodes(g, mapping)

def bokeh_draw(g):
    # an issue is that, bokeh doesn't support string nodes

    p = figure(x_range=(-2, 2), y_range=(-2, 2),
            x_axis_location=None, y_axis_location=None,
            tools="hover", tooltips="index: @index")
    p.grid.grid_line_color = None

    graph = from_networkx(g, nx.spring_layout, scale=1.8, center=(0,0))
    p.renderers.append(graph)

    # Add some new columns to the node renderer data source
    graph.node_renderer.data_source.data['index'] = list(range(len(G)))
    graph.node_renderer.data_source.data['colors'] = Category20_20

    graph.node_renderer.glyph.update(size=20, fill_color="colors")

    show(p)

def get_pos(x = None, y = None): # used when nodes added
    if x is None:
        print('not x and x is',x)
        x = random.randrange(0, 100)
    if y is None:
        y = random.randrange(0, 100)
    return (x, y)

def draw(g, mapping):
    print('Shared drawing func')
    # pos_sp = nx.spring_layout(g)
    pos = nx.get_node_attributes(g,'position')
    # pos_final = {'x':[],'y':[]}
    # print('------pos type: ', pos)
    # for i in pos:
    #     print(i)
    #     print(type(i))
    #     pos[i] = (pos[i][1], pos_sp[i][1])

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

def ReLU(input:int):
    return 0 if input <= 0 else input