import itertools
import networkx as nx

'''
type:
    unit(either rep or act): n = n
    inclusion: n = {a1,a2...an}
        rep
        physical_act
        sequence
    sequence: n = (a1,a2)
function:
    add_node(type, elements, label) -> id
    connect1(elements, target, weights)
        build inclusion relation
    connect2(sequence, target, weight)
        build sequence relation
    remove_node(id) ->
    get_node(id) -> node
    change_weight(node1, node2)
    set_weight(node1, node2)
'''


class Net:
    def __init__(self, unit_nodes=None) -> None:
        self.net = nx.DiGraph()
        self.nodes = self.net.nodes
        self.pos = {} # 可能要为seq添加z轴
        if unit_nodes:
            self.net.add_nodes_from(unit_nodes)

    def add_node(self, node):
        self.net.add_node(node)
        if isinstance(node, NonUnitNode):
            for n in node.graph.nodes:
                self.net.add_edge(n, node, seq=False)
            for e in node.graph.edges:
                self.net.add_edge(e[0], node, seq=True)
                self.net.add_edge(e[1], node, seq=True)

    def add_rep_edge(self, n1, n2):
        self.net.add_edge(n1, n2, seq=False)
        n2.add_nodes([n1])

    def add_seq_edge(self, seq, node):
        self.net.add_edge(seq[0], node, seq=True)
        self.net.add_edge(seq[1], node, seq=True)
        node.add_edge(seq)

    def get_unit_nodes(self):
        return [n for n in self.nodes if isinstance(n, UnitNode)]

    def compute_distances(self):
        distances = {}
        visited = set()
        queue = []

        # 初始化所有UnitNode的距离为0
        for n in self.nodes:
            if isinstance(n, UnitNode):
                distances[n] = 0
                queue.append(n)  # 将UnitNode添加到队列中，以便开始BFS

        # 使用BFS计算从NonUnitNode到UnitNode的距离
        while queue:
            current = queue.pop(0)
            for neighbor in self.net.successors(current):  # 获取当前节点的前驱节点
                # if neighbor not in visited:
                if neighbor in visited:
                    distances[neighbor] = max(distances[current] + 1, distances[neighbor])
                else:
                    distances[neighbor] = distances[current] + 1
                    visited.add(neighbor)
                    queue.append(neighbor)

        return distances

    def set_pos(self):
        distances = self.compute_distances()
        layer_counts = {}
        for n, dist in distances.items():
            x = dist
            y = layer_counts.get(x, 0)
            layer_counts[x] = y + 1
            self.pos[n] = (x, y)
        # 设置节点属性
        pos_attrs = {node: {"pos": position} for node, position in self.pos.items()}
        nx.set_node_attributes(self.net, pos_attrs)

# each node is a graph, including sequence(edges) and set(nodes)
class UnitNode:
    id_iter = itertools.count()

    def __init__(self, label=None, value=.0) -> None:
        self.id = next(self.id_iter)
        self.label = str(self.id) if label is None else label
        self.value = value

    def __str__(self):
        return self.label+'-'+str(self.id)


class NonUnitNode(UnitNode):
# reps = [node, node...node]
# seqs = [(node1, node2), (node3, node4)]
    def __init__(self, reps=None, seqs=None, bias=1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.bias = bias
        self.graph = nx.DiGraph()
        if reps:
            self.graph.add_nodes_from(reps)
        if seqs:
            self.graph.add_edges_from(seqs)

    def add_nodes(self, nodes) -> None:
        self.graph.add_nodes_from(nodes)

    def add_edges(self, edges) -> None:
        self.graph.add_edges_from(edges)
