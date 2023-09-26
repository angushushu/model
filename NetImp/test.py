import matplotlib.pyplot as plt
import networkx as nx
import random
from Net import *

# 创建Net对象
net = Net()

# 添加初始UnitNodes
initial_unit_nodes = [UnitNode(label=f"U{i}") for i in range(30)]
for node in initial_unit_nodes:
    net.add_node(node)

# 逐层添加NonUnitNode
previous_nodes = initial_unit_nodes
for i in range(1, 30):  # 10层
    new_nodes = []
    for _ in range(random.randint(1, 30)):  # 每层添加1到10个NonUnitNode
        nodes_to_connect = random.sample(previous_nodes, random.choice([1, len(previous_nodes)]))
        seqs = []
        reps = nodes_to_connect.copy()
        while len(reps) > 1:
            if random.choice([True, False]):
                chosen_seq = tuple(random.sample(reps, 2))
                seqs.append(chosen_seq)
                reps.remove(chosen_seq[0])
                reps.remove(chosen_seq[1])
            else:
                break

        non_unit_node = NonUnitNode(reps=reps, seqs=seqs)
        net.add_node(non_unit_node)
        new_nodes.append(non_unit_node)
    previous_nodes = new_nodes

# 设置位置
net.set_pos()

# 绘制网络
plt.figure(figsize=(12, 8))
node_colors = ["#303245" if isinstance(node, UnitNode) else "#F4D35E" for node in net.net.nodes()]
edge_colors = ["#EE964B" if net.net[u][v]["seq"] else "#06D6A0" for u, v in net.net.edges()]
edge_styles = ["dashed" if net.net[u][v]["seq"] else "solid" for u, v in net.net.edges()]
edge_widths = [2 if net.net[u][v]["seq"] else 1 for u, v in net.net.edges()]

nx.draw(net.net, pos=net.pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, edge_vmin=0,
        edge_vmax=1, width=edge_widths, style=edge_styles, node_size=300)
plt.show()
