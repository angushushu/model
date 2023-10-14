import random

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Graph_opt5 import Graph
from tqdm import tqdm

train_dataset = datasets.MNIST(root='./data',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)

test_dataset = datasets.MNIST(root='./data',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=True)

batch_size = 64
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)

for i, data in enumerate(train_loader):
    inputs, labels = data
    # print(inputs.shape)
    # print(labels.shape)
    break

img_test = test_dataset[0][0][0].numpy()
# plt.imshow(img_test, cmap='gray')
# plt.show()
# plt.imshow(img_test[:10], cmap='gray')
# plt.show()
# print(img_test.shape)
# print(test_dataset[0][1])

nn = Graph()
# input layer
for y in range(img_test.shape[0]):
    for x in range(img_test.shape[1]):
        nn.add_rep(f'{x}-{y}', x=x, y=y, z=0)

# output layer
for i in range(10):
    nn.add_rep(f'o{i}', x=i*3, y=15, z=1)

for i in range(10):
    for y in range(img_test.shape[0]):
        for x in range(img_test.shape[1]):
            nn.add_edge(f'{x}-{y}', f'o{i}', weight=random.uniform(0, 1))
# 100 ep - 127/192
# 10 ep - 115/192

# hidden
# for y2 in range(10):
#     for x2 in range(10):
#         nn.add_rep(f'h{x2}-{y2}', x=x2*3, y=y2*3, z=1)
#         for y in range(img_test.shape[0]):
#             for x in range(img_test.shape[1]):
#                 nn.add_edge(f'{x}-{y}', f'h{x2}-{y2}', weight=random.uniform(0, 1))
#         for i in range(10):
#             nn.add_edge(f'h{x2}-{y2}', f'o{i}', weight=random.uniform(0, 1))
# training
for i, data in enumerate(bar:=tqdm(train_loader)):
    for episode in range(10):
        bar.set_description(f'{episode+1 }/10')
        imgs, labels = data
        for j in range(len(imgs)):
            img = imgs[j][0].numpy()
            # print(img.shape)
            label = labels[j]
            # print(f'o{label}')
            input = dict()
            output = {f'o{label}':1}
            for y in range(img.shape[0]):
                for x in range(img.shape[1]):
                    # print(float(img[y][x]))
                    input[f'{x}-{y}'] = float(img[y][x])
                    # if img[y][x] > 0:
                    #     nn.add_edge(f'{x}-{y}', f'o{label}', weight=random.uniform(0,0.1))
                    # ep 10 - ~60
                    # for y2 in range(10):
                    #     for x2 in range(10):
                    #         nn.add_edge(f'{x}-{y}', f'h{x2}-{y2}', weight=random.uniform(0, 1))
                    # randomize weight?
            nn.forward_propagation(input)
            nn.backward_propagation(output)

correct = 0
total = 0
for i, data in enumerate(test_loader):
    imgs, labels = data
    for j in range(len(imgs)):
        img = imgs[j][0].numpy()
        label = labels[j]
        input = dict()
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                input[f'{x}-{y}'] = float(img[y][x])
        nn.forward_propagation(input)
        out_nodes = [node for node, data in nn.graph.nodes(data=True) if data['z'] == 1]
        max_value_node = max(out_nodes, key=lambda node: nn.graph.nodes[node]['value'])
        # print('predict:', nn.graph.nodes[max_value_node]['label'])
        # print(f'answer: o{label}')
        if nn.graph.nodes[max_value_node]['label'] == f'o{label}':
            correct += 1
        total += 1
print(f'{correct}/{total}')
nn.visualize_graph(['o1'])