import random

input_size: int = 10
hidden_layers: int = 3
hidden_nodes: int = 16
output_size: int = 10

# indexed as: [node_from][node_to]
weights:list[list[float]] = []
# indexed by layer. keep in mind that 0 index translates to 1 index
biases:list[float] = []

def setup_layers():
    weights.append([])
    for feature in range(0, input_size):
        weights[0].append(random.uniform(-1.000, 1.000))
    for layer in range(0, hidden_layers):
        weights.append([])
        for i in range(0, hidden_nodes):
            weights[layer + 1].append(random.uniform(-1.000, 1.000))
    weights.append([])
    for i in range(0, output_size):
        weights[len(weights)-1].append(random.uniform(-1.000, 1.000))
    for i in range(0,len(weights)-2):
        biases.append(0.00)

setup_layers()
print(biases)