import numpy as np

class Network():
    def __init__(self, input_size:int, hidden_size:int, hidden_layers:int, output_size:int) -> None:
        self.layers:list[np.ndarray]=[]
        self.layers.append(np.random.uniform(-1.000, 1.000, size=(input_size, hidden_size)))
        for i in range(0,hidden_layers-2):
            self.layers.append(np.random.uniform(-1.000, 1.000, size=(hidden_size, hidden_size)))
        self.layers.append(np.random.uniform(-1.000, 1.000, size=(hidden_size, output_size)))
