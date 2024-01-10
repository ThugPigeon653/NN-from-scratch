import numpy as np

class Network():
    def __init__(self, input_size:int, hidden_size:int, hidden_layers:int, output_size:int, batch_size:int) -> None:
        self.z = []
        self.activations = []
        self.weights: list[np.ndarray] = []
        self.bias: list[float] = None
        self.weights.append(np.random.uniform(-1.000, 1.000, size=(input_size, hidden_size)))
        self.z.append(np.zeros(shape=(batch_size, hidden_size)))  
        self.activations.append(np.zeros(shape=(batch_size, hidden_size)))  
        for i in range(0, hidden_layers - 1):
            self.weights.append(np.random.uniform(-1.000, 1.000, size=(hidden_size, hidden_size)))
            self.z.append(np.zeros(shape=(batch_size, hidden_size)))  
            self.activations.append(np.zeros(shape=(batch_size, hidden_size)))  
        self.weights.append(np.random.uniform(-1.000, 1.000, size=(hidden_size, output_size)))
        self.z.append(np.zeros(shape=(batch_size, output_size)))  
        self.activations.append(np.zeros(shape=(batch_size, output_size))) 
        self.bias = [0.00] * (len(self.weights))
        self.error: np.ndarray = np.zeros((batch_size, output_size))  
        self.batch_size = batch_size
        self.hidden_size=hidden_size

    @staticmethod
    def normalize(input:np.ndarray)->np.ndarray:
        min_val = np.min(input)
        max_val = np.max(input)
        return (input - min_val) / (max_val - min_val) * 2 - 1

    @staticmethod
    def activate(input:np.ndarray)->np.ndarray:
        return np.maximum(0,input)
    
    @staticmethod
    def activate_prime(input:np.ndarray)->np.ndarray:
        return np.where(input < 0, 0, input)

    def forward_propagate(self, feature:list[float], result:list[float]):
        i=0
        propagated_value:np.ndarray=np.array(feature).reshape(1,-1)
        weight_layers:int=len(self.weights)
        while i<weight_layers:
            z=propagated_value.dot(self.weights[i])
            z=z+self.bias[i]
            self.z[i]+=z
            z=self.normalize(z)
            propagated_value=self.activate(z)
            self.activations[i]+=propagated_value
            i+=1
        self.error+=(propagated_value-np.array(result))

    def backward_propagate(self):
        #weight_delta:np.ndarray=np.ndarray(shape=())
        layer_index:int=len(self.weights)-1
        weight_dels:list[np.ndarray]=[]
        avg_error=self.error/self.batch_size
        z=np.array(self.z[layer_index-1])/self.batch_size
        a=np.array(self.activations[layer_index])/self.batch_size
        # Final layer
        mem:np.ndarray=avg_error.T.dot(self.activate_prime(z))
        print(a.shape, mem.shape)
        w_del=a.dot(mem)
        print(mem.shape, w_del.shape)
        weight_dels.append(w_del)
        
        # Hidden Layers
        print(mem.shape, self.weights[layer_index].shape)
        mem=mem.T.dot(self.weights[layer_index].T)
        layer_index-=1
        z=np.array(self.z[layer_index-1])/self.batch_size
        a=np.array(self.activations[layer_index-1])/self.batch_size
        mem=self.activate_prime(z).dot(mem)
        print(mem.shape, a.shape)
        w_del=mem.dot(a.T)
        weight_dels.append(w_del)

        print(weight_dels)

n=Network(4, 8, 4 , 2, 1)
n.forward_propagate([1,2,3,4], [1,0])
n.backward_propagate()