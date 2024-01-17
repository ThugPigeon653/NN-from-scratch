import numpy as np

class Network():
    def __init__(self, input_size:int, hidden_size:int, hidden_layers:int, output_size:int, batch_size:int, learning_rate:float) -> None:
        self.x:list[list[np.ndarray]]=[]
        self.z:list[list[np.ndarray]]=[]
        self.activations = []
        self.weights: list[np.ndarray] = []
        self.bias: list[float] = None
        self.weights.append(np.random.uniform(-1.000, 1.000, size=(input_size, hidden_size)))
        self.activations.append(np.zeros(shape=(batch_size, hidden_size)))  
        for i in range(0, hidden_layers - 1):
            self.weights.append(np.random.uniform(-1.000, 1.000, size=(hidden_size, hidden_size)))
            self.activations.append(np.zeros(shape=(batch_size, hidden_size)))  
        self.weights.append(np.random.uniform(-1.000, 1.000, size=(hidden_size, output_size)))
        self.activations.append(np.zeros(shape=(batch_size, output_size))) 
        self.bias = [0.00] * (len(self.weights))
        self.error: np.ndarray = np.zeros((batch_size, output_size))  
        self.batch_size = batch_size
        self.hidden_size=hidden_size
        self.hidden_layers=hidden_layers
        self.learning_rate=learning_rate
        self.output_size=output_size

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
    
    # This method helps correctly save the cumulative inputs to every weight, across each batch. This value was not explicitly 
    # calculated during forward propagation, but is required for gradient descent. 
    @staticmethod
    def inputs_node_to_weight(input:np.ndarray, toNodes:int)->np.ndarray:
        lin:list[np.ndarray]=[]
        i=0
        while i <toNodes:
            lin.append(input.T)
            i+=1
        return np.hstack(lin)
        
    def forward_propagate(self, feature:list[float], result:list[float]):
        i=0
        highest_batch:int=len(self.x)
        self.x.append([])
        self.z.append([])
        propagated_value:np.ndarray=np.array(feature).reshape(1,-1)
        self.x[highest_batch].append(propagated_value.copy())
        #print(propagated_value.shape, self.weights[i].shape, propagated_value.dot(self.weights[i]).shape)
        self.z[highest_batch].append(propagated_value.dot(self.weights[i]))
        weight_layers:int=len(self.weights)
        while i<weight_layers:
            z:np.ndarray=propagated_value.dot(self.weights[i])
            z=z+self.bias[i]
            z=self.normalize(z)
            self.z[highest_batch].append(z.copy())
            propagated_value=self.activate(z)
            self.x[highest_batch].append(propagated_value.copy())
            self.activations[i]+=propagated_value
            i+=1
        self.error+=(np.array(result)-propagated_value)
        print(self.error==propagated_value, propagated_value==0)
        #print(np.array(result)-propagated_value, self.error)

    def backward_propagate(self, iteration):
        # average out cumulative values, before starting backprop process
        avg_error:np.ndarray=(self.error/self.batch_size).T
        avg_error_scalar:float=avg_error[0,0]-avg_error[1,0]
        self.error: np.ndarray = np.zeros((self.batch_size, self.output_size))
        print(f"Iteration: {iteration}  Cost: {avg_error_scalar**2}\n")
        wd:list[np.ndarray]=[]
        bd:list[float]=[]
        x_avg:list[np.ndarray] = [np.mean(np.array(inner_list), axis=0) for inner_list in zip(*self.x)]
        self.x=[]
        top_index:int=len(x_avg)-1
        current_index:int=1
        wd.append(np.outer(avg_error, x_avg[top_index-current_index]).T)
        # hidden layers
        k=avg_error.T
        for _ in range(0, self.hidden_layers):
            bd.append(avg_error_scalar)
            k=self.activate_prime(np.dot(wd[current_index-1], k.T).T)
            wd.append(np.outer(k, x_avg[top_index-current_index-1]).T)
            current_index+=1
        for w in range(0, len(wd)):
            self.weights[w]+=self.learning_rate*wd[len(wd)-1-w]
            #print(wd[len(wd)-1-w])

n=Network(4, 32, 4 , 2, 1, 1)

i=0
while i<10000:
    n.forward_propagate([1,7,3,5], [0.9,0.14])
    n.forward_propagate([1,2,8,9], [1,0.1])
    n.forward_propagate([1,6,3,4], [1,0.8])
    n.forward_propagate([1,7,2,4], [1,0.6])
    n.forward_propagate([1,2,3,9], [1,0])
    n.forward_propagate([1,2,3,4], [1,0])
    n.forward_propagate([1,7,3,4], [1,0])
    n.forward_propagate([1,2,3,9], [1,0])
    n.forward_propagate([1,2,3,4], [0.7,0.8])
    n.forward_propagate([1,7,3,4], [1,0])
    n.forward_propagate([1,2,3,9], [1,0])
    n.forward_propagate([1,2,3,4], [1,0])
    n.forward_propagate([1,7,3,4], [1,0])
    n.forward_propagate([1,2,3,9], [1,0])
    n.forward_propagate([1,2,3,4], [1,0])
    n.backward_propagate(i)
    i+=1