import numpy as np

class Network():
    def __init__(self, input_size:int, hidden_size:int, hidden_layers:int, output_size:int, batch_size:int) -> None:
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
        print(propagated_value)
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
        self.error+=(propagated_value-np.array(result))

    def backward_propagate(self):
        # average out the cumulative scores for z, x, e, etc. This is done in backprop method, because botrh actions happen 
        # exactly once per batch
        avg_error=self.error/self.batch_size
        x_avg:list[np.ndarray] = [np.mean(np.array(inner_list), axis=0) for inner_list in zip(*self.x)]
        print(len(x_avg))
        for i in range(0, len(x_avg)):
            print(x_avg[i].shape)
        #    #x_avg[i]=self.inputs_node_to_weight(x_avg[i], self.weights[i].shape[1])
        rp_z_avg:list[np.ndarray] = [self.activate_prime(np.mean(np.array(inner_list), axis=0)) for inner_list in zip(*self.z)]
        index:int=len(rp_z_avg)-2
        wd=[]
        # find all layer gradients
        

        '''print(avg_error.shape, rp_z_avg[index+1].shape, x_avg[index+1].shape)
        dels=avg_error*rp_z_avg[index+1]
        wd.append(dels*x_avg[index+1])
        print(dels.shape)
        index-=1
        dels=dels*self.weights[index+1]
        print(dels.shape, rp_z_avg[index+1].shape, x_avg[index+1].shape)
        dels=dels*rp_z_avg[index+1]
        wd.append(dels*x_avg[index+1])
        print(wd[1])

        dels=avg_error.T.dot(rp_z_avg[index]).T
        wd.append(dels*x_avg[index])
        index-=1    
        print(dels.shape)
        while index>=0:
            #print(f"dels= d{dels.shape} x z{rp_z_avg[index].shape} x w{self.weights[index].shape} x{x_avg[index].shape}")
            index-=1'''

n=Network(4, 8, 4 , 2, 1)
n.forward_propagate([1,7,3,4], [1,0])
n.forward_propagate([1,2,3,9], [1,0])
n.forward_propagate([1,2,3,4], [1,0])
n.backward_propagate()