import math
import numpy as np

class Op():
    __output:np.ndarray
    batch_size_count:int
    def __init__(self, y_dim, x_dim:int):
        self.batch_size_count=0
        self.output=np.zeros((y_dim, x_dim))
    
    @property
    def output(self) -> np.ndarray:
        return np.zeros_like(self.__output) if self.batch_size_count == 0 else self.__output / self.batch_size_count

    @output.setter
    def output(self, out:np.ndarray):
        self.__output=out

    def increment_count(self):
        self.batch_size_count+=1

class SigOp(Op):
    k:np.ndarray=None
    def forward_propagate(self, context:{}, inputs:np.ndarray)->np.ndarray:
        self.increment_count()
        if(self.batch_size_count==1):
            self.k=inputs
        else:
            self.k+=inputs
        def sigmoid_outer(power:float):
            if power > 0:
                power=min(power, 700)
            else:
                power=max(-700, power)
            return 1/(1+(math.e**(-power)))
        inputs=np.vectorize(sigmoid_outer)(inputs)
        self.output+=inputs
        return inputs

    def backward_propagate(self, context:{}={}):
        self.k=self.k/self.batch_size_count
        forward=self.forward_propagate(context, self.k)
        self.output=np.zeros(shape=self.output.shape)
        self.batch_size_count=0
        return forward*(1-forward)

class SumOp(Op):

    __weights:np.ndarray=None

    @property
    def weights(self)->np.ndarray:
        return self.__weights
    
    @weights.setter
    def weights(self, weights:np.ndarray):
        self.__weights=weights

    def forward_propagate(self, context:{}, weights:np.ndarray, x:np.ndarray, bias:float):
        self.increment_count()
        output=x.dot(weights)+bias
        if(self.weights is None):
            self.weights=weights
        else:
            self.weights+=weights
        self.output+=output
        return output
    
    def backward_propagate(self):
        bp=self.weights.T
        self.output=np.zeros(shape=self.output.shape)
        self.batch_size_count=0
        return bp
        

class MSEOp(Op):
    def forward_propagate(self, context:{}, inputs:np.ndarray, expected:np.ndarray)->float:
        self.increment_count()
        self.output=expected-inputs
        return np.sum((self.output**2)/2)
    
    def backward_propagate(self):
        bp=np.sum(self.output)
        self.output=np.zeros(shape=self.output.shape)
        self.batch_size_count=0
        return bp