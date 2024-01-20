import math
import numpy as np

class Op():
    output:np.ndarray
    def init(self, x_dim:int, y_dim:int):
        self.output=np.zeros(y_dim, x_dim)


class SigOp(Op):
    def forward_propagate(self, context:{}, inputs:np.ndarray)->np.ndarray:
        def sigmoid_outer(power:float):
            return 1/(1+(math.e**(-power)))
        inputs=np.vectorize(sigmoid_outer)(inputs)
        self.output+=inputs
        return inputs

    def backward_propagate(self, context:{}, inputs:np.ndarray):
        forward=self.forward_propagate(context, inputs)
        return forward*(1-forward)

class SumOp(Op):
    def forward_propagate(self, weights:np.ndarray, x:np.ndarray, bias:float):
        output=x.dot(weights)+bias
        self.output+=output
        return output
    
    




#print(SigOp().backward_propagate({}, np.array([[-0.9,-0.6,-0.3],[-0.1,0,0.1],[0.3,0.6,0.9]])))