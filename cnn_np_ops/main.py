import math
import numpy as np

class Op():
    def forward_propagate(context:{}, inputs:np.ndarray):
        pass
    
    def backward_propagate(context:{}, output_gradient:np.ndarray):
        pass

class SigOp():
    @staticmethod
    def forward_propagate(context:{}, inputs:np.ndarray)->np.ndarray:
        def sigmoid_outer(power:float):
            return 1/(1+(math.e**(-power)))
        return np.vectorize(sigmoid_outer)(inputs)

    def backward_propagate(self, context:{}, inputs:np.ndarray):
        forward=self.forward_propagate(context, inputs)
        print(forward*(1-forward))
        return forward*(1-forward)

print(SigOp().backward_propagate({}, np.array([[-0.9,-0.6,-0.3],[-0.1,0,0.1],[0.3,0.6,0.9]])))