import math
import numpy as np

class Op():
    def forward_propagate(context:{}, inputs:np.ndarray):
        pass
    
    def backward_propagate(context:{}, output_gradient:np.ndarray):
        pass

class SigOp():
    @staticmethod
    def forward_propagate(context:{}, inputs:np.ndarray):
        def sigmoid_outer(power:float):
            return 1/(1+(math.e**(-power)))
        inputs=np.vectorize(sigmoid_outer)(inputs)

    def backward_propagate(self, context:{}, inputs:np.ndarray):
        forward=self.forward_propagate(context, inputs)
        return forward*(np.ones_like(shape=inputs.shape))

class PlusOp():
    def forward_propagate(context:{}, a, b):
        return a+b
    
    def backward_propagate(context, output):
        return output, output
SigOp().forward_propagate({}, np.array([[0.1,0.2,0.3],[0.4,0.5,0.6]]))