# example implementation of ops.py. Proves simplicity of taking this approach.
# A list of Ops ensures that whatever happens in the forward pass will always
# be mirror in the backward pass. Each op contains the foward and backward 
# operations required for that OpNode. 
# The idea of OpNodes comes from Peter Bloems suggestions on machine learning. 
import ops
import numpy as np

feature_size=2
hidden_size=4
output_size=3

alpha:float=0.05
b1:float=0
b2:float=0

model:list[ops.Op]=[]
weights:list[np.ndarray]=[]

# Feature in, vec out
model.append(ops.SigOp(1,feature_size))
# hidden layer/s
model.append(ops.SumOp(1, hidden_size))
model.append(ops.SigOp(1, hidden_size))
# output layer
model.append(ops.SumOp(1, output_size))
model.append(ops.SigOp(1, output_size))
model.append(ops.MSEOp(1, output_size))

c=len(model)-3
for i in range(0, c):
    if i%2==0:
        weights.append(np.random.uniform(-1.000, 1.000, size=(model[i].output.shape[1], model[i+1].output.shape[1])))

# forward prop
feature:np.ndarray=np.array([[0.1, 0.8]])
expected:np.ndarray=np.array([[0, 0.9, 0]])
a=model[0].forward_propagate({},feature)
b=model[1].forward_propagate({}, weights[0], a, b1)
c=model[2].forward_propagate({}, b)
d=model[3].forward_propagate({}, weights[1], c, b2)
e=model[4].forward_propagate({}, d)
Loss=model[5].forward_propagate({}, e, expected)
# NOTE: The following (commented-out) line of code shows the forward pass as a 
# single nested function. This is identical to the above approach in practice. 
# Loss=f(x)=0.5(sig(xW+b))^2
#Loss=model[5].forward_propagate({}, model[4].forward_propagate({}, model[3].forward_propagate({}, weights[1], model[2].forward_propagate({}, model[1].forward_propagate({}, weights[0], model[0].forward_propagate({},feature), 0)), 0)), expected)
print(f"MSE Loss: {Loss}\n")

# backward prop. 
# 'bp' just stands for back-propagated. This is the vector of dk^i/dL^i. 
# Saving this value before computing nabla's not only saves a duplicate 
# computation, but actually saves ^2 computations per layer added to the 
# computation graph.
bp=model[5].backward_propagate()*model[4].backward_propagate()
b_nabla=sum(bp)
w_nabla=np.outer(bp, model[2].output).T
weights[1]-=alpha*w_nabla
b2-=alpha*b_nabla

bp=bp.dot(weights[1].T)*model[2].backward_propagate()
b_nabla=sum(bp)
w_nabla=np.outer(bp, model[0].output).T
weights[0]-=alpha*w_nabla
b1-=b_nabla