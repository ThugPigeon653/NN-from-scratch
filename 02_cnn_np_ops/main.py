# Hacky example implementation of ops.py. Proves simplicity of taking this approach.
import ops
import numpy as np


feature_size=2
hidden_size=4
output_size=3

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
b=model[1].forward_propagate({}, weights[0], a, 0)
c=model[2].forward_propagate({}, b)
d=model[3].forward_propagate({}, weights[1], c, 0)
e=model[4].forward_propagate({}, d)
Loss=model[5].forward_propagate({}, e, expected)

print(Loss)
#L:float=model[5].forward_propagate({},model[4].forward_propagate({},model[3].forward_propagate({},model[2].forward_propagate({},model[1].forward_propagate({},model[0].forward_propagate({},feature), weights[0])), weights[1])), expected)