# example implementation of ops.py. Proves simplicity of taking this approach.
# A list of Ops ensures that whatever happens in the forward pass will always
# be mirror in the backward pass. Each op contains the foward and backward 
# operations required for that OpNode. 
# The idea of OpNodes comes from Peter Bloem's suggestions on machine learning. 
import ops
import numpy as np
import json

feature_size=2
hidden_size=4
output_size=3

alpha:float=0.005
epsilon:float = 1e-8
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
        weights.append(np.random.uniform(-0.500, 0.500, size=(model[i].output.shape[1], model[i+1].output.shape[1])))
        #weights.append(np.ones(shape=(model[i].output.shape[1], model[i+1].output.shape[1])))

# NOTE: This loop has been set up to feed in one training example repeatedly. This
# is not typically useful, but serves to prove that the model will approach truth.
# Because the model averages saved outputs across the forward-iteration count,
# it can be fed batches, and will automatically average and clear saved outputs as
# needed (see ops.p)
# forward prop

json_content:{}={}     

for i in range(0,600):
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
    # 'bp' just stands for back-propagated. This is the vector of dk^i/dL. 
    # Saving this value before computing nabla's not only saves a duplicate 
    # computation, but actually saves ^2 computations per layer added to the 
    # computation graph.
    bp=model[5].backward_propagate()*model[4].backward_propagate()
    b2_nabla=np.sum(bp)
    w2_nabla=np.outer(bp, model[2].output).T

    bp=bp.dot(weights[1].T)*model[2].backward_propagate()
    b_nabla=np.sum(bp)
    w_nabla=np.outer(bp, model[0].output).T

    # apply nabla's
    weights[0]+=alpha*w_nabla
    b1+=alpha*b_nabla
    weights[1]+=alpha*w2_nabla
    b2+=alpha*b2_nabla

    # normalize weights. this helps to prevent exploding gradient, by keeping
    # -1<weight<1.
    weights[0] /= np.max([1.0, np.max(np.abs(weights[0])) / (1.0 + epsilon)])
    weights[1] /= np.max([1.0, np.max(np.abs(weights[1])) / (1.0 + epsilon)])
    json_content[i] = {"H0": a.tolist(), "k1": b.tolist(), "H1": c.tolist(), "k2": d.tolist(), "H2": e.tolist(), "L": Loss.tolist(), "W0":weights[0].tolist(), "W1":weights[1].tolist()}
    
    with(open("output.json", "w") as file):
        json.dump(json_content, file, indent=4)