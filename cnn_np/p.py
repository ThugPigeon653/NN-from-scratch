import numpy as np

l=[1,2,3,4]

a=np.array(l).reshape(1,-1)
print(np.outer(a.T,a))