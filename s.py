import random
import math
z=0
t=0
for i in range(0,1000):
    a=random.randint(0,9)
    b=random.randint(0,9)
    z+=1
    t+=abs(b-a) 
total=t/z
print(f'\n\n--------\n{total}')
