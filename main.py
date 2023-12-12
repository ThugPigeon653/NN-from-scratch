import numpy as np

class Node():
    __weights:np.ndarray

class Layer():
    __activation_type:int
    __nodes:list[Node]
    def __init__(self, activation_type, num_nodes):
        self.__activation_type=activation_type
        self.__nodes=[]
        for i in range(0,num_nodes):
            self.__nodes.append(Node)

    @staticmethod
    def activation(node:Node)->float:
        return