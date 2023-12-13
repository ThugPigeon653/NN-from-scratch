import sqlite3
import random
import uuid
import sys
from input_data import DataManager

class Network():
    def __init__(self, layer_count:int, layer_size:int, input_size:int, output_size:int, activation_type:int, learning_rate:float) -> None:
        self.layer_size=layer_size
        self.input_size=input_size
        self.output_size=output_size
        self.layer_count=layer_count
        self.learning_rate=learning_rate
        self.cum_error:float=0.00
        self.conn=sqlite3.Connection(f'hidden-layers-{uuid.uuid4().hex[:8]}')
        self.cursor=self.conn.cursor()
        self.run_sql_file('Create.sql')
        is_first_layer:bool=True
        for layer in range(0,layer_count):
            self.cursor.execute(f'INSERT INTO layer VALUES(?,?)', (layer,activation_type))
            for node in range(0,layer_size):
                self.cursor.execute(f'INSERT INTO node VALUES(?,?)', (layer,node))
                if is_first_layer==True:
                    layer_input_size:int=input_size
                else:
                    layer_input_size=layer_size
                for fromNode in range(0,layer_input_size):
                    weight:float=random.uniform(-1.000, 1.000)
                    self.cursor.execute(f'INSERT INTO weights VALUES(?,?,?,?)',(node,fromNode,layer,weight))
                is_first_layer=False
        for node in range(0, layer_size):
            for output in range(0, output_size):
                weight:float=random.uniform(-1.000, 1.000)
                self.cursor.execute('INSERT INTO weights VALUES(?,?,?,?)', (output, node, layer_count, weight))

    def run_sql_file(self, file:str):
        path:str=f"sql/{file}"
        try:
            with open(path, 'r') as sql_file:
                sql_commands = sql_file.read()
                self.cursor.executescript(sql_commands)
        except Exception as e:
            sys.stdout.write(f"Could not execute {path}.\n{e}")

    @staticmethod
    def activation_relu(input:float)->float:
        if input<0:
            output=0
        elif input>1:
            output=1
        else:
            output=input
        return output

    @staticmethod
    def calculate_error(predicted:int, actual:int):
        return predicted-actual

    def back_propagate(self, weight_keys:[], error:int):
        adjusted_error:float=self.learning_rate*float(error)
        self.cum_error+=adjusted_error
        for key in weight_keys:
            self.cursor.execute('UPDATE weights SET weight = weight - ? WHERE toNodeId=? AND fromNodeId=? AND layerId=?', (adjusted_error, key[0], key[1], key[2]))

    # Classification algorithm, reducing many floats to one int 
    def forward_propagate(self, input_data:list[float], expected_result:int, is_training:bool=True):
        weights_used:list=[]
        current_layer:list[float]=[]
        for i in range(0,self.layer_size):
            current_layer.append(0)
        k=0
        while k<=self.layer_count:
            i=0
            while i < len(input_data):
                j=0
                if(i!=len(input_data)-1):
                    output_size:int=self.layer_size
                else:
                    output_size=self.output_size
                while(j<output_size):
                    self.cursor.execute('SELECT weight, fromNodeId, toNodeId, layerId FROM weights WHERE fromNodeId=? AND toNodeId=?', (i,j))
                    weight, fromNodeId, toNodeId, layerId=self.cursor.fetchone()
                    current_layer[j]+=input_data[i]*weight
                    # TODO: In the query above, perform a join on 'weights' and 'layer', to also retrive the activation type. This is pointless before more
                    # activation functions are written, so its always relu for now.
                    activation_type:int=0
                    if activation_type==0:
                        current_layer[j]=self.activation_relu(current_layer[j])
                    if current_layer[j]>0:
                        weights_used.append((fromNodeId, toNodeId, layerId))
                    j+=1
                input_data=current_layer
                i+=1
            k+=1
        result:int=current_layer.index(max(current_layer))
        error=self.calculate_error(result, expected_result)
        if(is_training):
            self.back_propagate(weights_used, error)

class NetworkManager():
    reporting_freuquency:int=100

    def __init__(self) -> None:
        self.data:DataManager=DataManager()
        self.nn:Network=Network(3, 64, 28, 0, 5, 0.05)

    def train(self, iterations):
        for i in range(0,iterations):
            if(i%self.reporting_freuquency==0):
                if i!=0:
                    error:float=self.nn.cum_error/self.reporting_freuquency
                else:
                    error=0
                sys.stdout.write(f"\nIteration: {i}\tAvg error: {error}\n------------------\n")
                self.cum_error=0
            test_point:{}=self.data.get_random_training_data_point()
            self.nn.forward_propagate(test_point['inputs'],int(test_point['label']))

NetworkManager().train(20000)