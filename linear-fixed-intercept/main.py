import math
import sqlite3
import random
import uuid
import sys
from input_data import DataManager
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Network():
    
    activations:list[list[float]]=[]
    # TODO: SQLite INSERT statements always return -1 rowcount, so we cant use this param for error checking. use another approach,
    # preferably without extra calls to db
    def __init__(self, layer_count:int, layer_size:int, input_size:int, output_size:int, activation_type:int, learning_rate:float) -> None:
        self.layer_size=layer_size
        self.input_size=input_size
        self.output_size=output_size
        self.layer_count=layer_count
        self.learning_rate=learning_rate
        self.cum_error:float=0.00
        self.session_string:str=uuid.uuid4().hex[:8]
        print(f"INSTANCING DB ({self.session_string})...")
        self.conn=sqlite3.Connection(f'hidden-layers-{self.session_string}.db')
        self.cursor=self.conn.cursor()
        self.run_sql_file('Create.sql')
        nodes = [[] for _ in range(layer_count + 3)]
        # insert layers
        for layer_index in range(0,layer_count+2):
            self.cursor.execute('INSERT INTO layer VALUES(?,?)',(layer_index,0))
        # Insert Nodes
        for node_index in range(0, input_size):
            self.cursor.execute('INSERT INTO node VALUES(?,?)',(0,node_index))
            nodes[0].append(node_index)
        layer_index=1
        while layer_index<=layer_count+1:
            for node_index in range(0,layer_size):
                self.cursor.execute('INSERT INTO node VALUES(?,?)',(layer_index,node_index))
                nodes[layer_index].append(node_index)
            layer_index+=1
        for node_index in range(0,output_size):
            self.cursor.execute('INSERT INTO node VALUES(?,?)',(layer_count+2, node_index))
            nodes[layer_count+2].append(node_index)


        layer_index=1
        while layer_index<len(nodes):
            for input_node in nodes[layer_index-1]:
                for output_node in nodes[layer_index]:
                    #print(f"fromLayer: {layer_index-1}\tfromNode: {input_node}\ttoNode: {output_node}")
                    try:
                        self.cursor.execute('INSERT INTO weights VALUES(?,?,?,?)',(input_node, output_node, layer_index-1, random.uniform(-1.000, 1.000)))
                    except:
                        self.conn.commit()
                        self.conn.close()
                        raise Exception
            layer_index+=1
        self.conn.commit()        
        sys.stdout.write("DB initialization complete.")

    def run_sql_file(self, file:str):
        path:str=f"sql/{file}"
        try:
            with open(path, 'r') as sql_file:
                sql_commands = sql_file.read()
                self.cursor.executescript(sql_commands)
        except Exception as e:
            sys.stdout.write(f"Could not execute {path}.\n{e}")

    @staticmethod
    def layer_activation_relu(inputs:list[float])->list[float]:
        output:list[float]=[]
        for input in inputs:
            if input<=0:
                output.append(0)
            else:
                output.append(input)
        return output

    @staticmethod
    def node_activation_relu(input)->float:
        if input<0:
            output=0
        else:
            output=input
        return output

    @staticmethod
    def square_error(predicted:list[float], actual:list[float]):
        outcome=[]
        #print(f"-- {len(predicted)}    {len(actual)} --")
        for i in range(len(predicted)):
            outcome.append(((predicted[i]-actual[i])**2)/2)
        return outcome
    
    # I have intentionally left this un-typed, so it will work for any numerical value. Error handling should be implemented here.
    @staticmethod
    def normalize_list(values:[])->[]:
        #print(f'\t{values}')
        if not values:
            return []
        value_min = min(values) 
        value_max = max(values) 
        translation = (value_min + value_max) / 2 
        dilation = value_max - value_min  
        normalized_values = [2*((value - translation) / dilation) for value in values]
        return normalized_values

    # No PLSQL-like features exist in sqlite, so we perform that logic with python, using simple sql queries.
    def apply_weight(self, inputs:[], output_size:int, input_layer_index:int)->[]:
        #print(len(inputs), output_size)
        output:[]=[0]*output_size
        self.cursor.execute(f'CREATE VIEW layer_weights AS SELECT * FROM weights WHERE layerId = {input_layer_index}')
        self.conn.commit()
        for output_node_index in range(0,output_size):
            self.cursor.execute('SELECT toNodeId, fromNodeId, weight FROM layer_weights WHERE toNodeId = ?', (output_node_index,))
            weights_for_output_node=self.cursor.fetchall()
            for w in weights_for_output_node:
                toNode,fromNode,weight=w
                output[toNode]+=weight*inputs[fromNode]

        self.cursor.execute('DROP VIEW layer_weights')
        self.conn.commit()
        #print(len(output))
        return output

    def back_propagate(self, weight_keys:[], error:int):
        adjusted_error:float=self.learning_rate*float(error)
        self.cum_error+=error
        for key in weight_keys:
            self.cursor.execute('UPDATE weights SET weight = weight - ? WHERE toNodeId=? AND fromNodeId=? AND layerId=?', (adjusted_error, key[0], key[1], key[2]))

    # Classification algorithm, reducing many floats to one int 
    # NOTE: Numerous functions have been made from the previous implementation. Even though they are  only called by this function,
    # re-usability is important, because it means we can easily change how the model forward propagates
    def forward_propagate(self, input_data:list[float], expected_result:list[float], is_training:bool=True):
        # List format ->   [(<layer_index> , <node_used>])]
        self.used_nodes:[]=[]
        activations=[]
        for i in range(0,self.layer_count+1):            
            input_data=self.apply_weight(input_data, self.layer_size, i)
            input_data=self.normalize_list(input_data)
            input_data=self.layer_activation_relu(input_data)
            activations.append(input_data.copy())

        i=self.layer_count+1
        input_data=self.apply_weight(input_data, self.output_size, i)
        input_data=self.normalize_list(input_data)
        input_data=self.layer_activation_relu(input_data)
        activations.append(input_data.copy())

        error:list[float]=self.square_error(input_data, expected_result)

        i=0
        if len(self.activations)!=0:
            for i in range(0,len(error)):
                print(i)
                print(self.cum_error[i], error[i])
                self.cum_error[i]+=error[i]
            while i<len(activations):
                j=0
                while j<len(activations[j]):
                    self.activations[i][j]+=activations[i][j]
                    j+=1
                i+=1
        else:
            self.cum_error=error.copy()
            self.activations=activations.copy()


        # This logic was more complicated and messy than it needed to be. Leave it here until replaced

class NetworkManager():
    reporting_freuquency:int=5

    def __init__(self) -> None:
        self.data:DataManager=DataManager()
        self.nn:Network=Network(
            layer_count=3, 
            layer_size=64, 
            input_size=784, 
            output_size=10, 
            activation_type=0, 
            learning_rate=0.0001)

    def train(self, iterations:int, batch_size:int):
        self.reporting_freuquency=batch_size
        for i in range(0,iterations):
            if(i%self.reporting_freuquency==0):
                if i!=0:
                    for i in range(0, len(self.nn.cum_error)):
                        self.nn.cum_error[i]/self.reporting_freuquency
                    for i in range(0,len(self.nn.activations)):
                        for j in range(len(self.nn.activations[i])):
                            self.nn.activations[i][j]/=batch_size
                try:
                    with open("logs/training-log.txt") as file:
                        output:str=file.read()
                except:
                    output=""
                output+=f"\nIteration: {i}\tMean square error: {self.nn.cum_error}\nActivations: {self.nn.activations}\n------------------\n"
                with(open("logs/training-log.txt", "w") as file):
                    file.write(output)
                sys.stdout.write(output)
                self.nn.cum_error=[]
                self.nn.activations=[]
            test_point:{}=self.data.get_random_training_data_point()
            expected_result:list[float]=[0]*self.nn.output_size
            expected_result[int(test_point['label'])]=1
            self.nn.forward_propagate(test_point['inputs'],expected_result)

ml=NetworkManager()
ml.train(20000, 100)