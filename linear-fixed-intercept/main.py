import math
import sqlite3
import random
import uuid
import sys
from input_data import DataManager
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Network():
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
        print(f"INSTANCING DB;/.{self.session_string}")
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
    def square_error(predicted:int, actual:int):
        return math.pow(predicted-actual, 2)
    
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
        output:[]=[0]*len(inputs)
        self.cursor.execute(f'CREATE VIEW layer_weights AS SELECT * FROM weights WHERE layerId = {input_layer_index}')
        self.conn.commit()
        for output_node_index in range(0,output_size):
            self.cursor.execute('SELECT toNodeId, fromNodeId, weight FROM layer_weights WHERE toNodeId = ?', (output_node_index,))
            weights_for_output_node=self.cursor.fetchall()
            for w in weights_for_output_node:
                toNode,fromNode,weight=w
                adjustment=weight*inputs[fromNode]
                output[toNode]+=adjustment

                # Likely an excessively expensive way of tracking which parts of the network are to blame for error. 
                tup=(toNode, fromNode, input_layer_index)
                if output[toNode]>0:
                    if tup not in self.used_nodes:
                        self.used_nodes.append(tup)
                else:
                    if tup in self.used_nodes:
                        self.used_nodes.remove(tup)  
        self.cursor.execute('DROP VIEW layer_weights')
        self.conn.commit()
        return output

    def back_propagate(self, weight_keys:[], error:int):
        adjusted_error:float=self.learning_rate*float(error)
        self.cum_error+=error
        for key in weight_keys:
            self.cursor.execute('UPDATE weights SET weight = weight - ? WHERE toNodeId=? AND fromNodeId=? AND layerId=?', (adjusted_error, key[0], key[1], key[2]))

    # Classification algorithm, reducing many floats to one int 
    # NOTE: Numerous functions have been made from the previous implementation. Even though they are  only called by this function,
    # re-usability is important, because it means we can easily change how the model forward propagates
    def forward_propagate(self, input_data:list[float], expected_result:int, is_training:bool=True):
        # List format ->   [(<layer_index> , <node_used>])]
        self.used_nodes:[]=[]

        for i in range(0,self.layer_count+1):
            input_data=self.apply_weight(input_data, self.layer_size, i)
            input_data=self.normalize_list(input_data)
            input_data=self.layer_activation_relu(input_data)

        i=self.layer_count+1
        input_data=self.apply_weight(input_data, self.output_size, i)
        input_data=self.normalize_list(input_data)
        input_data=self.layer_activation_relu(input_data)

        result:int=input_data.index(max(input_data))
        error:int=self.square_error(result, expected_result)
        
        if(is_training):
            self.back_propagate(self.used_nodes, error)

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

    def train(self, iterations):
        for i in range(0,iterations):
            if(i%self.reporting_freuquency==0):
                if i!=0:
                    error:float=self.nn.cum_error/self.reporting_freuquency
                else:
                    error=0
                try:
                    with open("logs/training-log.txt") as file:
                        output:str=file.read()
                except:
                    output=""
                output+=f"\nIteration: {i}\tMean square error: {error}\n------------------\n"
                with(open("logs/training-log.txt", "w") as file):
                    file.write(output)
                sys.stdout.write(output)
                self.nn.cum_error=0
            test_point:{}=self.data.get_random_training_data_point()
            self.nn.forward_propagate(test_point['inputs'],int(test_point['label']))

ml=NetworkManager()
ml.train(20000)