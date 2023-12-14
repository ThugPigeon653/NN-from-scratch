import sqlite3
import random
import uuid
import sys
from input_data import DataManager

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
        is_first_layer:bool=True
        for layer in range(0,layer_count+1):
            self.cursor.execute(f'INSERT INTO layer VALUES(?,?)', (layer,activation_type))
            #print("ADDING layer")
            layer_size_for_layer:int
            if layer==0:
                layer_size_for_layer=input_size
            else:
                layer_size_for_layer=layer_size
            for node in range(0,layer_size_for_layer):
                self.cursor.execute(f'INSERT INTO node VALUES(?,?)', (layer,node))
                #print("ADDING node")
                if self.cursor.rowcount==0:
                    raise Exception
                if is_first_layer==True:
                    layer_input_size:int=input_size
                else:
                    layer_input_size=layer_size
                for fromNode in range(0,layer_input_size):
                    weight:float=random.uniform(-1.000, 1.000)
                    self.cursor.execute(f'INSERT INTO weights VALUES(?,?,?,?)',(node,fromNode,layer,weight))
                    #print("ADDING weight")
                    self.conn.commit()
                    if self.cursor.rowcount==0:
                        raise Exception
                is_first_layer=False
        # add the output layer
        
        self.cursor.execute(f'INSERT INTO layer VALUES(?,?)', (layer_size+1,activation_type))
        for i in range(0,output_size):
            self.cursor.execute(f'INSERT INTO node VALUES(?,?)', (layer_count+1,i))
            for j in range(0,layer_size):
                    self.cursor.execute(f'INSERT INTO weights VALUES(?,?,?,?)',(i,j,layer_count+1,random.uniform(-1.000,1.000)))

                
        for node in range(0, layer_size):
            for output in range(0, output_size):
                weight:float=random.uniform(-1.000, 1.000)
                self.cursor.execute('INSERT INTO weights VALUES(?,?,?,?)', (output, node, layer_count, weight))
                if self.cursor.rowcount==0:
                    raise Exception

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
        else:
            output=input
        return output

    @staticmethod
    def calculate_error(predicted:int, actual:int):
        return predicted-actual
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
        normalized_values = [0.500+((value - translation) / dilation) for value in values]
        return normalized_values

    def back_propagate(self, weight_keys:[], error:int):
        adjusted_error:float=self.learning_rate*float(error)
        self.cum_error+=adjusted_error
        for key in weight_keys:
            self.cursor.execute('UPDATE weights SET weight = weight - ? WHERE toNodeId=? AND fromNodeId=? AND layerId=?', (adjusted_error, key[0], key[1], key[2]))

    # Classification algorithm, reducing many floats to one int 
    def forward_propagate(self, input_data:list[float], expected_result:int, is_training:bool=True):
        #print(f"len: {len(input_data)}")
        weights_used:list=[]
        current_layer:list[float]=[]
        layer_cache:list[float]
        k=0
        # for each layer
        while k<=self.layer_count+1:
            current_layer=[]
            # fill current layer as empty
            for i in range(0,self.layer_size):
                current_layer.append(0)
            # For the first layer, we should only make the layer the size of input. After that, the size of hidden layers.
            if k==0:
                layer_size=len(input_data)
                layer_cache=input_data
            else:
                layer_size=self.layer_size
                layer_cache=current_layer
            #print('\t', k, self.layer_count-1)
            if(k==self.layer_count-1):
                output_size:int=self.output_size
                #print(i, len(input_data))
                print("VERY LAST VERY LAST VERY LAST VERY LAST VERY LAST VERY LAST VERY LAST VERY LAST ")
            else:
                #print(f"{i} is not equal to {output_size}")
                output_size=self.layer_size
            i=0
            # For every node in this layer
            while i < layer_size:
                j=0
                # Output size will always match hidden layer size, except do the output which matches our intended output range

                #print(f'\t\t{self.output_size}')
                # For all possible outputs
                #print(output_size)
                while(j<output_size):
                    # get weight for this input-output combo
                    self.cursor.execute('SELECT weight, fromNodeId, toNodeId, layerId FROM weights WHERE fromNodeId=? AND toNodeId=?', (i,j))
                    weight, fromNodeId, toNodeId, layerId=self.cursor.fetchone()
                    # add input by weight to this node
                    adjustment:float=layer_cache[i]*weight
                    current_layer[j]+=adjustment
                    if adjustment>0:
                        weights_used.append((fromNodeId, toNodeId, layerId))
                    j+=1
                i+=1
            # normalize, then activate here
            current_layer=self.normalize_list(current_layer)
            for layer_index in range(0,len(current_layer)):
                current_layer[layer_index]=self.activation_relu(current_layer[layer_index])
            layer_cache=current_layer
            k+=1
        result:int=current_layer.index(max(current_layer))
        error=self.calculate_error(result, expected_result)
        if(is_training):
            self.back_propagate(weights_used, error)

class NetworkManager():
    reporting_freuquency:int=100

    def __init__(self) -> None:
        self.data:DataManager=DataManager()
        self.nn:Network=Network(
            layer_count=3, 
            layer_size=64, 
            input_size=784, 
            output_size=10, 
            activation_type=0, 
            learning_rate=0.05)

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

ml=NetworkManager()
ml.train(20000)