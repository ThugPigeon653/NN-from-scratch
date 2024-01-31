# This model assumes that we are attempting to output a value from the same set
# used as input. For example, if we feed the nn lower-case letters, it will be
# attempting to predict the next low-case letter in the sequence.

# Currently the model does not have bias. training will be unstable juntil bias 
# is added to the network. Also, a relationship is needed between learning rate
# and number of epochs. 

import numpy as np
import random

charset:np.ndarray = np.array([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', '!', '"', '#', '%',
    '&', "'", '(', ')', ',', '-', '.', '/', ':', ';', 
    ' '
])[:, np.newaxis]

class RNN():
    learning_rate:float=0.0001
    @staticmethod
    def softmax(input_arr:np.ndarray)->np.ndarray:
        exp_x = np.exp(input_arr) 
        return exp_x / np.sum(exp_x)
        
    def softmax_prime(self, input_arr: np.ndarray) -> np.ndarray:
        return np.exp(input_arr)/np.sum(np.exp(input_arr))

    def __init__(self) -> None:
        self.x_index:int=len(charset)
        self.bias=[0]*3
        self.u:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_index,self.x_index))
        self.v:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_index,self.x_index))
        self.w:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_index,self.x_index))
        #self.h:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_index, self.x_index))

    @staticmethod
    def loss(input_arr:np.ndarray, predicted_value:np.ndarray)->np.ndarray:
        return 0.5*(input_arr-predicted_value)**2

    @staticmethod
    def loss_prime(input_arr:np.ndarray, predicted_value:np.ndarray)->np.ndarray:
        return input_arr-predicted_value

    def forward_propagate(self, feature:str, epochs:int):
        # Forward propagate
        y:np.ndarray=None
        a:np.ndarray
        h:np.ndarray=np.zeros(shape=(1, self.x_index))
        y:np.ndarray=None
        z:np.ndarray=None
        c:np.ndarray=None
        old_x:np.ndarray=np.zeros(shape=(1,self.x_index))
        dL_dy:np.ndarray=np.zeros(shape=(1,self.x_index))
        count=0
        s=0
        for char in feature:
                # Encode one-hot
                x:np.ndarray=np.zeros(shape=(1, self.x_index))
                char=char.lower()
                hot_index:int=int(np.where(charset==char)[0][0])
                x[0][hot_index]=1
                # loss performed after discovering the correct next feature. This is
                # effectively calculating retrospective loss of previous iteration. 
                if y is not None: 
                    dL_du=np.outer((x-y).dot(self.v.T),dh_du)
                    dL_dw=np.outer((x-y).dot(self.v.T),dh_dw)
                    dL_dv=np.outer((x-y),(h))

                    self.u-=dL_du*self.learning_rate
                    self.w-=dL_dw*self.learning_rate
                    self.v-=dL_dv*self.learning_rate

                    s+=np.sum(self.loss(x,y))
                a=x.dot(self.u)
                z=h.dot(self.w)
                b=z+a
                h=np.tanh(b)
                dh_du=(1-h)*(x+(old_x.dot(self.u.T)))
                dh_dw=(1-h)*(x+(h.dot(self.w.T)))
                c=h.dot(self.v)
                y=self.softmax(c)
                old_x=x
                count+=1
        print(s/len(feature))


    def make_prediction(self, prediction_length:int, output_file_index:int, start_seed:str=""):
        if start_seed==None or start_seed=="":
            start_seed="The words here dont really matter, they just provide a small sample of text."
        y:np.ndarray=None
        a:np.ndarray
        output:str=""
        h:np.ndarray=np.zeros(shape=(1, self.x_index))
        for char in start_seed:
            x:np.ndarray=np.zeros(shape=(1, self.x_index))
            char=char.lower()
            hot_index:int=int(np.where(charset==char)[0][0])
            x[0][hot_index]=1
            a=x.dot(self.u)
            z=h.dot(self.w)
            b=z+a
            h=np.tanh(b)
            c=h.dot(self.v)
            y=self.softmax(c)

        for _ in range(0, prediction_length):
            x:np.ndarray=np.zeros(shape=(1, self.x_index))
            char=char.lower()
            hot_index:int=int(np.where(charset==char)[0][0])
            x[0][hot_index]=1
            a=x.dot(self.u)
            z=h.dot(self.w)
            b=z+a
            h=np.tanh(b)
            c=h.dot(self.v)
            y=self.softmax(c)
            output+=str(charset[np.argmax(y)][0])
            #print(np.argmax(y), str(charset[np.argmax(y)][0]))
            if len(output)%80==0:
                output+='\n'
        with open(f"output_{output_file_index}.txt", "w") as file:
            file.write(output)

def sample_implementation():
    # sample usage
    with open("sample.txt", "r") as file:
        feature=file.read()
        feature=feature.replace("\n", " ").replace("\t", " ")

    cleaned_feature=""

    while cleaned_feature!=feature:
        if cleaned_feature!="":
            feature=cleaned_feature
        cleaned_feature=feature.replace("  ", " ")
        
    feature=cleaned_feature
    cleaned_feature=""

    for char in feature:
        if char in charset:
            cleaned_feature+=char

    feature=cleaned_feature

    model=RNN()
    i=0
    iterations=2000
    seq_len:int=40
    while i<iterations:
        j=random.randint(0,len(feature)-seq_len)
        sub_feature:str=feature[j:j+seq_len]
        model.forward_propagate(sub_feature, 1)
        if i%(iterations/10)==0:
            model.make_prediction(3000, i)
        i+=1

sample_implementation()