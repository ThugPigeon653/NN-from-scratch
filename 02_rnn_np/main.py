# This model assumes that we are attempting to output a value from the same set
# used as input. For example, if we feed the nn lower-case letters, it will be
# attempting to predict the next low-case letter in the sequence.

# Currently the model does not have bias. training will be unstable juntil bias 
# is added to the network. Also, a relationship is needed between learning rate
# and number of epochs. 

import numpy as np

class RNN():
    learning_rate:float=0.0000000000001
    @staticmethod
    def softmax(input_arr:np.ndarray)->np.ndarray:
        exp_x = np.exp(input_arr - np.max(input_arr)) 
        return exp_x / np.sum(exp_x, axis=0)
    
    def softmax_prime(self, input_arr: np.ndarray) -> np.ndarray:
        return np.exp(input_arr)/np.sum(np.exp(input_arr))

    def __init__(self, chars) -> None:
        self.x_argmax:int=len(chars)
        self.u:np.ndarray=np.random.uniform(-1.000, 1.000, size=(1,self.x_argmax))
        self.v:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_argmax,self.x_argmax))
        self.w:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_argmax,1))
        self.h:np.ndarray=np.zeros(shape=(self.x_argmax, self.x_argmax))

    @staticmethod
    def loss_prime(input_arr:np.ndarray, predicted_value:np.ndarray)->np.ndarray:
        return input_arr/np.where(predicted_value == 0, 1e-10, predicted_value)

    def forward_propagate(self, feature:str, epochs:int):
        # Forward propagate
        y:np.ndarray=None
        a:np.ndarray
        i=0
        while i<epochs:    
            for char in feature:
                # Encode one-hot
                x:np.ndarray=np.zeros(shape=(self.x_argmax, 1))
                char=char.lower()
                hot_index:int=int(np.where(charset==char)[0][0])
                x[hot_index][0]=1
                # BPTT is handled for each forward layer
                if y is not None:
                    dl_dy:np.ndarray=self.loss_prime(y,x)
                    dy_dc:np.ndarray=self.softmax_prime(y)
                    dc_dv:np.ndarray=self.h
                    v_nabla=(dl_dy*dy_dc).T.dot(dc_dv)
                    dc_dh=self.v
                    dh_da=np.ones(shape=(self.h.shape))-np.square(np.tanh(self.h.dot(self.v)))
                    da_du=x
                    u_nabla=(dl_dy*dy_dc).T.dot(dc_dh)
                    da_dz=1
                    dz_dw=self.h
                    w_nabla=(dl_dy*dy_dc).T.dot(dh_da).dot(dz_dw).T
                    self.u-=u_nabla*self.learning_rate
                    self.w-=w_nabla*self.learning_rate
                    self.v-=v_nabla*self.learning_rate
                a=x.dot(self.u)+(self.h.dot(self.v))
                self.h:np.ndarray=np.tanh(a)
                z:np.ndarray=self.h.dot(self.w)
                y=self.softmax(z)
                i+=1

    def make_prediction(self, prediction_length:int, start_seed:str=""):
        if start_seed==None or start_seed=="":
            start_seed="The words here dont really matter, they just provide a small sample of text."
        y:np.ndarray=None
        a:np.ndarray
        output:str=""
        for char in start_seed:
            # Encode one-hot
            x:np.ndarray=np.zeros(shape=(self.x_argmax, 1))
            char=char.lower()
            hot_index:int=int(np.where(charset==char)[0][0])
            x[hot_index][0]=1
            a=x.dot(self.u)+(self.h.dot(self.v))
            self.h:np.ndarray=np.tanh(a)
            z:np.ndarray=self.h.dot(self.w)
            y=self.softmax(z)
        for i in range(0, prediction_length):
            # Encode one-hot
            x:np.ndarray=np.zeros(shape=(self.x_argmax, 1))
            char=char.lower()
            hot_index:int=np.argmax(y)
            x[hot_index][0]=1
            a=x.dot(self.u)+(self.h.dot(self.v))
            self.h:np.ndarray=np.tanh(a)
            z:np.ndarray=self.h.dot(self.w)
            y=self.softmax(z)
            output+=charset[np.argmax(y)][0]
            if len(output)%80==0:
                output+='\n'
        with open("output.txt", "w") as file:
            file.write(output)

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

charset:np.ndarray = np.array([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', '!', '"', '#', '%',
    '&', "'", '(', ')', ',', '-', '.', '/', ':', ';', 
    ' '
])[:, np.newaxis]

for char in feature:
    if char in charset:
        cleaned_feature+=char

feature=cleaned_feature

model=RNN(charset)
model.forward_propagate(feature, 3000)
model.make_prediction(1000)