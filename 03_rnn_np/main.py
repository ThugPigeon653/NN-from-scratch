# This model assumes that we are attempting to output a value from the same set
# used as input. For example, if we feed the nn lower-case letters, it will be
# attempting to predict the next low-case letter in the sequence.

# Currently the model does not have bias. training will be unstable juntil bias 
# is added to the network. Also, a relationship is needed between learning rate
# and number of epochs. 

import numpy as np
import sys

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
    def loss_prime(input_arr:np.ndarray, predicted_value:np.ndarray)->np.ndarray:
        return input_arr/np.where(predicted_value == 0, 1e-10, predicted_value)

    def forward_propagate(self, feature:str, epochs:int):
        # Forward propagate
        y:np.ndarray=None
        a:np.ndarray
        h:np.ndarray=np.zeros(shape=(1, self.x_index))
        y:np.ndarray=None
        z:np.ndarray=None
        c:np.ndarray=None
        old_x:np.ndarray
        for char in feature:
                # Encode one-hot
                x:np.ndarray=np.zeros(shape=(1, self.x_index))
                char=char.lower()
                hot_index:int=int(np.where(charset==char)[0][0])
                x[0][hot_index]=1
                # loss performed after discovering the correct next feature. This is
                # effectively calculating retrospective loss of previous iteration. 
                if y is not None: 
                    L=(x*-1)*np.log(y)
                    dL_dy=(-1*x)/y
                    dy_dc=self.softmax_prime(c)
                    dc_dv=h
                    dc_dh=self.v
                    dh_db=np.square(np.ones(shape=(b.shape))/np.cosh(b))
                    # db_dz=db_da=1 // this is left as comment, to remind that this
                    # partial derivative has a gradient of 1 - so does not need to 
                    # accounted for
                    da_du=old_x
                    da_dw=h

                    dL_db=(dL_dy*dy_dc).dot(dc_dh.T)*dh_db

                    dL_du=np.outer(dL_db, da_du).T
                    dL_dw=np.outer(dL_db, da_dw).T
                    dL_dv=np.outer((dL_dy*dy_dc),(dc_dv)).T

                    self.u-=dL_du*self.learning_rate
                    self.v-=dL_dv*self.learning_rate
                    self.w-=dL_dw*self.learning_rate
                    
                a=x.dot(self.u)
                z=h.dot(self.w)
                b=z+a
                h=np.tanh(b)
                c=h.dot(self.v)
                y=self.softmax(c)
                old_x=x




    '''def forward_propagate(self, feature:str, epochs:int):
        # Forward propagate
        y:np.ndarray=None
        a:np.ndarray
        i=0
        learning_rate:float=self.learning_rate/(epochs*len(feature))
        while i<epochs:    
            for char in feature:
                # Encode one-hot
                x:np.ndarray=np.zeros(shape=(self.x_index, 1))
                char=char.lower()
                hot_index:int=int(np.where(charset==char)[0][0])
                x[hot_index][0]=1
                # BPTT is handled for each forward layer
                if y is not None:
                    dl_dy:np.ndarray=self.loss_prime(y,x)
                    dy_dc:np.ndarray=self.softmax_prime(y)
                    dc_dv:np.ndarray=self.h
                    vb_nabla=(dl_dy*dy_dc).T
                    v_nabla=vb_nabla.dot(dc_dv)
                    dc_dh=self.v
                    dh_da=np.ones(shape=(self.h.shape))-np.square(np.tanh(self.h.dot(self.v)))
                    da_du=x
                    ub_nabla=(dl_dy*dy_dc).T.dot(dc_dh).dot(dh_da)
                    u_nabla=ub_nabla*(da_du.T)
                    # da_dz=1
                    dz_dw=self.h
                    wb_nabla=(dl_dy*dy_dc).T.dot(dc_dh).dot(dh_da)
                    w_nabla=wb_nabla.dot(dz_dw).T
                    wb_nabla=wb_nabla.T
                    self.u-=u_nabla*learning_rate
                    self.w-=w_nabla*learning_rate
                    self.v-=v_nabla*learning_rate

                    self.bias[0]-=np.sum(ub_nabla)*learning_rate
                    self.bias[1]-=np.sum(vb_nabla)*learning_rate
                    self.bias[2]-=np.sum(wb_nabla)*learning_rate
                a=(x.dot(self.u)+self.bias[0])+(self.h.dot(self.v)+self.bias[1])
                self.h:np.ndarray=np.tanh(a)
                z:np.ndarray=self.h.dot(self.w)+self.bias[2]
                y=self.softmax(z)
                i+=1'''

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
            print(np.argmax(y), str(charset[np.argmax(y)][0]))
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
    iterations=20
    while i<20:
        model.forward_propagate(feature, 1)
        model.make_prediction(3000, i)
        i+=1