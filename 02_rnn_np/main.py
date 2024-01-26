# This model assumes that we are attempting to output a value from the same set
# used as input. For example, if we feed the nn lower-case letters, it will be
# attempting to predict the next low-case letter in the sequence.
import numpy as np

class RNN():

    @staticmethod
    def softmax(input_arr:np.ndarray)->np.ndarray:
        exp_x = np.exp(input_arr - np.max(input_arr)) 
        return exp_x / np.sum(exp_x, axis=0)
    
    def softmax_prime(self, input_arr: np.ndarray) -> np.ndarray:
        s: np.ndarray = self.softmax(input_arr)
        return np.diagflat(s) - np.outer(s, s)

    def __init__(self, chars) -> None:
        self.x_argmax:int=len(chars)
        self.u:np.ndarray=np.random.uniform(-1.000, 1.000, size=(1,self.x_argmax))
        self.v:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_argmax,self.x_argmax))
        self.w:np.ndarray=np.random.uniform(-1.000, 1.000, size=(self.x_argmax,1))
        self.h:np.ndarray=np.zeros(shape=(self.x_argmax, self.x_argmax))

    def forward_propagate(self, feature:np.ndarray):
        # Forward propagate
        for char in feature:
            # Encode one-hot
            x:np.ndarray=np.zeros(shape=(self.x_argmax, 1))
            char=char.lower()
            hot_index:int=int(np.where(charset==char)[0][0])
            x[hot_index][0]=1
            self.h:np.ndarray=np.tanh(x.dot(self.u)+(self.h.dot(self.v)))
            z:np.ndarray=self.h.dot(self.w)
            y:np.ndarray=self.softmax(z)
            print((y).shape)

            
            dy_dw = self.h.T.dot(self.softmax_prime(z))
            dy_dv = self.h.T.dot(self.softmax_prime(z)).dot(self.u.T)
            dy_du = x.T.dot(self.v).dot(self.softmax_prime(z)) * (1 - self.h**2)




            print(self.w.shape, dy_dw.shape)
            print(self.v.shape, dy_dv.shape)
            print(self.u.shape, dy_du.shape)
            print(charset[np.argmax(y)][0])

# sample usage
feature='The words here dont really matter, they just provide a small sample of text.'
charset:np.ndarray = np.array([
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
    'u', 'v', 'w', 'x', 'y', 'z', '!', '"', '#', '%',
    '&', "'", '(', ')', ',', '-', '.', '/', ':', ';', 
    ' '
])[:, np.newaxis]

model=RNN(charset)
model.forward_propagate(feature)