import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class Cell:

    def __init__(
            self,
            short_array:np.ndarray,
            input_array:np.ndarray,
            long_array:np.ndarray,
            params:np.ndarray=None
            
            ):

        if params is None:
            params = np.array([
                np.random.randn(4),
                np.random.randn(4),
                np.random.randn(4)
            ])

        
        self.params=params

        self.short = short_array.astype(float)
        self.input = input_array.astype(float)
        self.long = long_array.astype(float)
        self.context_vector = np.nan





    def forward(self):

        ws,wi,b=self.params[0],self.params[1],self.params[2]

        x = sigmoid(self.short * ws[0] + self.input * wi[0] + b[0])
        self.long = self.long * x

        x = sigmoid(self.short * ws[1] + self.input * wi[1] + b[1]) * tanh(self.short * ws[2] + self.input * wi[2] + b[2])
        self.long = self.long + x

        x = sigmoid(self.short * ws[3] + self.input * wi[3] + b[3])
        self.short = tanh(self.long) * x

        self.context_vector = (self.long, self.short)

        return self.context_vector




params=[np.array([2.7,2,1.41,4.38]),np.array([1.62,1.65,0.94,-0.19]),np.array([1.62,.62,.32,.59])]
short=np.array([1,2])
input=np.array([1,2])
long=np.array([2,2])
cell=Cell(short_array=short,input_array=input,long_array=long,params=params)
cell.forward()
print(cell.context_vector)

