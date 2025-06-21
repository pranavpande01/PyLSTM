import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class Cell:

    def __init__(self,short_array,input_array,long_array,params:np.ndarray=None):

        if params is None:
            params = np.array([
                np.random.randn(4),
                np.random.randn(4),
                np.random.randn(4)
            ])

        
        self.params=params
        self.ws=params[0]
        self.wi=params[1]
        self.b=params[2]

        self.long=long_array
        self.input=input_array
        self.short=short_array
        self.context_vector=np.nan






    def forward(self):
        x=self.short*self.ws[0]+self.input*self.wi[0]+self.b[0]
        x=sigmoid(x)
        self.long=self.long*x

        x=sigmoid(self.short*self.ws[1]+self.input*self.wi[1]+self.b[1])*tanh(self.short*self.ws[2]+self.input*self.wi[2]+self.b[2])
        self.long=self.long+x

        x=sigmoid(self.short*self.ws[3]+self.input*self.wi[3]+self.b[3])
        x=tanh(self.long)*x
        self.short=x
        self.context_vector=self.long,self.short
       
        self.context_vector




params=[np.array([2.7,2,1.41,4.38]),np.array([1.62,1.65,0.94,-0.19]),np.array([1.62,.62,.32,.59])]
short=np.array([1,2])
input=np.array([1,2])
long=np.array([2,2])
cell=Cell(short_array=short,input_array=input,long_array=long)
cell.forward()
print(cell.context_vector)