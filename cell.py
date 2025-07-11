import numpy as np


class Cell:

    def __init__(
            self,
            gates:np.ndarray|np.ndarray,
            input_array:np.ndarray,
            params:np.ndarray=None
            ):

        if params is None:
            params = np.array([
                np.random.randn(4),
                np.random.randn(4),
                np.random.randn(4)
            ])
        short_array=gates[0]
        long_array=gates[1]
        
        self.params=params

        self.short = short_array.astype(float)
        self.input = input_array.astype(float)
        self.long = long_array.astype(float)
        self.context_vector = np.nan


    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def tanh(self,x):
        return np.tanh(x)


    def forward(self):

        ws,wi,b=self.params[0],self.params[1],self.params[2]
        long=self.long
        short=self.short
        x = self.sigmoid(self.short * ws[0] + self.input * wi[0] + b[0])
        long = long * x

        x = self.sigmoid(self.short * ws[1] + self.input * wi[1] + b[1]) * self.tanh(self.short * ws[2] + self.input * wi[2] + b[2])
        long = long + x

        x = self.sigmoid(self.short * ws[3] + self.input * wi[3] + b[3])
        short = self.tanh(long) * x

        self.context_vector = (short,long)

        return self.context_vector

    def update(self,params_new):
        self.params=params_new
        self.forward()

        




