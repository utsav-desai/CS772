import numpy
from utils import *

class Model:
    input_size:int = None
    output_size:int = None
    layers:list = None

    def __init__(self) -> None:
        pass

    def forward(self):
        pass

    def backward(self):
        pass


class Palindrome_Model(Model):
    super.__init__()

    def __init__(self, input_size:int=10, output_size:int = 1, hidden_layer_size:list =None):
        self.input_size = input_size
        self.output_size = output_size
        np.random.rand(input_size, hidden_layer_size)
        self.layers.append()

        for i in range(len(hidden_layer_size)):
            self.layers.append()

        
