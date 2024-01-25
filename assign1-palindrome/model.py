import numpy
from typing import Union
from utils import *

class Palindrome_Model:

    input_size:int = None
    output_size:int = None
    layers:list = []
    weights = None
    learning_rate = 1e-3

    def __init__(self, input_size:int=10, output_size:int = 1, hidden_layer_sizes:list =[], activation: Union[str, list]="relu"):
        self.input_size = input_size
        self.output_size = output_size
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        print(layer_sizes)
        if isinstance(activation, list):
            if len(layer_sizes)!= (len(activation) + 1):
                print("Size of actiavation doesn't match the size of added layers !\nUsing ReLU by default!")
        elif isinstance(activation, str):
            if activation.lower() not in ['relu', 'linear', 'sigmoid']:
                raise NotImplementedError(f"'{activation.upper()}' Activation function is not implemented !!")
            activation = ['relu' for i in range(max(len(layer_sizes)-2,0))] + [activation]   
                                      
        print(activation)
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                {
                "name": f"layer-{i+1}", 
                "weights": np.random.rand(layer_sizes[i]+1, layer_sizes[i+1]),
                "activation": activation[i],
                }  
            )

    def set_optimizer(self, lr):
        self.learning_rate = lr

    def forward(self, input:np.array):
        act_funcs = {
            "relu": Relu(),
            "linear": Linear(),
            "sigmoid": Sigmoid()
        }
        x = input.flatten()
        
        for layer in self.layers:
            x = np.insert(x, 0,1)
            x = np.dot(x, layer["weights"])
            x = act_funcs[layer["activation"]].activate(x)

        return x

    def grad(self):
        pass

        
    def loss(self, x, y):
        pass
