import numpy as np
import os
import pickle
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

        # print(layer_sizes)

        if isinstance(activation, list):
            if len(layer_sizes)!= (len(activation) + 1):
                print("Size of actiavation doesn't match the size of added layers !\nUsing ReLU by default!")

        elif isinstance(activation, str):
            if activation.lower() not in ['relu', 'linear', 'sigmoid']:
                raise NotImplementedError(f"'{activation.upper()}' Activation function is not implemented !!")
            activation = ['relu' for i in range(max(len(layer_sizes)-2,0))] + [activation]   

        # print(activation)
            
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                {
                "name": f"layer-{i+1}", 
                "weights": np.random.normal(scale=0.5, size=(layer_sizes[i], layer_sizes[i+1])),
                "biases": np.random.normal(scale=1, size=(layer_sizes[i+1])),
                "activation": activation[i],
                }  
            )

    def set_optimizer(self, lr:float, loss:str="mse"):
        self.learning_rate = lr
        self.loss = loss

    def forward(self, input:np.array):
        act_funcs = {
            "relu": Relu(),
            "linear": Linear(),
            "sigmoid": Sigmoid()
        }
        x = input.flatten()
        
        for layer in self.layers:
            # x = np.insert(x, 0,1)
            x = np.dot(x, layer["weights"]) + layer['biases']
            x = act_funcs[layer["activation"]].activate(x)
        return x

    def backward(self, input, target):
        act_funcs = {
            "relu": Relu(),
            "linear": Linear(),
            "sigmoid": Sigmoid()
        }
        # Initialize gradients
        gradients = [np.zeros_like(layer["weights"]) for layer in self.layers]

        # Forward pass to compute intermediate values
        x = input.flatten()
        intermediates = [x]
        for layer in self.layers:
            x = np.insert(x, 0, 1)
            intermediates.append(x)
            x = np.dot(x, layer["weights"])
            x = act_funcs[layer["activation"]].activate(x)

        # Compute loss gradient
        loss_gradient = x - target  # Assuming mean squared error loss

        # Backward pass to update gradients
        for i in range(len(self.layers) - 1, -1, -1):
            # Compute gradients for the current layer
            activation_gradient = act_funcs[self.layers[i]["activation"]].grad(intermediates[i + 1])
            gradients[i] = np.outer(intermediates[i], loss_gradient * activation_gradient)

            # Compute loss gradient for the next layer in the backward pass
            loss_gradient = np.dot(loss_gradient * activation_gradient, self.layers[i]["weights"][1:].T)

        # Update weights using gradients and learning rate
        for i in range(len(self.layers)):
            self.layers[i]["weights"] -= self.learning_rate * gradients[i]

        

    def save(self, path="./weights"):
        with open( os.path.join('./weights/', 'layers.pkl'), 'wb') as file:
            pickle.dump(self.layers, file)

    def load(self, path):
        with open(path, 'rb') as file:
            self.layers = pickle.load(file)
