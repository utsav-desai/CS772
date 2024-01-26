import numpy as np
import os
import pickle
from typing import Union
from utils.activation import *
from utils.losses import *
from tqdm import tqdm
np.random.seed(69)


class Palindrome_Model:

    input_size:int = None
    output_size:int = None
    layers:list = []
    weights = None
    learning_rate = 1e-3
    loss_metric = None

    loss_funcs = {
        "mse": MSE(),
        "bce": BCE()
    }
    act_funcs = {
        "relu": Relu(),
        "linear": Linear(),
        "sigmoid": Sigmoid()
    }

    def __init__(self, input_size:int=10, output_size:int = 1, hidden_layer_sizes:list =[], activation: Union[str, list]="linear"):
        self.input_size = input_size
        self.output_size = output_size
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]

        if isinstance(activation, list):
            if len(layer_sizes)!= (len(activation) + 1):
                print("Size of actiavation doesn't match the size of added layers !\nUsing ReLU by default!")

        elif isinstance(activation, str):
            if activation.lower() not in ['relu', 'linear', 'sigmoid']:
                raise NotImplementedError(f"'{activation.upper()}' Activation function is not implemented !!")
            activation = ['linear' for i in range(max(len(layer_sizes)-2,0))] + [activation]   
            
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                {
                "name": f"layer-{i+1}", 
                "weights": np.random.normal(scale=0.5, size=(layer_sizes[i], layer_sizes[i+1])),
                "biases": np.random.normal(scale=1, size=(layer_sizes[i+1])),
                "activation": activation[i],
                }  
            )

    def set_optimizer(self, lr:float = 5e-3, loss:str = "mse"):
        self.learning_rate = lr
        self.loss_metric = self.loss_funcs[loss]

    def forward(self, input:np.array):

        x = input.flatten()
        for i in range(len(self.layers)):
            x = np.dot(x, self.layers[i]["weights"]) + self.layer[i]['biases']
            x = self.act_funcs[self.layers[i]["activation"]].activate(x)

        return x

    def predict(self, X):
        predictions = []
        for x in X:
            y = self.forward(x)
            if y >=0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)


    def backward(self, input, target):
        if self.loss_metric ==None:
            self.set_optimizer()
        act_funcs = {
            "relu": Relu(),
            "linear": Linear(),
            "sigmoid": Sigmoid()
        }
        # Initialize gradients
        weight_gradients = [np.zeros_like(layer["weights"]) for layer in self.layers]
        bias_gradients = [np.zeros_like(layer['biases']) for layer in self.layers]

        x = input.flatten()
        net = [x]               ## Same notation as used in slides, represents the output of layer
        o = []
        for layer in self.layers:
            x = np.dot(x, layer["weights"]) + layer['biases']
            net.append(x)
            x = act_funcs[layer["activation"]].activate(x)
            o.append(x)

        loss_gradient = self.loss_metric.grad(o[-1], target)
        
        for i in range(len(self.layers) - 1, -1, -1):
            # Compute gradients for the current layer
            activation_gradient = act_funcs[self.layers[i]["activation"]].grad(net[i + 1])
            weight_gradients[i] = np.outer(net[i], loss_gradient * activation_gradient)
            bias_gradients[i] = loss_gradient * activation_gradient

            # Compute loss gradient for the next layer in the backward pass
            loss_gradient = np.dot(loss_gradient * activation_gradient, self.layers[i]["weights"].T)

        # Update weights and biases using gradients and learning rate
        for i in range(len(self.layers)):
            self.layers[i]["weights"] -= self.learning_rate * weight_gradients[i]
            self.layers[i]["biases"] -= self.learning_rate * bias_gradients[i]

    def train(self, X_train, y_train, epochs=10):
        accuracies, losses = [], []
        for epoch in range(epochs):
            total_loss = 0.0
            with tqdm(total=len(X_train), desc=f"Epoch {epoch + 1}/{epochs}", unit="sample") as pbar:
                for i in range(len(X_train)):
                    input_sample = X_train[i]
                    target = y_train[i]
                    predicted = self.forward(input_sample)
                    self.backward(input_sample, target)
                    loss = self.loss(predicted, target)
                    total_loss += loss
                    # pbar.set_postfix(loss=loss)
                    pbar.update()
                accuracy  = accuracy_metric(self.predict(X_train), y_train)
                pbar.set_postfix(loss=total_loss,accuracy=accuracy)
                accuracies.append(accuracy)
                losses.append(total_loss)
        return accuracies, losses


    def loss(self, predicted, target):
        return self.loss_metric.loss(predicted, target)
        
    def save(self, path="./weights"):
        with open( os.path.join('./weights/', 'layers.pkl'), 'wb') as file:
            pickle.dump(self.layers, file)

    def load(self, path):
        with open(path, 'rb') as file:
            self.layers = pickle.load(file)
