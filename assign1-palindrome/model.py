import numpy as np
import os
import pickle
from typing import Union
from utils.activation import *
from utils.losses import *
from tqdm import tqdm


optimization_methods = {
    'sgd',  # Stochastic Gradient Descent
    'bgd',  # Batch Gradient Descent
    'mbgd'  # Mini-Batch Gradient Descent
    }

class Palindrome_Model:

    input_size:int = None
    output_size:int = None
    layers:list = []
    weights = None
    learning_rate = 1e-3
    loss_metric = None
    weight_gradients=None
    bias_gradients=None

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

        # if isinstance(activation, list):
        #     if len(layer_sizes)!= (len(activation) + 1):
        #         print("Size of actiavation doesn't match the size of added layers !\nUsing linear by default!")
                
        if isinstance(activation, str):
            if activation.lower() not in ['relu', 'linear', 'sigmoid']:
                raise NotImplementedError(f"'{activation.upper()}' Activation function is not implemented !!")
            activation = ['linear' for i in range(max(len(layer_sizes)-2,0))] + [activation.lower()]   
            
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                {
                "name": f"layer-{i+1}", 
                "weights": np.random.normal(scale=0.5, size=(layer_sizes[i], layer_sizes[i+1])),
                "biases": np.random.normal(scale=1, size=(layer_sizes[i+1])),
                "activation": activation[i],
                }  
            )

    def set_optimizer(self, lr:float = 5e-3, loss:str = "mse",optimization_method="bgd"):
        self.learning_rate = lr
        self.loss_metric = self.loss_funcs[loss]
        if optimization_method.lower() in optimization_methods:
            self.optimization_method = optimization_method
        else:
            self.optimization_method='mbgd'


    def forward(self, inputs:Union[list, np.ndarray]):
        o = []
        for x in inputs:
            x = x.flatten()
            for i in range(len(self.layers)):
                x = np.dot(x, self.layers[i]["weights"]) + self.layers[i]['biases']
                x = self.act_funcs[self.layers[i]["activation"]].activate(x)
            o.append(x)
        return np.array(o)
    
    def __call__(self,inputs):
        return self.forward(inputs)

    def predict(self, X):
        predictions = []
        for x in X:
            y = self.forward([x])
            if y[0] >=0.5:
                predictions.append(1)
            else:
                predictions.append(0)
        return np.array(predictions)


    def backward(self, inputs, targets):
        """
        Same notation are used as in slides, 
        'x'     : represent input
        'net'   : represents the output of layer without activation
        'o'     : represents the output after activation  
        """
        if self.loss_metric ==None:
            self.set_optimizer()
        # Initialize gradients
        # print("Removing gradients...")
        self.weight_gradients = [np.zeros_like(layer["weights"]) for layer in self.layers]
        self.bias_gradients = [np.zeros_like(layer['biases']) for layer in self.layers]
        total_loss =0.0

        for input, target in zip(inputs, targets):
            x = input.flatten()
            net, o = [x], [x]
            for i in range(len(self.layers)):
                x = np.dot(x, self.layers[i]["weights"]) + self.layers[i]['biases']
                net.append(x)
                x = self.act_funcs[self.layers[i]["activation"]].activate(x)
                o.append(x)

            loss = self.loss(x, target)
            total_loss += loss
            
            loss_gradient = self.loss_metric.grad(x, target)                        # ∂L/∂o
            
            for i in range(len(self.layers) - 1, -1, -1):
                # Compute gradients for the current layer
                activation_gradient = self.act_funcs[self.layers[i]["activation"]].grad(net[i + 1])
                self.weight_gradients[i] += np.outer(o[i], loss_gradient * activation_gradient)
                self.bias_gradients[i] += loss_gradient * activation_gradient

                # Compute loss gradient for the next layer in the backward pass
                loss_gradient = np.dot(loss_gradient * activation_gradient, self.layers[i]["weights"].T)

            for i in range(len(self.layers)):
                self.layers[i]["weights"] -= self.learning_rate * (self.weight_gradients[i]/len(inputs))
                self.layers[i]["biases"] -= self.learning_rate * (self.bias_gradients[i]/len(inputs))

    
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        accuracies, losses = [], []
        for epoch in range(epochs):
            total_loss = 0.0
            with tqdm(total=len(X_train), desc=f"Epoch {epoch + 1}/{epochs}", unit="sample") as pbar:

                if self.optimization_method=='mbgd':
                    x = 5

                elif self.optimization_method=='bgd':
                    for i in range(0, len(X_train), batch_size):
                        input_samples = X_train[i: i+batch_size]
                        targets = y_train[i:i+batch_size]
                        predicted = self.forward(input_samples)
                        batch_loss = self.loss(predicted, targets)                    
                        total_loss += batch_loss
                        self.backward(input_samples, targets)
                        pbar.update(batch_size)
                else:
                    x = 5
               
                accuracy = accuracy_metric(self.predict(X_train), y_train)
                pbar.set_postfix(loss=total_loss, accuracy=accuracy)
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
