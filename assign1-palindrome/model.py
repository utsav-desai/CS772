import numpy as np
import os
import pickle
from typing import Union
from utils.activation import *
from utils.losses import *
from tqdm import tqdm
from utils.optimizer import *


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

    def __init__(self, input_size:int=10, output_size:int = 1, hidden_layer_sizes:list =[], activation: Union[str, list]="relu"):
        """
        Input Arguements:

        hidden_layer_sizes : List of sizes of hidden layers. For eg, [2,3] corresponds to 2 hidden layers of sizes 
        2 and 3 nodes in each layer respectively.
        activation: This arguements takes two types of objects a list and a string. If a string is provided only last layer will have activation 
        of the provided string, if you want custom activations for each layer, provide a list of strings.     
        """
        self.input_size = input_size
        self.output_size = output_size
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]     # List of no of nodes in layers of the network 

        if isinstance(activation, list):
            if len(layer_sizes)!= (len(activation) + 1):
                print("Size of actiavation doesn't match the size of added layers !\nUsing relu by default!")
            activation = ['relu' for i in range(max(len(layer_sizes)-1,0))]
                
        if isinstance(activation, str):
            if activation.lower() not in ['relu', 'linear', 'sigmoid']:
                raise NotImplementedError(f"'{activation.upper()}' Activation function is not implemented !!")
            activation = ['relu' for i in range(max(len(layer_sizes)-2,0))] + [activation.lower()]   
            
        for i in range(len(layer_sizes)-1):
            self.layers.append(
                {
                "name": f"layer-{i+1}", 
                "weights": np.random.rand(layer_sizes[i], layer_sizes[i+1]),
                "biases": np.random.rand(layer_sizes[i+1]),
                "activation": activation[i],
                }  
            )


    def forward(self, inputs:Union[list, np.ndarray]):
        """
        inputs: shape=(batch_size, feature_len )
        """
        outputs = []
        for x in inputs:
            # x = x.flatten()
            for i in range(len(self.layers)):
                x = self.layers[i]["weights"].T @ x + self.layers[i]['biases']
                x = self.act_funcs[self.layers[i]["activation"]].activate(x)

            outputs.append(x)                 # nparray of shape = [1,]
        return np.array(outputs).flatten()    # nparray of shape = [batch_size,1]
    
    def __call__(self,inputs):
        return self.forward(inputs)

    def predict(self, X):
        predictions = [] 
        outputs = self.forward(X)
        for output in outputs:
            if output >=0.5:
                predictions.append(1)
            else:
                predictions.append(0)

        return np.array(predictions)
    

    def set_optimizer(self, lr:float = 0.01, loss:str = "mse", optimizer="sgd"):
        self.learning_rate = lr
        self.loss_metric = self.loss_funcs[loss]
        if optimizer.lower() == 'sgd':
            self.optimizer = SGD(lr)
        # elif optimizer.lower() == 'sgd_mom':
        #     self.optimizer = SGDMomentum(lr)
        else:
            print("Optimizer not implemented!\nOptimizer has been set to SGD instead!")
            self.optimizer=SGD(lr)

    def backward(self, inputs, targets):
        """
            Input:
                Takes a batch of data as input
                inputs: A list of input data, shape=[batch_size, feature_size]
                targets: List of targets,eg [1,2,3] or [[1,0],[0,1]]

            Output:
                None
                Updates the layer gradeints and passes them to 
                optimizer objects, which update the weights accordingly    

            Same notation are used as in slides, 
            'x'     : represent input
            'net'   : represents the output of layer without activation
            'o'     : represents the output after activation  
        """
        if self.loss_metric ==None:
            raise Exception("Optimizer not set! Set optimizer before training !")
            return None
        
        # Initialize gradients
        self.weight_gradients = [np.zeros_like(layer["weights"]) for layer in self.layers]
        self.bias_gradients = [np.zeros_like(layer['biases']) for layer in self.layers]

        total_loss =0.0

        for input, target in zip(inputs, targets):
            x = input
            net, o = [x], [x]      # Same notation as in slides
            for i in range(len(self.layers)):
                x = self.layers[i]["weights"].T @ x + self.layers[i]['biases']
                net.append(x)
                x = self.act_funcs[self.layers[i]["activation"]].activate(x)
                o.append(x)
            
            x = x.flatten()
            loss = self.loss(x, target)
            total_loss += loss
            
            loss_gradient = self.loss_metric.grad(x, target)                        # ∂L/∂On
            
            
            for i in range(len(self.layers) - 1, -1, -1):
                activation_gradient = self.act_funcs[self.layers[i]["activation"]].grad(o[i + 1])
                # print(f"Act Grad: {activation_gradient}")
                self.weight_gradients[i] += np.outer(o[i], loss_gradient * activation_gradient) / len(inputs) 
                self.bias_gradients[i] += loss_gradient * activation_gradient / len(inputs)

                # Compute loss gradient for the next layer in the backward pass
                loss_gradient = np.dot(loss_gradient * activation_gradient, self.layers[i]["weights"].T)
            
        # self.weight_gradients = self.weight_gradients           ## Average gradient for batch
        # self.bias_gradients = self.bias_gradients / len(inputs)

        self.optimizer.update_weights(self)

            # for i in range(len(self.layers)):
            #     self.layers[i]["weights"] -= self.learning_rate * (self.weight_gradients[i]/len(inputs))
            #     self.layers[i]["biases"] -= self.learning_rate * (self.bias_gradients[i]/len(inputs))

    
    def train(self, X_train, y_train, epochs=10, batch_size=16):
        accuracies, losses = [], []
        for epoch in range(epochs):
            total_loss = 0.0
            
            with tqdm(total=len(X_train), desc=f"Epoch {epoch + 1}/{epochs}", unit="sample") as pbar:
                if self.optimizer.name.lower() == 'mbgd':
                        indices = np.random.permutation(len(X_train))
                        indices = indices[:batch_size]
                        input_samples = X_train[indices]
                        targets = y_train[indices]
                        predicted = self.forward(input_samples)
                        batch_loss = self.loss(predicted, targets)                    
                        total_loss += batch_loss
                        self.backward(input_samples, targets)
                        pbar.update(len(X_train))
                        accuracy = accuracy_metric(self.predict(X_train), y_train)
                        pbar.set_postfix(loss=total_loss, accuracy=accuracy)
                        accuracies.append(accuracy)
                        losses.append(total_loss)

                elif self.optimizer.name.lower() == 'bgd':
                        for i in range(0, len(X_train), batch_size):
                            input_samples = X_train[i:i + batch_size]
                            targets = y_train[i:i + batch_size]

                            predicted = self.forward(input_samples)
                            batch_loss = self.loss(predicted, targets)                    
                            total_loss += batch_loss
                            self.backward(input_samples, targets)
                            pbar.update(batch_size)
                        accuracy = accuracy_metric(self.predict(X_train), y_train)
                        pbar.set_postfix(loss=total_loss, accuracy=accuracy)
                        accuracies.append(accuracy)
                        losses.append(total_loss)

                elif self.optimizer.name.lower() == "sgd":                        
                        index = np.random.randint(0, len(X_train))
                        input_samples = [X_train[index]]
                        targets = [y_train[index]]
                        predicted = self.forward(X_train)
                        loss = self.loss(predicted, targets)                    
                        total_loss += loss
                        self.backward(input_samples, targets)
                        pbar.update(1)
                        
                else:
                    pass

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
