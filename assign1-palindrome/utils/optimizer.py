import numpy as np

class Optimizer:
    def __init__(self, learning_rate, name):
        self.learning_rate = learning_rate
        self.name = name

    def update_weights(self, model):
        raise NotImplementedError("Subclasses must implement the 'update_weights' method")

class SGD(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate,"SGD")

    def update_weights(self, model):
        for i in range(len(model.layers)):
            model.layers[i]["weights"] -= self.learning_rate * (model.weight_gradients[i])
            model.layers[i]["biases"] -= self.learning_rate * (model.bias_gradients[i])


class SGDMomentum(Optimizer):
    def __init__(self, learning_rate, momentum=0.69):
        super().__init__(learning_rate,"SGD_Mom")
        self.momentum = momentum
        self.velocity_weights = None
        self.velocity_biases = None

    def initialize_velocity(self, model):
        self.velocity_weights = [np.zeros_like(layer["weights"]) for layer in model.layers]
        self.velocity_biases = [np.zeros_like(layer['biases']) for layer in model.layers]

    def update_weights(self, model):
        if self.velocity_weights is None:
            self.initialize_velocity(model)

        for i in range(len(model.layers)):
            self.velocity_weights[i] = self.momentum * self.velocity_weights[i] + self.learning_rate * (model.weight_gradients[i])
            model.layers[i]["weights"] -= self.velocity_weights[i]

            self.velocity_biases[i] = self.momentum * self.velocity_biases[i] + self.learning_rate * (model.bias_gradients[i])
            model.layers[i]["biases"] -= self.velocity_biases[i]