import numpy as np



"""

Some loss functions 
"""
class Loss():
    def __init__(self, name) -> None:
        self.name = name

    def loss(self) -> float:
        raise NotImplementedError("Subclass must implement 'loss' method")

    def grad(self) -> float:
        raise NotImplementedError("Subclass must implement 'grad' method")

class MSE(Loss):    ## Mean squarred Error

    def __init__(self) -> None:
        super().__init__("mse")

    def loss(self, predicted, target):
        if isinstance(target,list):
            target= np.array(target)
        if isinstance(predicted,list):
            predicted= np.array(predicted)
        return np.mean((predicted - target)**2)/2
    
    def grad(self, predicted, target):
        return (predicted-target)         ## Ignoring other normalization constant
        

class BCE(Loss):        # Binary Crossentropy
    def __init__(self, ) -> None:
        super().__init__("binary_cross_entropy")
    
    def loss(self, predicted, target, epsilon=1e-12) -> float:
        predicted = np.array(predicted).flatten()
        target = np.array(target).flatten()
        predicted = np.clip(predicted, epsilon, 1 - epsilon)      # clip values to avoid log(0)
        return -np.sum(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

    def grad(self, predicted, target, epsilon=1e-9):
        predicted = np.clip(predicted, epsilon, 1 - epsilon)
        return (predicted-target)/(predicted*(1-predicted))


def accuracy_metric(predicted, target):
    if len(target)!=len(predicted):
        raise ValueError("Both predicted and target vectors should be same size!")

    correct = 0
    for i in range(len(target)):
        if target[i] == predicted[i]:
            correct += 1
    return correct / float(len(target)) * 100.0