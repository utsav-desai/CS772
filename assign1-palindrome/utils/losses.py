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
        if isinstance(target)==list:
            target= np.array(target)

        if isinstance(predicted)==list:
            predicted= np.array(predicted)

        np.mean((predicted - target)**2)
    
    def grad(self, predicted, target):
        return (predicted-target)               ## Ignoring other normalization constant
        


class BCE(Loss):        # Binary Crossentropy
    def __init__(self, ) -> None:
        super().__init__("binary_cross_entropy")
    
    def loss(self, y_pred, y_true, epsilon=1e-12) -> float:
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)      # clip values to avoid log(0)
        return -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def grad(self, predicted, target):
        return (predicted-target)/(predicted*(1-predicted))
