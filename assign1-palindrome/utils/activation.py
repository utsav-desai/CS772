import numpy as np

"""
This section contains some activation functions
like sigmoid, relu, and linear
"""
class Activation:

    def __init__(self, name):
        self.name = name

    def activate(self, x):
        raise NotImplementedError("Subclasses must implement the 'activate' method")
    def grad(self, x):
        raise NotImplementedError("Subclasses must implement the 'grad' method")


class Sigmoid(Activation):
    
    def __init__(self,):
        super().__init__("sigmoid")

    def activate(self, x):
        x = np.clip(x, -500.0, 500.0)   ## To aid overflow warning
        return 1.0 / (1.0 + np.exp(-1 * x))
    
    def grad(self, x):
        sig = self.activate(x)
        sig = np.clip(sig, 1e-7, 1.0)
        return sig*(1-sig)
    

class Linear(Activation):
    def __init__(self):
        super().__init__("linear")
    
    def activate(self, x):
        return x
    
    def grad(self, x):
        return 1.0
    
    
class Relu(Activation):
    def __init__(self):
        super().__init__("relu")
    def activate(self, x):
        return np.maximum(0, x)
    
    def grad(self, x):
        """
        However relu is not differentiable at 0, we can take an arbitrary 
        finite value for gradient at 0. This works well in practice.
        """
        return 1.0 if x > 0 else 0.0

