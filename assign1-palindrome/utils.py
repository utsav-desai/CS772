import numpy as np


"""
A class for the Dataset
"""
class Palindrome_Dataset:
    bit_size = None
    palindromes: np.array = None
    non_palindromes:np.array = None

    def __init__(self, n_bits=10) -> None:
        self.bit_size= n_bits
        palindromes, non_palindromes = [], []
        for i in range(2**self.bit_size):
            binary_string = format(i, '010b')
            if binary_string == binary_string[::-1]:
                palindromes.append(binary_string)
            else:
                non_palindromes.append(binary_string)
        
        palindromes = np.array(palindromes)
        non_palindromes = np.array(non_palindromes)
        self.palindromes = palindromes
        self.non_palindromes = non_palindromes
        

    def get_data(self, biasing_factor=0, shuffle=True) -> [np.array, np.array]:
        x, y = np.concatenate((self.non_palindromes, self.palindromes)), []
        for i in range(biasing_factor):
            x = np.concatenate((x, self.palindromes))
        if shuffle:
            np.random.shuffle(x.flat)
        
        for binary_string in x: 
            if binary_string == binary_string[::-1]:
                y.append(1)
            else:
                y.append(0)
        y = np.array(y)
        x = np.array([np.reshape(np.array([int(char) for char in s]), (self.bit_size,)) for s in x])
        return x,y
        


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
    
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__("sigmoid")
        self.alpha = alpha
        self.beta = beta

    def activate(self, x):
        x = np.clip(x, -500, 500)   ## To aid overflow warning
        return self.beta / (1.0 + np.exp(-1 *self.alpha* x))

    def grad(self, x):
        sig = self.activate(x)
        return sig*(1-sig)
    

class Linear(Activation):
    def __init__(self):
        super().__init__("linear")
    
    def activate(self, x):
        return x
    
    def grad(self, x=None):
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


"""

Some loss functions 
"""

def mse(y_true, y_pred):
    if isinstance(y_true)==list:
        y_true= np.array(y_true)
    
    if isinstance(y_pred)==list:
        y_pred= np.array(y_pred)
    np.mean((y_true - y_pred)**2)

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()