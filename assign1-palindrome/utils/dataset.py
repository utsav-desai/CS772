import numpy as np

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
        

