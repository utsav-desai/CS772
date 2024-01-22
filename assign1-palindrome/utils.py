import numpy as np



class Dataset:


    def __init__(self, n, h) -> None:
        self.n = n
        self.h = h


    def get_data(self, biasing_factor=0) -> [np.array, np.array]:
        x, palindromes = [], []
        for i in range(2**self.n):
            binary_string = format(i, '010b')
            x.append(binary_string)
            if binary_string == binary_string[::-1]:
                palindromes.append(binary_string)
        x = np.array(x)
        palindromes = np.array(palindromes)
        
        ## To somewhat balance the unbalanced classes in the dataset
        for i in range(biasing_factor):
            x = np.concatenate((x, palindromes))
        return x
        

        


