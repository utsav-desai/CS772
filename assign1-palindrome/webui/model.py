import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


weights_ih = np.load('/Users/utsavmdesai/Documents/Utsav/SEM 8/CS 772/CS772/assign1-palindrome/new_weights/weights_ih2.npy')
weights_ho = np.load('/Users/utsavmdesai/Documents/Utsav/SEM 8/CS 772/CS772/assign1-palindrome/new_weights/weights_ho2.npy')
bias_ih = np.load('/Users/utsavmdesai/Documents/Utsav/SEM 8/CS 772/CS772/assign1-palindrome/new_weights/bias_ih2.npy')
bias_ho = np.load('/Users/utsavmdesai/Documents/Utsav/SEM 8/CS 772/CS772/assign1-palindrome/new_weights/bias_ho2.npy')

input_size = 10
def predict_palindrome(x):
    inp = np.reshape(np.array([int(char) for char in x]), (1, input_size))
    output = sigmoid(np.dot(sigmoid(np.dot(inp, weights_ih)+ bias_ih), weights_ho)+bias_ho)
    if output > 0.5:
        # print('Palindrome')
        return 'Palindrome'
    else:
        # print('Not Palindrome')
        return 'Not Palindrome'
    # return output