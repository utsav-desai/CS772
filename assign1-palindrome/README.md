# Assignment-1 : Palindrome Classification Problem

## Training Methods

Initially we trained a network with 4 neuron in the hidden layer, on visualizing the weights we saw the pattern in only 2 out of 4 weights. This gave us an idea to implement the a network with 2 neurons in the hidden layer, but our training wasn't converging (stuck at a local mininima). To resolve this problem we copied the weights from 4 neuron network to 2 neuron network, and trained the model again, and training was succesful and we got 100% accuracy on both classes.

## How to Use this code

There is palindrome.ipynb file which contains the whole implementation of Backpropagation as well as model training.
Just go through each shell in the notebook. K-fold cross validation is also implemented at the end of the notebook.

## How to run UI

Go to *webui* folder present in the main folder and run the test.ipynb file; You'll get a URL for testing the model. Enter any string of length 10 made of 1 and 0 and press submit, you'll get the result in the adjacent textbox.

## Python Packages Required

    - numpy==1.24.3
    - matplotlib==3.7.5
    - sklearn = 1.2.1
