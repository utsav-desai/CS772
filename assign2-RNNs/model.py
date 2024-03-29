import pickle
import numpy as np
from activation import *
from utils import *
from sklearn.model_selection import KFold


"""
Some representation which will help understand this code base.
B       - Batch Size
Tx      - Length of Input string
X_i_b   -   
X_i_j   -   

"""

class SingleRecurrentPerceptron:
    def __init__(self, vec_len=10, lr=0.05):
          
        # Initialize weights and bias
        self.weights = np.random.randn(vec_len)
        self.threshold = np.random.randn(1)
        self.lr = lr

   
    def forward(self, inputs):
        """inputs-- (B, Tx, 10)"""   
        prediction= []    #(B, Tx)
        X_i_b = []      #(B, Tx, 10)
        for j in range(len(inputs)):
            out=[]
            X_i_j = []
            Tx, _ = inputs[j].shape
            y_prev=0
            for i in range(Tx):
                x = np.concatenate([inputs[j][i], np.array([y_prev])])
                X_i_j.append(x)
                net = x.T @ self.weights - self.threshold[0]
                oi = sigmoid(net)
                y_prev = oi
                out.append(oi)
            prediction.append(np.array(out))
            X_i_b.append(np.array(X_i_j))
        return X_i_b, prediction

    # def backward(self, inputs, target):
    #     """inputs-- (B, Tx, 10)
    #        target-- (B, Tx)
    #         """   
    #     X, prediction = self.forward(inputs)

    #     for i in range(len(inputs)):      # iterate over each example
    #         delta_w = np.zeros(10)
    #         for j in range(len(inputs[i])):     # iterate over each time
    #             x = X[i][j]
    #             delta_w += -self.lr * (target[i][j]-prediction[i][j]) * (x)
    #         self.weights += delta_w
    

            
    def backward_gpt(self, inputs, targets):

        X_i_b, prediction = self.forward(inputs)
         
        B = len(inputs)  # Get batch size, sequence length, and feature dim

        # Initialize gradients for weights and bias
        self.weights_grad = np.zeros_like(self.weights)
        self.threshold_grad = np.zeros_like(self.threshold)
        sequence_lengths = [len(t) for t in targets]

        # Calculate gradients for output layer (using element-wise multiplication)
        for b in range(B):
            Tx = sequence_lengths[b]
            for t in range(Tx):
                delta_t = (prediction[b][t] - targets[b][t]) * sigmoid_derivative(prediction[b][t])
                self.weights_grad += X_i_b[b][t]*delta_t
                self.threshold_grad += delta_t

            # BPTT (using chain rule)
            delta_prev = 0
            for t in reversed(range(Tx)):
                if t + 1 < Tx:
                    delta_t = delta_prev * sigmoid_derivative(prediction[b][t]) + np.dot(delta_t, self.weights)
                else:
                    delta_t = delta_prev * sigmoid_derivative(prediction[b][t])
                self.weights_grad -= X_i_b[b][t] * delta_t*0.3  # Exclude previous output
                delta_prev = delta_t

        # Normalize gradients by batch size
        self.weights_grad /= B
        self.threshold_grad /= B

        # Update weights and bias
        self.weights -= self.lr * self.weights_grad
        self.threshold -= self.lr * self.threshold_grad

    def calculate_loss(self, inputs, targets):
        
        """
        This function calculates the total loss for a minibatch of sequences.

        Args:
        inputs: Batch of input sequences (B, Tx_max, vec_len).
        targets: Batch of ground truth sequences (B, Tx_max).

        Returns:
        The average loss over the minibatch.
        """
        B = len(inputs)  # Get batch size, max sequence length, and feature dim

        # Initialize loss to zero
        loss = 0
        accuracy = 0
        _, predictions = self.forward(inputs)
        # Forward pass for each example in the minibatch
        for b in range(B):
            # Calculate loss per example using cross-entropy
          
            loss += cross_entropy_loss(predictions[b], targets[b])
            accuracy += np.mean(targets[b]==(predictions[b]>0.5).astype(int))
            
        # Average loss over the minibatch
        return loss / B, accuracy/B


        
    def train(self, inputs, targets, epochs):

        """inputs-- (B, Tx, 10)
           target-- (B, Tx)
            """           
        
        for iter in range(epochs):
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            train_loss = 0
            val_loss = 0
            train_accuracy = 0
            val_accuracy = 0
            for train_index, val_index in kf.split(inputs):
                train_inputs, val_inputs = [inputs[i] for i in train_index], [inputs[i] for i in val_index]
                train_targets, val_targets = [targets[i] for i in train_index], [targets[i] for i in val_index]
                self.backward_gpt(inputs, targets)
                delta_loss, delta_accuracy = self.calculate_loss(train_inputs,train_targets)
                train_loss += delta_loss
                train_accuracy += delta_accuracy
                delta_loss, delta_accuracy = self.calculate_loss(val_inputs,val_targets )

                val_loss += delta_loss
                val_accuracy += delta_accuracy
            print(f"epoch: {iter:.2f}, training loss : {train_loss/5:.2f}, training accuracy: {train_accuracy*100/5:.2f}%, validation loss: {val_loss/5:.2f}, validation accuracy: {val_accuracy*100/5:.2f}%")
    

    def save(self,path="model.pkl"):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self = pickle.load(f) 
        return self