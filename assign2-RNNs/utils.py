import numpy as np
import json
import os

"""
Evaluation Metrics
"""

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    total_loss = np.sum(loss)
    return total_loss

def calculate_accuracy(y_true, y_pred):
    acc =  np.mean(y_true == (y_pred>0.5).astype(int))
    return acc



"""
Data Loading
"""
def fetch_data(train_path, test_path):
    with open(train_path, 'r') as f:
        train_data = [json.loads(line) for line in f]
    # Preprocess training data
    train_sentences = []
    train_labels = []
    for entry in train_data:
        tokens = entry['tokens']
        pos_tags = entry['pos_tags']
        chunk_tags = np.array(entry['chunk_tags'])
        
        train_sentences.append(pos_tags)
        train_labels.append(chunk_tags)

    with open(test_path, 'r') as f:
        test_data = [json.loads(line) for line in f]
    # Preprocess test data
    test_sentences = []
    test_labels = []
    for entry in test_data:
        tokens = entry['tokens']
        pos_tags = entry['pos_tags']
        chunk_tags = np.array(entry['chunk_tags'])
        
        test_sentences.append(pos_tags)
        test_labels.append(chunk_tags)
    
    return train_sentences, test_sentences,  train_labels, test_labels




"""
Function to process the POS data
into One hot vectors,
that is :
Input: [DT/TT/NN/OT]    Shape: (1)
Output : [DT TT NN OT]  Shape: (4,1)
"""
def one_hot_encode(input_list):
    encoded_list = []
    for item in input_list:
        one_hot_vector = np.zeros(4)
        one_hot_vector[item - 1] = 1  # Adjust index to start from 0
        encoded_list.append(one_hot_vector.tolist())
    return np.array(encoded_list)


"""
Function to process the POS data
into Recurrent Perceptron input format,
that is :
Input : [DT TT NN OT]
"""
def into_ho(X_train):
    X_train_ho = []
    for i in range(len(X_train)):
        X = one_hot_encode(X_train[i])
        temp = []
        for j in range(len(X)):
            if j==0:
                temp.append(np.concatenate([np.array([1.0,0.0,0.0,0.0,0.0]), X[j]]))
            else:
                temp.append((np.concatenate([np.array([0]), X[j-1], X[j]] )))
        X_train_ho.append(np.array(temp))
    return X_train_ho