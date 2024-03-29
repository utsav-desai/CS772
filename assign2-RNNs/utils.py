import numpy as np

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    total_loss = np.sum(loss)
    return total_loss

def calculate_accuracy(y_true, y_pred):
    acc =  np.mean(y_true == (y_pred>0.5).astype(int))
    return acc