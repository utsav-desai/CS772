import json
import os
import numpy as np


def one_hot_encode(input_list):
    encoded_list = []
    for item in input_list:
        one_hot_vector = np.zeros(4)
        one_hot_vector[item - 1] = 1  # Adjust index to start from 0
        encoded_list.append(one_hot_vector.tolist())
        return encoded_list


def load_data(path):
    with open(path, 'r') as f:
        data = [json.loads(line) for line in f]

    sentences = []
    pos_emb=[]
    labels = []

    for entry in data:
        
        sentence = " ".join(entry['tokens'])
        pos_tags = np.array(entry['pos_tags'])
        chunk_tags =np.array(entry['chunk_tags'])
        
        sentences.append(sentence)
        pos_emb.append(pos_tags)
        labels.append(chunk_tags)
    
    return {"sentences": sentences, "POS": pos_emb ,"labels": labels}


def fetch_data():
    train_data = load_data("../data/train.jsonl")
    test_data = load_data("../data/test.jsonl")

