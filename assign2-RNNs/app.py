from flask import Flask, render_template, request, jsonify
import numpy as np
import gradio as gr
from activation import *
from utils import *
from model import SingleRecurrentPerceptron


model = SingleRecurrentPerceptron()
model = model.load("models/recurrent_perceptron_80%.pkl")


def forward_per_input(inputs):
    # inputs = one_hot_encode(inputs)
    inputs = into_ho([inputs])[0]
    """inputs-- (Tx, 10) """   
    out=[]    #(Tx, 1)
    X_i_j = []  #(Tx, 10)
    Tx = len(inputs)
    y_prev=0
    for i in range(Tx):
        x = np.concatenate([inputs[i], np.array([y_prev])])
        X_i_j.append(x)
        x = x.T @ model.weights - model.threshold[0]
        x = sigmoid(x)
        out.append(x)
    out = (np.array(out) > 0.5) * 1
    return out

def predict(POS_Tags):
    inp = POS_Tags.split(',')
    POS_Tags = [int(i) for i in inp]
    print(POS_Tags)
    result = forward_per_input(POS_Tags)
    out = [str(i) for i in result]
    out = ','.join(out)
    return out

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    data = request.get_json()
    POS_Tags = data["inputString"]
    out = predict(POS_Tags)
    return jsonify({"result": out})

if __name__ == "__main__":

    examples = [
        ["1,4,3,1,4,4,3,1,4", "1,1,1,0,1,1,1,0,1"]
    ]

    iface = gr.Interface(fn=predict, inputs=gr.Textbox(label="POS Tags"), outputs=gr.Textbox(label="Noun Chunks"), title="Noun Chunk Marker", examples=examples)
    iface.launch(share=True)