from flask import Flask, render_template, request, jsonify
import numpy as np
import gradio as gr
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model =  AutoModelForSequenceClassification.from_pretrained("./my_model_weights")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def classify_text(text):
    # Preprocess the text
    text = text.lower()
    text = re.sub("[#=><\/.]", "", text)
    text = re.sub("@\w+", "", text)

    # Tokenize the text
    tokenized_text = tokenizer(text)

    # Convert the tokenized text to a tensor
    input_ids = torch.tensor(tokenized_text["input_ids"]).unsqueeze(0)
    attention_mask = torch.tensor(tokenized_text["attention_mask"]).unsqueeze(0)

    # Make predictions
    outputs = model(input_ids, attention_mask=attention_mask)
    predictions = torch.argmax(outputs.logits, dim=-1)

    # Return the predictions
    return predictions.item()

demo = gr.Interface(
    fn=classify_text,
    inputs=gr.Textbox(label="Enter a text sentence here"),
    outputs="label",
    examples=[
        "This is a disaster!",
        "Earthquake is expected in china !",
        "I'm feeling happy.",
    ],
)

demo.launch(share=True)