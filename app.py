from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import matplotlib.pyplot as plt

app = Flask(__name__)

# Load the tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained('C:/Users/91938/Downloads/MINI/berttoken')
    model = BertForSequenceClassification.from_pretrained('C:/Users/91938/Downloads/MINI/bertmodel')
    model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    try:
        stemmed_data = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading the file: {e}"}), 400

    if 'text' not in stemmed_data.columns:
        return jsonify({"error": "CSV must contain a 'text' column"}), 400

    texts = stemmed_data['text'].tolist()
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():  # Disable gradient calculations
            outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1).numpy()
    except Exception as e:
        return jsonify({"error": f"Error predicting the sentiment: {e}"}), 500

    num_positive = (predictions == 1).sum()
    num_negative = (predictions == 0).sum()

    if num_positive > num_negative:
        result = "Positive"
    elif num_positive < num_negative:
        result = "Negative"
    else:
        result = "Neutral"

    # Visual representation using a pie chart
    labels = ['Positive', 'Negative']
    sizes = [num_positive, num_negative]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)  # explode the 1st slice (positive)

    plt.figure(figsize=(8, 6))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title(f"Sentiment Analysis Result: {result}")
    plt.show()


    return jsonify({
        "Number of Positive Texts": int(num_positive),
        "Number of Negative Texts": int(num_negative),
        "Result": result
    })

if __name__ == '__main__':
    app.run(debug=True)
