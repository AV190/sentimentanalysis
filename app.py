from flask import Flask, request, jsonify
import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer
from pymongo import MongoClient
from flask_cors import CORS  # Import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the tokenizer and model
try:
    tokenizer = BertTokenizer.from_pretrained('C:/Users/91938/Downloads/MINI/berttokennew')
    model = BertForSequenceClassification.from_pretrained('C:/Users/91938/Downloads/MINI/bertmodelnew')
    model.eval()  # Set model to evaluation mode
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

# Configure MongoDB client
client = MongoClient('localhost', 27017)
db = client['csv_db']
option_counter = db['option_counter']

# Initialize the option counter if it doesn't exist
if option_counter.count_documents({}) == 0:
    option_counter.insert_one({"latest_option": 0})

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    collection_name = request.form.get('collection', 'default_collection')
    
    try:
        data = pd.read_csv(file)
    except Exception as e:
        return jsonify({"error": f"Error reading the file: {e}"}), 400

    if 'text' not in data.columns:
        return jsonify({"error": "CSV must contain a 'text' column"}), 400

    # Get the latest option number and calculate the new option numbers
    latest_option_doc = option_counter.find_one_and_update(
        {},
        {"$inc": {"latest_option": data.shape[0]}},
        return_document=True,
        upsert=True
    )
    start_option = latest_option_doc['latest_option'] + 1

    # Add unique 'option' field to each record
    data['option'] = range(start_option, start_option + data.shape[0])

    # Store data into specified MongoDB collection
    collection = db[collection_name]
    data_to_insert = data.to_dict('records')
    collection.insert_many(data_to_insert)

    return jsonify({"message": "File uploaded and data inserted successfully"}), 201

@app.route('/predict_option', methods=['GET'])
def predict_option():
    option = request.args.get('option')
    collection_name = request.args.get('collection', 'default_collection')

    if not option:
        return jsonify({"error": "No option provided"}), 400

    # Fetch data from the specified collection
    collection = db[collection_name]
    try:
        option = int(option)  # Attempt to convert option to an integer
        data = list(collection.find({"option": option}))
    except ValueError:
        if option == 'all':
            data = list(collection.find())
        else:
            return jsonify({"error": "Invalid option provided"}), 400

    if not data:
        return jsonify({"error": "No data found for the selected option"}), 404

    texts = [item['text'] for item in data]

    if not texts:
        return jsonify({"error": "No text data found for the selected option"}), 404

    try:
        # Tokenize all texts and ensure they are in the correct format for the model
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        
        # Perform inference
        with torch.no_grad():  # Disable gradient calculations
            outputs = model(**inputs)
        
        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    except Exception as e:
        return jsonify({"error": f"Error predicting the sentiment: {e}"}), 500

    # Calculate number of negative, neutral, and positive predictions
    num_negative = int((predictions == 0).sum())
    num_neutral = int((predictions == 1).sum())
    num_positive = int((predictions == 2).sum())
    
    # Determine the final sentiment result
    if num_negative > num_neutral and num_negative > num_positive:
        result = "Negative"
    elif num_neutral > num_negative and num_neutral > num_positive:
        result = "Neutral"
    else:
        result = "Positive"

    return jsonify({
        "result": result,
        "negative": num_negative,
        "neutral": num_neutral,
        "positive": num_positive
    })

if __name__ == '__main__':
    app.run(debug=True)
