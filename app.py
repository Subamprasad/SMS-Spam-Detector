from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = 'model1.pkl'
vectorizer_path = 'vectorizer1.pkl'

model_loaded = False
model = None
vectorizer = None

# Check if files exist
if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        model_loaded = True
        print("✅ Model and vectorizer loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model_loaded = False
else:
    print("⚠️ Model files not found. Please run your notebook first!")
    print(f"Looking for: {model_path} and {vectorizer_path}")

# Load spam dataset for demo
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df[['v1', 'v2']]
        df.columns = ['target', 'text']
        df['target'] = df['target'].map({'ham': 0, 'spam': 1})
    else:
        df = None
        print("⚠️ CSV columns not found")
except Exception as e:
    df = None
    print(f"⚠️ Error loading spam.csv: {e}")

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model_loaded:
            return jsonify({
                'error': 'Model not loaded. Please run your notebook first to generate model1.pkl and vectorizer1.pkl'
            }), 400
        
        data = request.json
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        if len(message) < 5:
            return jsonify({'error': 'Message too short'}), 400
        
        # Preprocess the message
        processed_message = preprocess_text(message)
        
        # Transform the message using the vectorizer
        message_vec = vectorizer.transform([processed_message])
        
        # Make prediction
        prediction = model.predict(message_vec)[0]
        probability = model.predict_proba(message_vec)[0]
        
        result = {
            'prediction': 'Spam' if prediction == 1 else 'Ham',
            'confidence': float(max(probability)),
            'prob_spam': float(probability[1]),
            'prob_ham': float(probability[0]),
            'message': message
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in predict: {e}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/demo', methods=['GET'])
def demo():
    """Get random demo messages"""
    try:
        if df is None:
            return jsonify({'error': 'Dataset not available'}), 404
        
        # Get 5 random samples
        samples = df.sample(min(5, len(df))).to_dict('records')
        return jsonify(samples)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Check if model is loaded"""
    return jsonify({
        'model_loaded': model_loaded,
        'model_file': os.path.exists(model_path),
        'vectorizer_file': os.path.exists(vectorizer_path),
        'dataset_available': df is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 SPAM DETECTOR WEB APP")
    print("="*60)
    print(f"✅ Model Loaded: {model_loaded}")
    print(f"✅ Vectorizer Loaded: {model_loaded}")
    print(f"✅ Dataset Available: {df is not None}")
    print("\n📌 Instructions:")
    print("1. Open http://localhost:5000 in your browser")
    print("2. Enter a message to classify")
    print("3. Click 'Classify' to get results")
    print("\n" + "="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)