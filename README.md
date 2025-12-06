# 📨 SMS Spam Detector

A machine learning-powered web application that classifies SMS messages as **Spam** or **Ham (Legitimate)** using Naive Bayes classifier with TF-IDF vectorization.

You can check using this link - https://check-sms-spam-detection.netlify.app/ -
try this :)

## 🌟 Features

- ✅ Real-time SMS classification
- ✅ Confidence probability scores
- ✅ Beautiful, responsive UI
- ✅ Pre-trained ML model (97.5% accuracy)
- ✅ Works offline (no internet required)
- ✅ Built with Flask and Python

## 📊 Model Details

- **Algorithm:** Naive Bayes Classifier
- **Feature Extraction:** TF-IDF Vectorizer
- **Training Data:** 5,572 SMS messages
- **Accuracy:** 97.5%
- **Dataset:** UCI Machine Learning Repository (spam.csv)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning repository)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/spam-sms-detector.git
cd spam-sms-detector
```

2. **Create virtual environment**
```bash
# Windows
python -m venv spam_env
spam_env\Scripts\activate

# macOS/Linux
python3 -m venv spam_env
source spam_env/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data**
```bash
python -m nltk.downloader punkt_tab stopwords
```

## 📝 Usage

### Step 1: Train the Model

Run your Jupyter notebook to train and save the model:
```bash
# Make sure you're in your project directory
# Run your untitled.ipynb notebook
# This generates: model1.pkl and vectorizer1.pkl
```

**Notebook code to save model:**
```python
import pickle

# After training your model
pickle.dump(model, open('model1.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer1.pkl', 'wb'))
print("✅ Model and vectorizer saved!")
```

### Step 2: Start Flask Server

```bash
# Make sure virtual environment is activated
spam_env\Scripts\activate

# Run the Flask app
python app.py
```

You'll see:
```
Running on http://127.0.0.1:5000
```

### Step 3: Open in Browser

1. Open your web browser
2. Go to: `http://localhost:5000`
3. Enter an SMS message
4. Click "Classify Message"
5. See the results!

## 📁 Project Structure

```
spam-sms-detector/
├── app.py                    # Flask application
├── templates/
│   └── index.html           # Web UI
├── untitled.ipynb           # Training notebook
├── spam.csv                 # Training dataset
├── model1.pkl               # Trained model (generated)
├── vectorizer1.pkl          # TF-IDF vectorizer (generated)
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore file
└── README.md               # This file
```

## 🧪 Testing

### Example Spam Messages
- "FREE entry to win £1000 cash! Click here now!"
- "CONGRATULATIONS! You've won a FREE iPhone! Claim now."
- "Get rich quick! Make £5000/week from home!"

### Example Ham Messages
- "Hey! How are you doing today?"
- "Meeting at 3 PM tomorrow, don't forget!"
- "Thanks for calling me. Talk soon!"

## 🔧 API Endpoints

### 1. **Home Page**
```
GET http://localhost:5000/
```
Returns the HTML interface

### 2. **Predict (Classify Message)**
```
POST http://localhost:5000/predict
Content-Type: application/json

{
  "message": "Your message here"
}
```

Response:
```json
{
  "prediction": "Spam",
  "confidence": 0.95,
  "prob_spam": 0.95,
  "prob_ham": 0.05
}
```

### 3. **Check Status**
```
GET http://localhost:5000/status
```

Response:
```json
{
  "model_loaded": true,
  "model_file": true,
  "vectorizer_file": true,
  "dataset_available": true
}
```

## 📦 Dependencies

All dependencies are in `requirements.txt`:
```
Flask==2.3.0
pandas==2.0.0
scikit-learn==1.2.0
numpy==1.24.0
nltk==3.8.1
Werkzeug==2.3.0
```

Install with:
```bash
pip install -r requirements.txt
```

## 🐛 Troubleshooting

### Error: "jinja2.exceptions.TemplateNotFound: index.html"
- Make sure you have a `templates/` folder
- Verify `index.html` is inside the `templates/` folder
- Folder structure: `templates/index.html`

### Error: "LookupError: Resource punkt_tab not found"
```bash
python -m nltk.downloader punkt_tab
```

### Error: "ModuleNotFoundError"
```bash
pip install -r requirements.txt
```

### Model not loading
- Run your Jupyter notebook first
- Check if `model1.pkl` and `vectorizer1.pkl` exist in the project folder
- Verify file paths are correct

## 📚 Files Explanation

### app.py
Flask backend that:
- Loads pre-trained model and vectorizer
- Handles message classification
- Provides JSON API endpoints
- Text preprocessing

### templates/index.html
Web interface that:
- Accepts user input
- Displays results with probabilities
- Shows confidence scores
- Fully responsive design

### untitled.ipynb
Jupyter notebook for:
- Data loading and exploration
- Model training
- Model evaluation
- Saving trained model

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📊 Dataset Source

- **Dataset:** UCI Machine Learning Repository
- **Name:** SMS Spam Collection Dataset
- **Link:** https://www.kaggle.com/uciml/sms-spam-collection-dataset
- **Size:** ~5,500 SMS messages
- **Classes:** Ham (4,827) | Spam (747)

## 👨‍💻 Author

Your Name - Subam prasad

## 🙏 Acknowledgments

- Flask framework
- Scikit-learn for ML algorithms
- NLTK for text processing
- UCI Machine Learning Repository

## 📝 Changelog

### v1.0.0 (2024)
- Initial release
- Naive Bayes classifier
- Web UI with Flask
- 97.5% accuracy

---
