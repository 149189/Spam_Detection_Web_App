from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and tokenizer
model = tf.keras.models.load_model("spam_model.h5")
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define constants
MAX_SEQUENCE_LENGTH = 50

# Preprocessing function
def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])  
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')  
    return padded_sequence

# Home Page (User Input Form)
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form['message']  # Get input text from form
        processed_text = preprocess_text(user_text)  # Preprocess text
        prediction = model.predict(processed_text)[0][0]  # Get prediction
        label = "Spam" if prediction > 0.5 else "Not Spam"  # Convert to label
        confidence = round(float(prediction) * 100, 2)  # Confidence Score
        
        return render_template('index.html', user_text=user_text, label=label, confidence=confidence)

    return render_template('index.html', user_text="", label="", confidence="")

# Run Flask
if __name__ == '__main__':
    app.run(debug=True)
