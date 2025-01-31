from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle


app = Flask(__name__)


model = tf.keras.models.load_model("spam_model.h5")
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)


MAX_SEQUENCE_LENGTH = 50


def preprocess_text(text):
    sequence = tokenizer.texts_to_sequences([text])  
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')  
    return padded_sequence


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_text = request.form['message']  
        processed_text = preprocess_text(user_text)  
        prediction = model.predict(processed_text)[0][0] 
        label = "Spam" if prediction > 0.5 else "Not Spam"  
        confidence = round(float(prediction) * 100, 2) 
        
        return render_template('index.html', user_text=user_text, label=label, confidence=confidence)

    return render_template('index.html', user_text="", label="", confidence="")


if __name__ == '__main__':
    app.run(debug=True)
