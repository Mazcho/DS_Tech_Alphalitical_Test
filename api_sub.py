from flask import Flask, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from preprocess import preprocess_text

app = Flask(__name__)

# Load model and tokenizer
model = load_model('cnn_model_baru.h5')

with open('tokenizer_baru.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Store logs in memory
log_entries = []

@app.route('/')
def home():
    return "Welcome to the Sentiment Classification API"

@app.route('/classify', methods=['GET'])
def classify_sentiment():
    # The specific input text
    text = "kita akan semakin cerdas jika kita terus menerus melatih otak kita"

    # Preprocess the text
    processed_text = preprocess_text(text)

    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([processed_text])
    sequence_padded = pad_sequences(sequence, maxlen=200)

    # Perform sentiment classification
    prediction = model.predict(sequence_padded)
    sentiment = ['negative', 'neutral', 'positive'][prediction.argmax()]

    # Store log
    log_entry = {
        'input': text,
        'processed': processed_text,
        'output': sentiment
    }
    log_entries.append(log_entry)

    # Return the result in the desired format
    return jsonify({'output': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
