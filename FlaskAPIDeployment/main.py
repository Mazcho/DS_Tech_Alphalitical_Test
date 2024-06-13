from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from preprocess import preprocess_text

app = Flask(__name__)

# Memuat model dan tokenizer
model = load_model('cnn_model_baru.h5')

with open('tokenizer_baru.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Menyimpan log dalam memori
log_entries = []

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_sentiment():
    # Memeriksa apakah permintaan memiliki data JSON dan teks yang valid
    data = request.get_json()
    if 'text' not in data:
        return jsonify({'error': 'Teks tidak ditemukan dalam permintaan.'}), 400
    text = data['text']
    if not text:
        return jsonify({'error': 'Teks tidak boleh kosong.'}), 400

    # Prapemrosesan teks
    processed_text = preprocess_text(text)

    # Tokenisasi dan padding teks
    sequence = tokenizer.texts_to_sequences([processed_text])
    sequence_padded = pad_sequences(sequence, maxlen=200)

    # Melakukan klasifikasi sentimen
    prediction = model.predict(sequence_padded)
    sentiment = ['negative', 'neutral', 'positive'][prediction.argmax()]

    # Menyimpan log
    log_entry = {
        'input': text,
        'processed': processed_text,
        'output': sentiment
    }
    log_entries.append(log_entry)

    # Mengembalikan hasil klasifikasi dalam bentuk respons JSON
    return jsonify({'input': text, 'processed': processed_text, 'output': sentiment, 'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
