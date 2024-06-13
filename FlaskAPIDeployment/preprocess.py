# preprocess.py
import re
import string

def preprocess_text(text):
    text = text.lower()  # Ubah teks menjadi huruf kecil
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))  # Hapus tanda baca
    text = text.strip()  # Hapus spasi di awal dan akhir
    return text
