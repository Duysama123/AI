from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import pickle
import numpy as np
import os

app = Flask(__name__)

# Load các model và encoder
model = tf.keras.models.load_model('vietnamese_emotion_bilstm_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('label_encoder.pickle', 'rb') as handle:
    label_encoder = pickle.load(handle)

# Bảng ánh xạ cảm xúc và tên file nhạc
emotion_to_music_filename = {
    'vui vẻ': 'music/joy_music.mp3',
    'buồn bã': 'music/healing_music.mp3',
    'tức giận': 'music/relaxing_music.mp3',
    'sợ hãi': 'music/calm_music.mp3',
    'ngạc nhiên': 'music/soothing_music.mp3',
    'ghê tởm': 'music/peaceful_music.mp3',
}

default_music_filename = 'music/default_music.mp3'

# Hàm gợi ý nhạc
def suggest_music(emotion):
    return emotion_to_music_filename.get(emotion, default_music_filename)

# Route chính
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        text = request.form['text']
        if not text:
            return jsonify({"error": "Vui lòng nhập văn bản"}), 400

        # Tiền xử lý văn bản
        sequence = tokenizer.texts_to_sequences([text])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)

        # Dự đoán cảm xúc
        prediction = model.predict(padded_sequence)
        predicted_label = np.argmax(prediction, axis=1)
        vietnamese_emotion = label_encoder.inverse_transform(predicted_label)[0]

        # Đề xuất nhạc
        music_filename = suggest_music(vietnamese_emotion)

        # Trả về kết quả dưới dạng JSON
        return jsonify({
            'emotion': vietnamese_emotion,
            'music_filename': music_filename
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
