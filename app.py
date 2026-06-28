from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# 1. Memuat "Otak" dan alat penerjemah AI yang BARU
model = joblib.load('ai_bruteforce_model.pkl')
scaler = joblib.load('scaler.pkl')
encoders = joblib.load('encoders.pkl') # Menggunakan multiple encoders sekarang

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Mengambil data dari website HTML/Laravel
        data = request.json
        
        # Menerjemahkan teks (misal: "root") menjadi angka rahasia AI
        ip_encoded = encoders['ip'].transform([data['source_ip']])[0]
        user_encoded = encoders['user'].transform([data['username']])[0]
        event_encoded = encoders['event'].transform([data['event_type']])[0]
        status_encoded = encoders['status'].transform([data['status']])[0]
        
        # Menggabungkan 4 fitur tersebut ke dalam matriks 
        fitur = np.array([[ip_encoded, user_encoded, event_encoded, status_encoded]])
        
        # Menyamakan skala angka
        fitur_scaled = scaler.transform(fitur)
        
        # AI melakukan tebakan
        prediksi_hasil = model.predict(fitur_scaled)[0]
        
        return jsonify({
            'status': 'sukses',
            'prediksi': str(prediksi_hasil) # Outputnya langsung 'normal' atau 'brute_force'
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'pesan': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)