# app.py - Flask aplikasi tanpa TensorFlow
from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load model dan scaler
print("🔄 Loading model...")
model_data = joblib.load('models/model_weights.pkl')
scaler_X = joblib.load('models/scaler_X.pkl')
scaler_y = joblib.load('models/scaler_y.pkl')

# Fungsi aktivasi sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

# Kelas Neural Network untuk prediksi
class NeuralNetwork:
    def __init__(self, weights, biases):
        self.weights = [np.array(w) for w in weights]
        self.biases = [np.array(b) for b in biases]
    
    def predict(self, X):
        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            current_input = sigmoid(z)
        return current_input

# Inisialisasi model
nn = NeuralNetwork(model_data['weights'], model_data['biases'])
print("✅ Model berhasil dimuat!")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil data dari form
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        depth = float(request.form['depth'])
        
        # Validasi input
        if not (-90 <= latitude <= 90):
            return render_template('result.html', error="Latitude harus antara -90 dan 90")
        if not (-180 <= longitude <= 180):
            return render_template('result.html', error="Longitude harus antara -180 dan 180")
        if not (0 <= depth <= 700):
            return render_template('result.html', error="Depth harus antara 0 dan 700 km")
        
        # Siapkan data untuk prediksi
        input_data = np.array([[latitude, longitude, depth]])
        input_scaled = scaler_X.transform(input_data)
        prediction_scaled = nn.predict(input_scaled)
        prediction = scaler_y.inverse_transform(prediction_scaled)
        magnitude = float(prediction[0][0])
        
        # Kategorikan tingkat bahaya
        if magnitude < 4.0:
            risk_level = "Rendah"
            risk_color = "success"
            description = "Gempa kecil, jarang menimbulkan kerusakan"
        elif magnitude < 5.5:
            risk_level = "Sedang"
            risk_color = "warning"
            description = "Gempa sedang, dapat dirasakan, kerusakan ringan"
        elif magnitude < 7.0:
            risk_level = "Tinggi"
            risk_color = "danger"
            description = "Gempa besar, berpotensi merusak bangunan"
        else:
            risk_level = "Sangat Tinggi"
            risk_color = "dark"
            description = "Gempa dahsyat, berpotensi tsunami dan kerusakan parah"
        
        return render_template('result.html',
                               magnitude=round(magnitude, 2),
                               latitude=latitude,
                               longitude=longitude,
                               depth=depth,
                               risk_level=risk_level,
                               risk_color=risk_color,
                               description=description)
    
    except ValueError:
        return render_template('result.html', error="Input tidak valid. Pastikan semua field terisi angka.")
    except Exception as e:
        return render_template('result.html', error=f"Terjadi kesalahan: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)