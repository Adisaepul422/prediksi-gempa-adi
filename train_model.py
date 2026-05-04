# train_model.py - Backpropagation murni tanpa TensorFlow
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ========== 1. FUNGSI BACKPROPAGATION ==========

class NeuralNetwork:
    """Neural Network dengan Backpropagation dari awal (NumPy only)"""
    
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01):
        """
        input_size: jumlah fitur input
        hidden_sizes: list jumlah neuron di setiap hidden layer
        output_size: jumlah neuron output
        learning_rate: kecepatan belajar
        """
        self.learning_rate = learning_rate
        
        # Inisialisasi bobot dan bias
        self.weights = []
        self.biases = []
        
        # Layer sizes: [input, hidden1, hidden2, ..., output]
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            # Inisialisasi bobot dengan nilai kecil acak (Xavier initialization)
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def sigmoid(self, x):
        """Fungsi aktivasi Sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip untuk mencegah overflow
    
    def sigmoid_derivative(self, x):
        """Turunan fungsi Sigmoid"""
        return x * (1 - x)
    
    def forward(self, X):
        """Forward propagation"""
        self.activations = [X]
        self.z_values = []
        
        current_input = X
        for i in range(len(self.weights)):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self.sigmoid(z)
            self.activations.append(a)
            current_input = a
        
        return self.activations[-1]
    
    def backward(self, X, y, output):
        """Backward propagation"""
        m = X.shape[0]  # jumlah sampel
        gradients_w = []
        gradients_b = []
        
        # Hitung error di output layer
        error = output - y
        delta = error * self.sigmoid_derivative(output)
        
        # Backpropagate melalui layer
        for i in range(len(self.weights) - 1, -1, -1):
            # Gradien untuk bobot layer i
            grad_w = np.dot(self.activations[i].T, delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m
            gradients_w.insert(0, grad_w)
            gradients_b.insert(0, grad_b)
            
            # Propagate delta ke layer sebelumnya
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.activations[i])
        
        return gradients_w, gradients_b
    
    def update_weights(self, gradients_w, gradients_b):
        """Update bobot menggunakan gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients_w[i]
            self.biases[i] -= self.learning_rate * gradients_b[i]
    
    def train(self, X, y, epochs=200, batch_size=32, validation_split=0.2, patience=10, verbose=True):
        """Training neural network dengan early stopping"""
        
        # Split data untuk validasi
        val_size = int(len(X) * validation_split)
        X_train = X[val_size:]
        y_train = y[val_size:]
        X_val = X[:val_size]
        y_val = y[:val_size]
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        best_biases = None
        
        for epoch in range(epochs):
            # Training per batch
            epoch_loss = 0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                # Forward propagation
                output = self.forward(batch_X)
                
                # Hitung loss (MSE)
                batch_loss = np.mean((output - batch_y) ** 2)
                epoch_loss += batch_loss
                num_batches += 1
                
                # Backward propagation
                gradients_w, gradients_b = self.backward(batch_X, batch_y, output)
                
                # Update bobot
                self.update_weights(gradients_w, gradients_b)
            
            avg_train_loss = epoch_loss / num_batches
            train_losses.append(avg_train_loss)
            
            # Validasi
            val_output = self.forward(X_val)
            val_loss = np.mean((val_output - y_val) ** 2)
            val_losses.append(val_loss)
            
            if verbose and (epoch+1) % 20 == 0:
                print(f"Epoch {epoch+1}/{epochs} - loss: {avg_train_loss:.6f} - val_loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Simpan bobot terbaik
                best_weights = [w.copy() for w in self.weights]
                best_biases = [b.copy() for b in self.biases]
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping di epoch {epoch+1}")
                    break
        
        # Kembalikan ke bobot terbaik
        if best_weights is not None:
            self.weights = best_weights
            self.biases = best_biases
        
        return train_losses, val_losses
    
    def predict(self, X):
        """Prediksi dengan model yang sudah dilatih"""
        return self.forward(X)


# ========== 2. LOAD DATASET ==========
print("📂 Loading dataset...")

# Buat data sintetis jika dataset tidak ada
import os
if not os.path.exists('data/earthquake.csv'):
    print("⚠️ Dataset tidak ditemukan, membuat data sintetis...")
    np.random.seed(42)
    n_samples = 1000
    data = {
        'latitude': np.random.uniform(-90, 90, n_samples),
        'longitude': np.random.uniform(-180, 180, n_samples),
        'depth': np.random.uniform(0, 700, n_samples),
        'mag': 4 + np.random.exponential(1, n_samples) + 
               np.random.normal(0, 0.5, n_samples) +
               (np.random.uniform(-90, 90, n_samples) / 90) * 2
    }
    df = pd.DataFrame(data)
    df['mag'] = df['mag'].clip(3, 9.5)
    print(f"✅ Data sintetis dibuat: {len(df)} sampel")
else:
    df = pd.read_csv('data/earthquake.csv')
    print(f"✅ Dataset loaded: {df.shape}")

# ========== 3. SELEKSI FITUR ==========
# Sesuaikan nama kolom dengan dataset Anda
possible_lat = ['latitude', 'Latitude', 'lat', 'Lat']
possible_lon = ['longitude', 'Longitude', 'lon', 'Lon', 'long']
possible_depth = ['depth', 'Depth', 'd']
possible_mag = ['mag', 'Magnitude', 'magnitude', 'Mag', 'magitude']

lat_col = next((c for c in possible_lat if c in df.columns), None)
lon_col = next((c for c in possible_lon if c in df.columns), None)
depth_col = next((c for c in possible_depth if c in df.columns), None)
mag_col = next((c for c in possible_mag if c in df.columns), None)

if lat_col and lon_col and depth_col and mag_col:
    X = df[[lat_col, lon_col, depth_col]].values
    y = df[mag_col].values.reshape(-1, 1)
    print(f"✅ Kolom ditemukan: {lat_col}, {lon_col}, {depth_col}, {mag_col}")
else:
    print("⚠️ Kolom tidak ditemukan, menggunakan indeks default")
    X = df.iloc[:, :3].values
    y = df.iloc[:, -1].values.reshape(-1, 1)

print(f"Data shape: X={X.shape}, y={y.shape}")

# ========== 4. PREPROCESSING ==========
# Normalisasi
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=42
)

print(f"Training: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

# ========== 5. BANGUN MODEL ==========
print("\n🏗️ Membangun Neural Network...")
nn = NeuralNetwork(
    input_size=3,           # latitude, longitude, depth
    hidden_sizes=[64, 32, 16],  # 3 hidden layers
    output_size=1,
    learning_rate=0.01
)

# ========== 6. TRAINING ==========
print("\n🚀 Memulai training...")
train_losses, val_losses = nn.train(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    patience=10,
    verbose=True
)

# ========== 7. EVALUASI ==========
print("\n📊 Evaluasi model...")
y_pred_scaled = nn.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

mse = mean_squared_error(y_test_actual, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_actual, y_pred)

print(f"\n📈 Hasil Evaluasi:")
print(f"   - MSE: {mse:.4f}")
print(f"   - RMSE: {rmse:.4f}")
print(f"   - R² Score: {r2:.4f}")

# ========== 8. VISUALISASI ==========
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Model Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(y_test_actual, y_pred, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel('Actual Magnitude')
plt.ylabel('Predicted Magnitude')
plt.title('Prediction vs Actual')
plt.tight_layout()
plt.savefig('static/training_result.png', dpi=100)
plt.show()

print("\n✅ Grafik disimpan ke 'static/training_result.png'")

# ========== 9. SIMPAN MODEL ==========
# Simpan bobot dan scaler
model_data = {
    'weights': [w.tolist() for w in nn.weights],
    'biases': [b.tolist() for b in nn.biases],
    'hidden_sizes': [64, 32, 16],
    'learning_rate': 0.01
}
joblib.dump(model_data, 'models/model_weights.pkl')
joblib.dump(scaler_X, 'models/scaler_X.pkl')
joblib.dump(scaler_y, 'models/scaler_y.pkl')

print("\n✅ Model dan scaler berhasil disimpan di folder 'models/'")

# Test prediksi sederhana
print("\n🔮 Test prediksi:")
test_input = np.array([[0, 0, 100]])  # latitude 0, longitude 0, depth 100
test_scaled = scaler_X.transform(test_input)
test_pred = nn.predict(test_scaled)
test_actual = scaler_y.inverse_transform(test_pred)
print(f"   Input: lat=0, lon=0, depth=100 km")
print(f"   Prediksi magnitudo: {test_actual[0][0]:.2f}")