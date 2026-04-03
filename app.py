from flask import Flask, render_template, request, redirect
import joblib
import numpy as np
import pandas as pd  # Tambahkan pandas untuk konsistensi nama fitur
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import os
from pathlib import Path

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'model_jumlah_pasien.pkl')
test_data_path = os.path.join(BASE_DIR, 'test_data.pkl')

model = None
X_all, y_all, y_pred_all, mae, mse, r2 = [None] * 6

def load_assets():
    global model, X_all, y_all, y_pred_all, mae, mse, r2
    try:
        if model is None:
            # Gunakan jalur absolut agar Vercel tidak bingung
            bundle_path = os.path.join(BASE_DIR, 'bundle_model.pkl')
            
            if os.path.exists(bundle_path):
                # Jika pakai cara bundle (rekomendasi)
                data = joblib.load(bundle_path)
                model = data['model']
                X_all, y_all, y_pred_all, mae, mse, r2 = data['stats']
                print("Berhasil memuat dari bundle_model.pkl")
            else:
                # Jika file bundle tidak ada, coba muat satuan (cadangan)
                model = joblib.load(model_path)
                X_all, y_all, y_pred_all, mae, mse, r2 = joblib.load(test_data_path)
                print("Berhasil memuat dari file pkl satuan")
        return True
    except Exception as e:
        print(f"Gagal total memuat aset: {e}")
        return False

def plot_comparison():
    if not load_assets(): return ""
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Ambil nilai tahun untuk sumbu X
    tahun_vals = X_all['tahun'].values
    plt.scatter(tahun_vals, y_all, color='#2563eb', label='Data Aktual', s=120, edgecolors='white', linewidth=2, zorder=3)
    
    # Perbaikan Peringatan: Gunakan DataFrame agar nama kolom cocok dengan saat training
    x_range_vals = np.linspace(tahun_vals.min() - 0.5, tahun_vals.max() + 0.5, 100).reshape(-1, 1)
    x_range_df = pd.DataFrame(x_range_vals, columns=['tahun']) 
    y_range = model.predict(x_range_df)
    
    plt.plot(x_range_vals, y_range, color='#ef4444', linewidth=3, label='Tren Prediksi Linear', zorder=2)
    plt.title('Visualisasi Tren Pertumbuhan Pasien', fontsize=16, fontweight='bold', pad=20, color='#1e293b')
    plt.xlabel('Tahun', fontsize=12, fontweight='600')
    plt.ylabel('Jumlah Pasien (Jiwa)', fontsize=12, fontweight='600')
    plt.legend(frameon=True, facecolor='white', shadow=True)
    plt.tight_layout()
    
    img = io.BytesIO()
    plt.savefig(img, format='png', dpi=120)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

@app.route('/')
def index():
    if load_assets():
        plot_url = plot_comparison()
        return render_template('index.html', plot_url=plot_url, mae=f"{mae:.2f}", mse=f"{mse:.2f}", r2=f"{r2:.2f}")
    return "Gagal memuat model. Pastikan file .pkl ada di root folder."

@app.route('/predict', methods=['POST'])
def predict():
    if not load_assets(): return redirect('/')
    try:
        tahun = int(request.form['tahun'])
        
        # Perbaikan Peringatan: Bungkus input ke dalam DataFrame dengan nama kolom 'tahun'
        input_df = pd.DataFrame([[tahun]], columns=['tahun'])
        pred = model.predict(input_df)[0]
        
        plot_url = plot_comparison()
        return render_template('index.html', plot_url=plot_url, mae=f"{mae:.2f}", mse=f"{mse:.2f}", r2=f"{r2:.2f}", prediksi=round(pred, 0), input_tahun=tahun)
    except Exception as e:
        print(f"Prediction error: {e}")
        return redirect('/')
app = app
if __name__ == '__main__':
    app.run(debug=True)
