# Klasterisasi Tingkat Stres Mahasiswa dengan K-Means

Aplikasi berbasis **Streamlit** untuk mengelompokkan tingkat stres mahasiswa menggunakan algoritma **K-Means Clustering**. Sistem dilengkapi dengan preprocessing data, evaluasi cluster (Elbow Method dan Silhouette Score), visualisasi PCA, serta export hasil clustering.

## Fitur

- Eksplorasi data
- Preprocessing otomatis
  - Penghapusan data duplikat
  - Penanganan missing value
  - Standarisasi Z-Score
- Evaluasi jumlah cluster
  - Elbow Method
  - Silhouette Score
- K-Means Clustering
- Visualisasi PCA 2D
- Analisis karakteristik cluster
- Export hasil clustering ke CSV

## Teknologi

- Python
- Streamlit
- Pandas
- NumPy
- Plotly

## Instalasi

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur Project

```text
├── app.py
├── data.csv
├── requirements.txt
└── README.md
```

## Alur Sistem

```text
Dataset
   ↓
Preprocessing
   ↓
Standarisasi Z-Score
   ↓
Evaluasi Cluster
   ↓
K-Means Clustering
   ↓
Visualisasi PCA
   ↓
Analisis Cluster
```

## Output

- Hasil cluster mahasiswa
- Nilai WCSS (Within Cluster Sum of Squares)
- Nilai Silhouette Score
- Visualisasi PCA 2D
- Analisis cluster
- Export hasil dalam format CSV
