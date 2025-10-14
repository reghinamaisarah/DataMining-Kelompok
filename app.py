import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as pex
import plotly.graph_objects as go
import io
from datetime import datetime

st.set_page_config(page_title="K-Means Clustering – Stres Mahasiswa", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .metric-card {
        background: #f5f0eb;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(101, 67, 33, 0.1);
        border: 1px solid #d4c4b0;
    }
    .metric-card h3 {
        color: #654321;
        margin: 0.5rem 0 0 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .metric-card b {
        color: #8B4513;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .note {
        background: #faf7f4;
        border-left: 4px solid #A0826D;
        padding: 1rem;
        border-radius: 8px;
        color: #5a4a3a;
        margin: 1rem 0;
    }
    .warn {
        background: #fff9f0;
        border-left: 4px solid #D2691E;
        padding: 1rem;
        border-radius: 8px;
        color: #5a4a3a;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f5f0eb;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #654321;
        font-weight: 500;
        padding: 0.75rem 1.5rem;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #A0826D 0%, #8B6F47 100%);
        color: white;
    }
    .main-header {
        background: #F5F0EB;
        padding: 1rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(101, 67, 33, 0.3);
        color: #412E27 !important;
        text-align: center;
    }
    .main-header h1 {
        margin: 0 0 0.5rem 0;
        font-size: 2rem;
        font-weight: 600;
    }
    .main-header p {
        margin: 0;
        opacity: 0.95;
        line-height: 1.6;
    }
    .stButton > button {
        background: #7C6259;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: #A7958E;
        box-shadow: 0 4px 12px rgba(101, 67, 33, 0.3);
    }
    .stMetric {
        background: #faf7f4;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e8dfd5;
    }
    .stMetric label {
        color: #8B4513;
        font-weight: 500;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #654321;
    }
    h3, h4 {
        color: #654321;
        font-weight: 600;
        margin-top: 2rem;
    }
    .stDataFrame {
        border: 1px solid #e8dfd5;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True
)

st.markdown(
    f"""
    <div class="main-header">
      <h1>Klasterisasi Tingkat Stres Mahasiswa </h1>
    </div>
    """, unsafe_allow_html=True
)

# Data Loader
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data.csv")
    except FileNotFoundError:
        st.error("File 'data.csv' tidak ditemukan. Taruh file di folder yang sama.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Gagal membaca data.csv: {e}")
        return pd.DataFrame()

df_raw = load_data()
if df_raw.empty:
    st.stop()

# Util Standarisasi, PCA, Silhouette
def zscore_scale(X):
    X = np.asarray(X, dtype=float)
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / (sd + 1e-9), mu, sd

def pca_2d(X):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc, rowvar=False)
    cov = np.nan_to_num(cov, nan=0.0)
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        U, S, VT = np.linalg.svd(Xc, full_matrices=False)
        eigvecs = VT.T
        eigvals = (S**2)/(X.shape[0]-1)
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:2]]
    X2 = Xc @ W
    ratio = eigvals[idx[:2]] / (eigvals.sum() + 1e-12)
    return X2, ratio, W

def silhouette_avg(X, labels):
    X = np.asarray(X); labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2: return 0.0
    n = len(X)
    s = np.zeros(n)
    for i in range(n):
        same = X[labels == labels[i]]
        a = np.mean([np.linalg.norm(X[i] - p) for p in same if not np.array_equal(p, X[i])]) if len(same) > 1 else 0.0
        b = min([
            np.mean([np.linalg.norm(X[i] - p) for p in X[labels == c]])
            for c in uniq if c != labels[i]
        ]) if len(uniq) > 1 else 0.0
        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    return float(np.mean(s))

def silhouette_per_sample(X, labels):
    X = np.asarray(X); labels = np.asarray(labels)
    uniq = np.unique(labels)
    if len(uniq) < 2: return np.zeros(len(X))
    s = np.zeros(len(X))
    for i in range(len(X)):
        same = X[labels == labels[i]]
        a = np.mean([np.linalg.norm(X[i]-p) for p in same if not np.array_equal(p,X[i])]) if len(same)>1 else 0.0
        others = [np.mean([np.linalg.norm(X[i]-p) for p in X[labels==c]]) for c in uniq if c!=labels[i]]
        b = min(others) if others else 0.0
        s[i] = (b-a)/max(a,b) if max(a,b)>0 else 0.0
    return s

# K-Means Clustering
def kmeans_fit(X, k=3, max_iter=300, tol=1e-4, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, k, replace=False)
    C = X[idx].copy()
    inertia_hist = []
    for _ in range(max_iter):
        d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        labels = np.argmin(d, axis=1)
        newC = np.array([X[labels == j].mean(axis=0) if np.any(labels == j) else C[j] for j in range(k)])
        inertia = float(np.sum((X - newC[labels])**2))
        inertia_hist.append(inertia)
        if np.linalg.norm(newC - C) <= tol:
            C = newC
            break
        C = newC
    return labels, C, inertia_hist

# Data Preprocessing 
EXCLUDE_COLS = [
    "Timestamp",
    "Nama",
    "Jenis Kelamin",
    "Program Studi",
    "Tingkat Semester (angka saja, contoh: 3)",
]

def preprocess_pipeline(df):
    rep = {}
    dfp = df.copy()
    
    # Bersihkan whitespace pada kolom kategoris (string)
    for col in dfp.select_dtypes(include=['object']).columns:
        if dfp[col].dtype == 'object':
            dfp[col] = dfp[col].str.strip() 

    # 1) Duplikat
    before = len(dfp)
    dfp = dfp.drop_duplicates().reset_index(drop=True)
    rep["duplicates_removed"] = int(before - len(dfp))

    # 2) Missing (sebelum)
    rep["missing_before"] = int(dfp.isnull().sum().sum())

    # 3) Imputasi sederhana (jika ada)
    for c in dfp.columns:
        if pd.api.types.is_numeric_dtype(dfp[c]):
            if dfp[c].isnull().any():
                dfp[c] = dfp[c].fillna(dfp[c].mean())
        else:
            if dfp[c].isnull().any():
                mode_val = dfp[c].mode().iloc[0] if not dfp[c].mode().empty else "Unknown"
                dfp[c] = dfp[c].fillna(mode_val)

    # 4) Tentukan fitur numerik untuk K-Means (exclude kolom demografi/identitas)
    num_cols_all = dfp.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols_all if c not in EXCLUDE_COLS]

    # 5) Z-Score untuk fitur (buat suffix _standardized) + siapkan matriks fitur standardized
    X_src = dfp[feature_cols].values if feature_cols else np.empty((len(dfp), 0))
    X_std, mu, sd = zscore_scale(X_src) if X_src.size else (X_src, None, None)
    std_cols = []
    for i, c in enumerate(feature_cols):
        dfp[c + "_standardized"] = X_std[:, i]
        std_cols.append(c + "_standardized")

    # 6) Missing (sesudah)
    rep["missing_after"] = int(dfp.isnull().sum().sum())
    rep["std_features_count"] = len(std_cols)
    rep["feature_cols_used"] = feature_cols

    # kolom deskriptif untuk analisis (disimpan di dfp, tapi tidak ikut perhitungan)
    rep["descriptive_cols"] = [c for c in ["Timestamp","Nama","Jenis Kelamin","Program Studi","Tingkat Semester (angka saja, contoh: 3)"] if c in dfp.columns]

    return dfp, std_cols, rep

# Jalankan preprocessing saat startup (sekali)
df_proc, STD_COLS, PREP_REPORT = preprocess_pipeline(df_raw)

COLORFUL_PALETTE = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8',
    '#F7DC6F', '#BB8FCE', '#85C1E2', '#F8B195', '#FF8C94',
    '#95E1D3', '#C44569', '#6BCB77', '#4D96FF', '#FFD93D'
]

# Daftar Tabs 
tab1, tab2, tab4, tab3 = st.tabs(["Eksplorasi Data", "Preprocessing", "Evaluasi", "K-Means"])

# Tab 1: Eksplorasi
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f"<div class='metric-card'><b>Total Data</b><h3>{len(df_raw)}</h3></div>", unsafe_allow_html=True)
    with c2: st.markdown(f"<div class='metric-card'><b>Jumlah Kolom</b><h3>{df_raw.shape[1]}</h3></div>", unsafe_allow_html=True)
    with c3: st.markdown(f"<div class='metric-card'><b>Kolom Numerik</b><h3>{df_raw.select_dtypes(include=[np.number]).shape[1]}</h3></div>", unsafe_allow_html=True)
    with c4: st.markdown(f"<div class='metric-card'><b>Missing Values</b><h3>{int(df_raw.isnull().sum().sum())}</h3></div>", unsafe_allow_html=True)

    st.markdown("### Preview Data")
    st.dataframe(df_raw.head(20), use_container_width=True, height=400)
    with st.expander("Lihat seluruh data (scrollable)"):
        st.dataframe(df_raw, use_container_width=True, height=500)

    # Visualisasi Jenis Kelamin and Semester
    row1 = st.columns(2)
    with row1[0]:
        if "Jenis Kelamin" in df_raw.columns:
            vc = df_raw["Jenis Kelamin"].value_counts()
            fig = pex.pie(values=vc.values, names=vc.index, title="Distribusi Jenis Kelamin", hole=.4,
                         color_discrete_sequence=['#A0826D', '#8B6F47', '#D4A574'])
            fig.update_layout(font=dict(color='#654321'))
            st.plotly_chart(fig, use_container_width=True)
    with row1[1]:
        semcol = "Tingkat Semester (angka saja, contoh: 3)"
        if semcol in df_raw.columns:
            valid_sem = [1, 3, 5, 7, 9]
            vc = df_raw[semcol].value_counts().reindex(valid_sem, fill_value=0)
            fig = pex.bar(x=vc.index, y=vc.values, labels={"x":"Semester","y":"Jumlah"}, title="Distribusi Semester",
                         color_discrete_sequence=['#8B4513'])
            fig.update_layout(font=dict(color='#654321'), xaxis=dict(tickvals=valid_sem, ticktext=[str(s) for s in valid_sem]))
            st.plotly_chart(fig, use_container_width=True)

    # Visualisasi Program Studi
    if "Program Studi" in df_raw.columns:
        vc = df_raw["Program Studi"].value_counts()
        fig = pex.bar(x=vc.index, y=vc.values, color=vc.index, labels={"x":"Prodi","y":"Jumlah"}, title="Distribusi Prodi")
        fig.update_layout(font=dict(color='#654321'), height=400)
        st.plotly_chart(fig, use_container_width=True)

# Tab 2: Preprocessing 
with tab2:
    st.markdown("### Ringkasan Preprocessing")
    st.markdown(
        """
        <div class='note'>
        Urutan: (1) Hapus duplikat → (2) Imputasi sederhana (jika ada) → (3) Tentukan fitur numerik relevan →
        (4) Standarisasi Z-Score. <br>
        Kolom berikut <strong>TIDAK</strong> dipakai untuk perhitungan K-Means: 
        <code>Timestamp</code>, <code>Nama</code>, <code>Jenis Kelamin</code>, <code>Program Studi</code>, 
        <code>Tingkat Semester (angka saja, contoh: 3)</code>.
        </div>
        """, unsafe_allow_html=True
    )
    m1, m2, m3, m4 = st.columns(4)
    with m1: st.metric("Duplikat dihapus", PREP_REPORT["duplicates_removed"])
    with m2: st.metric("Missing sebelum", PREP_REPORT["missing_before"])
    with m3: st.metric("Missing sesudah", PREP_REPORT["missing_after"])
    with m4: st.metric("Fitur dipakai (std)", PREP_REPORT["std_features_count"])

    st.markdown("### Fitur yang digunakan untuk K-Means")
    st.write(PREP_REPORT["feature_cols_used"] if PREP_REPORT["feature_cols_used"] else "-")

    st.markdown("### Matriks Fitur (Standardized) – untuk K-Means (preview)")
    if len(STD_COLS) >= 2:
        st.dataframe(df_proc[STD_COLS].head(30), use_container_width=True, height=400)
        with st.expander("Lihat seluruh matriks fitur (scrollable)"):
            st.dataframe(df_proc[STD_COLS], use_container_width=True, height=500)
        buf_feat = io.StringIO()
        df_proc[STD_COLS].to_csv(buf_feat, index=False)
        st.download_button("Unduh Matriks Fitur (standardized CSV)", data=buf_feat.getvalue(),
                           file_name="features_standardized.csv", mime="text/csv", use_container_width=True)
    else:
        st.warning("Fitur standar < 2.")

    st.markdown("### Kolom Deskriptif (tersimpan untuk analisis, tidak ikut perhitungan)")
    st.write(PREP_REPORT["descriptive_cols"] if PREP_REPORT["descriptive_cols"] else "-")

# Tab 3 : Evaluasi 
with tab4:
    st.markdown("### Evaluasi Model – Elbow & Silhouette vs K")
    st.markdown(
        """
        <div class='note'>
        <strong>Elbow:</strong> cari titik siku pada WCSS vs K (penurunan mulai melambat). <br>
        <strong>Silhouette:</strong> -1..1 (semakin tinggi semakin baik). <br>
        <strong>Catatan:</strong> Minimal K valid untuk evaluasi adalah 2.
        </div>
        """, unsafe_allow_html=True
    )

    if len(STD_COLS) < 2:
        st.warning("Fitur standar < 2. Evaluasi tidak dapat dihitung.")
    else:
        maxK = st.slider("Uji sampai K =", 2, 12, 8, key="eval_maxK")
        if st.button("Jalankan Evaluasi K (Elbow & Silhouette)", use_container_width=True, key="btn_eval"):
            X = df_proc[STD_COLS].to_numpy()
            Ks = list(range(2, maxK+1))
            WCSS, SIL = [], []
            for k in Ks:
                lab, C, hist = kmeans_fit(X, k=k, seed=42)
                WCSS.append(hist[-1] if hist else 0.0)
                SIL.append(silhouette_avg(X, lab))
            st.session_state["eval_Ks"] = Ks
            st.session_state["eval_WCSS"] = WCSS
            st.session_state["eval_SIL"] = SIL
            st.success("Evaluasi selesai.")

        if "eval_Ks" in st.session_state:
            Ks = st.session_state["eval_Ks"]
            WCSS = st.session_state["eval_WCSS"]
            SIL = st.session_state["eval_SIL"]

            # Grafik Elbow & Silhouette
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=Ks, y=WCSS, mode="lines+markers", name="WCSS",
                                    line=dict(color='#FF6B6B', width=3),
                                    marker=dict(size=8, color='#FF6B6B')))
            fig.add_trace(go.Scatter(x=Ks, y=SIL, mode="lines+markers", name="Silhouette", yaxis="y2",
                                    line=dict(color='#4ECDC4', width=3),
                                    marker=dict(size=8, color='#4ECDC4')))
            fig.update_layout(
                title="Elbow (WCSS) & Silhouette vs K",
                xaxis_title="K",
                yaxis=dict(title="WCSS", titlefont=dict(color='#FF6B6B')),
                yaxis2=dict(title="Silhouette", overlaying="y", side="right", titlefont=dict(color='#4ECDC4')),
                height=480,
                font=dict(color='#654321'),
                plot_bgcolor='#faf7f4',
                paper_bgcolor='#ffffff'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Rekomendasi 1: Silhouette terbaik 
            bestK_sil = Ks[int(np.argmax(SIL))]

            # Rekomendasi 2: Elbow terbaik (metode jarak-ke-garis) 
            x1, y1 = Ks[0], WCSS[0]
            x2, y2 = Ks[-1], WCSS[-1]
            vx, vy = (x2 - x1), (y2 - y1)
            denom = np.sqrt(vx*vx + vy*vy) + 1e-12
            dists = []
            for k, w in zip(Ks, WCSS):
                px, py = (k - x1), (w - y1)
                cross = abs(vx*py - vy*px)
                dists.append(cross / denom)
            bestK_elbow = Ks[int(np.argmax(dists))]

            st.success(f"Rekomendasi Silhouette: **K = {bestK_sil}** (Silhouette tertinggi).")
            st.info(f"Rekomendasi Elbow: **K = {bestK_elbow}** (titik siku WCSS).")

            # Stabilitas Silhouette antar seed
            st.markdown("### Stabilitas Silhouette antar Seed")
            X = df_proc[STD_COLS].to_numpy()
            rows = []
            seeds = [0, 7, 21, 42, 77]
            for k in Ks:
                vals = []
                for sd in seeds:
                    lab, C, _ = kmeans_fit(X, k=k, seed=sd)
                    vals.append(silhouette_avg(X, lab))
                rows.append({"K": k, "Silhouette mean": float(np.mean(vals)), "Silhouette std": float(np.std(vals))})
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Perbandingan Silhouette per-sample K=2 vs K=3
            st.markdown("### Silhouette per-sample: K=2 vs K=3")
            cols_ps = st.columns(2)
            with cols_ps[0]:
                if 2 in Ks:
                    lab2, C2, _ = kmeans_fit(X, k=2, seed=42)
                    s2 = silhouette_per_sample(X, lab2)
                    fig2 = go.Figure(go.Bar(x=list(range(len(s2))), y=np.sort(s2),
                                           marker=dict(color='#FF6B6B')))
                    fig2.update_layout(title=f"Silhouette per-sample (K=2) | mean={s2.mean():.4f}",
                                       xaxis_title="Index Terurut", yaxis_title="Silhouette",
                                       font=dict(color='#654321'),
                                       plot_bgcolor='#faf7f4')
                    st.plotly_chart(fig2, use_container_width=True)

            with cols_ps[1]:
                if 3 in Ks:
                    lab3, C3, _ = kmeans_fit(X, k=3, seed=42)
                    s3 = silhouette_per_sample(X, lab3)
                    fig3 = go.Figure(go.Bar(x=list(range(len(s3))), y=np.sort(s3),
                                           marker=dict(color='#4ECDC4')))
                    fig3.update_layout(title=f"Silhouette per-sample (K=3) | mean={s3.mean():.4f}",
                                       xaxis_title="Index Terurut", yaxis_title="Silhouette",
                                       font=dict(color='#654321'),
                                       plot_bgcolor='#faf7f4')
                    st.plotly_chart(fig3, use_container_width=True)

            st.info("Interpretasi: Pilih K berdasar tujuan. Silhouette sering memilih K kecil (mis. 2); Elbow memberi kompromi pada titik 'siku'. Kamu boleh mengambil K domain-driven (mis. 3 level stres: rendah–sedang–tinggi).")

# Tab 4: K-Means
with tab3:
    st.markdown("### Jalankan Klasterisasi (K-Means Clustering)")
    if len(STD_COLS) < 2:
        st.warning("Fitur standar < 2. Pastikan ada minimal 2 fitur numerik relevan.")
    else:
        if "eval_SIL" in st.session_state and "eval_Ks" in st.session_state:
            Ks = st.session_state["eval_Ks"]
            SIL = st.session_state["eval_SIL"]
            startK = int(Ks[int(np.argmax(SIL))])
        else:
            startK = 3

        K = st.slider("Jumlah Cluster (K)", 2, 10, startK, key="km_k")
        seed = st.number_input("Seed (reproducible)", min_value=0, value=42, step=1, key="km_seed")
        st.caption("Centroid awal diinisialisasi acak dengan seed tetap untuk hasil reproducible.")

        if st.button("Jalankan K-Means", use_container_width=True, key="btn_kmeans"):
            X = df_proc[STD_COLS].to_numpy()
            labels, C, inertia_hist = kmeans_fit(X, k=K, seed=seed)
            sil = silhouette_avg(X, labels)
            X2, ratio, W = pca_2d(X)

            st.session_state["km_labels"] = labels
            st.session_state["km_centroids"] = C
            st.session_state["km_inertia_hist"] = inertia_hist
            st.session_state["km_sil"] = sil
            st.session_state["km_X2"] = X2
            st.session_state["km_ratio"] = ratio
            st.session_state["km_W"] = W
            st.session_state["km_K"] = K
            st.success("Clustering selesai.")

        if "km_labels" in st.session_state:
            labels = st.session_state["km_labels"]; K = st.session_state["km_K"]
            X2 = st.session_state["km_X2"]; ratio = st.session_state["km_ratio"]

            st.markdown("### Visualisasi PCA 2D")
            C = st.session_state["km_centroids"]
            W = st.session_state["km_W"]
            C_pca = C @ W

            fig = go.Figure()
            for i in range(K):
                mask = labels == i
                fig.add_trace(go.Scatter(
                    x=X2[mask, 0], y=X2[mask, 1],
                    mode='markers',
                    name=f'Cluster {i}',
                    marker=dict(size=6, color=COLORFUL_PALETTE[i % len(COLORFUL_PALETTE)])
                ))
            fig.add_trace(go.Scatter(
                x=C_pca[:, 0], y=C_pca[:, 1],
                mode='markers',
                name='Centroids',
                marker=dict(size=14, symbol='x', color='#000000', line=dict(width=2, color='white'))
            ))
            fig.update_layout(
                title=f"PCA 2D – K={K} (PC1 {ratio[0]*100:.1f}%, PC2 {ratio[1]*100:.1f}%)",
                xaxis_title="PC1",
                yaxis_title="PC2",
                height=480,
                font=dict(color='#654321'),
                plot_bgcolor='#faf7f4',
                paper_bgcolor='#ffffff'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Distribusi anggota
            st.markdown("### Distribusi Anggota Cluster")
            vc = pd.Series(labels).value_counts().sort_index()
            cols = st.columns(K)
            for i in range(K):
                with cols[i]:
                    cnt = int(vc.get(i, 0))
                    st.metric(f"Cluster {i}", f"{cnt}")

            # Metrik
            st.markdown("### Metrik Kualitas (untuk K saat ini)")
            inertia = st.session_state["km_inertia_hist"][-1] if st.session_state["km_inertia_hist"] else 0.0
            sil = st.session_state["km_sil"]
            c1, c2 = st.columns(2)
            with c1: st.metric("WCSS (Inertia)", f"{inertia:.2f}")
            with c2: st.metric("Silhouette", f"{sil:.4f}")

            # Deskriptif cluster (pakai kolom demografi – tidak ikut perhitungan)
            st.markdown("### Deskripsi Cluster (demografis – tidak ikut perhitungan)")
            demo_cols = [c for c in ["Jenis Kelamin","Program Studi","Tingkat Semester (angka saja, contoh: 3)"] if c in df_raw.columns]
            df_desc = df_raw.copy(); df_desc["Cluster"] = labels

            # Komposisi jenis kelamin
            if "Jenis Kelamin" in demo_cols:
                st.write("Komposisi **Jenis Kelamin** per cluster")
                st.dataframe(
                    df_desc.pivot_table(index="Cluster", columns="Jenis Kelamin", aggfunc="size", fill_value=0),
                    use_container_width=True
                )

            # Distribusi Semester
            sem_col = "Tingkat Semester (angka saja, contoh: 3)"
            if sem_col in demo_cols:
                st.write("Distribusi **Semester** per cluster")

                # pastikan tipe int
                df_desc[sem_col] = pd.to_numeric(df_desc[sem_col], errors="coerce").astype("Int64")

                valid_sem = [1, 3, 5, 7, 9]

                # Count
                sem_count = (
                    df_desc.groupby(["Cluster", sem_col]).size()
                    .unstack(fill_value=0)
                    .reindex(columns=valid_sem, fill_value=0)
                    .sort_index()
                )
                st.dataframe(sem_count, use_container_width=True)

                # Modus
                sem_mode = sem_count.idxmax(axis=1).to_frame("Semester Terbanyak (modus)")
                st.table(sem_mode)

            # Export CSV (gabung demografi)
            st.markdown("### Unduh Hasil")
            out = df_raw.copy()
            out["Cluster"] = labels
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Unduh Hasil (CSV)", data=buf.getvalue(),
                               file_name=f"clustering_results_k{K}.csv", mime="text/csv",
                               use_container_width=True)

st.caption("Note: Kolom demografi (Timestamp, Nama, Jenis Kelamin, Program Studi, Semester) tidak dipakai untuk perhitungan K-Means, hanya untuk deskripsi/interpretasi.")