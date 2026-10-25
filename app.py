import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.datasets import load_wine

st.set_page_config(page_title="Dashboard Analisis Multivariat", layout="wide")
st.title("Dashboard Analisis Multivariat")
st.caption("Eksplorasi, visualisasi, uji normalitas, dan deteksi outlier multivariat")


# Fungsi bantu
def mardia_test(X):
    """Uji normalitas multivariat Mardia (skewness & kurtosis)."""
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    Xc = X - X.mean(axis=0)
    S = (Xc.T @ Xc) / n
    D = Xc @ np.linalg.pinv(S) @ Xc.T
    b1 = (D ** 3).sum() / n ** 2
    b2 = (np.diag(D) ** 2).sum() / n
    df = p * (p + 1) * (p + 2) / 6
    stat_skew = n * b1 / 6
    p_skew = 1 - stats.chi2.cdf(stat_skew, df)
    z_kurt = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    p_kurt = 2 * (1 - stats.norm.cdf(abs(z_kurt)))
    return pd.DataFrame({
        "Uji": ["Mardia Skewness", "Mardia Kurtosis"],
        "Statistik": [stat_skew, z_kurt],
        "p-value": [p_skew, p_kurt],
        "Kesimpulan": [
            "Normal" if p_skew > 0.05 else "Tidak normal",
            "Normal" if p_kurt > 0.05 else "Tidak normal",
        ],
    })


def mahalanobis_d2(X):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0)
    S = np.cov(X, rowvar=False)
    return np.einsum("ij,jk,ik->i", Xc, np.linalg.pinv(S), Xc)


# Sidebar: data
st.sidebar.header("Data")
file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx"])

if file is not None:
    if file.name.endswith(".csv"):
        raw = pd.read_csv(file, sep=None, engine="python")
    else:
        raw = pd.read_excel(file)
else:
    raw = load_wine(as_frame=True).frame
    st.sidebar.info("Belum upload data, memakai dataset contoh (Wine).")

num_cols_all = raw.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in raw.columns if c not in num_cols_all]

st.sidebar.subheader("Pilih variabel numerik")
num_cols = st.sidebar.multiselect(
    "Variabel untuk analisis", num_cols_all, default=num_cols_all[: min(6, len(num_cols_all))]
)

if len(num_cols) < 2:
    st.warning("Pilih minimal 2 variabel numerik di sidebar.")
    st.stop()

df = raw[num_cols].dropna().reset_index(drop=True)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Eksplorasi Data", "Visualisasi", "Uji Normalitas", "Deteksi Outlier"]
)

# Tab 1: Eksplorasi
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah observasi", len(raw))
    c2.metric("Variabel numerik", len(num_cols_all))
    c3.metric("Variabel kategorik", len(cat_cols))
    c4.metric("Missing values", int(raw.isna().sum().sum()))

    st.subheader("Cuplikan data")
    st.dataframe(raw.head(20), use_container_width=True)

    st.subheader("Statistik deskriptif")
    desc = df.describe().T
    desc["skewness"] = df.skew()
    desc["kurtosis"] = df.kurt()
    st.dataframe(desc.round(3), use_container_width=True)

    st.subheader("Matriks korelasi")
    metode = st.radio("Metode", ["pearson", "spearman"], horizontal=True)
    st.dataframe(df.corr(method=metode).round(3), use_container_width=True)

# Tab 2: Visualisasi
with tab2:
    st.subheader("Heatmap korelasi")
    corr = df.corr(method=metode)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Distribusi variabel")
    v = st.selectbox("Pilih variabel", num_cols)
    a, b = st.columns(2)
    a.plotly_chart(px.histogram(df, x=v, nbins=30, marginal="rug"), use_container_width=True)
    b.plotly_chart(px.box(df, y=v, points="outliers"), use_container_width=True)

    st.subheader("Boxplot semua variabel (distandarisasi)")
    z = (df - df.mean()) / df.std()
    st.plotly_chart(px.box(z.melt(var_name="Variabel", value_name="Z-score"),
                           x="Variabel", y="Z-score"), use_container_width=True)

    st.subheader("Scatter plot")
    x1, y1 = st.columns(2)
    sx = x1.selectbox("Sumbu X", num_cols, index=0)
    sy = y1.selectbox("Sumbu Y", num_cols, index=1)
    st.plotly_chart(px.scatter(df, x=sx, y=sy, trendline="ols"), use_container_width=True)

    st.subheader("Scatter matrix")
    st.plotly_chart(px.scatter_matrix(df, dimensions=num_cols[:6]), use_container_width=True)

# Tab 3: Normalitas
with tab3:
    st.subheader("Normalitas univariat (Shapiro-Wilk)")
    sw = []
    for c in num_cols:
        x = df[c].sample(min(len(df), 5000), random_state=1)
        s, p = stats.shapiro(x)
        sw.append([c, s, p, "Normal" if p > 0.05 else "Tidak normal"])
    st.dataframe(pd.DataFrame(sw, columns=["Variabel", "W", "p-value", "Kesimpulan"]).round(4),
                 use_container_width=True)

    st.subheader("Normalitas multivariat (Mardia)")
    st.dataframe(mardia_test(df).round(4), use_container_width=True)
    st.caption("H0: data berdistribusi normal multivariat. Tolak H0 jika p-value < 0,05.")

    st.subheader("Q-Q plot variabel")
    qv = st.selectbox("Variabel Q-Q plot", num_cols, key="qq")
    (osm, osr), (slope, intercept, _) = stats.probplot(df[qv], dist="norm")
    fig = go.Figure()
    fig.add_scatter(x=osm, y=osr, mode="markers", name="Data")
    fig.add_scatter(x=osm, y=slope * np.array(osm) + intercept, mode="lines", name="Garis normal")
    fig.update_layout(xaxis_title="Kuantil teoritis", yaxis_title="Kuantil sampel")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Chi-square Q-Q plot (multivariat)")
    d2 = np.sort(mahalanobis_d2(df))
    n, p = df.shape
    teori = stats.chi2.ppf((np.arange(1, n + 1) - 0.5) / n, df=p)
    fig = go.Figure()
    fig.add_scatter(x=teori, y=d2, mode="markers", name="D²")
    fig.add_scatter(x=teori, y=teori, mode="lines", name="Garis y = x")
    fig.update_layout(xaxis_title="Kuantil chi-square", yaxis_title="Jarak Mahalanobis D²")
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Outlier
with tab4:
    st.subheader("Deteksi outlier multivariat (jarak Mahalanobis)")
    alpha = st.slider("Taraf signifikansi (alpha)", 0.001, 0.10, 0.05, 0.001)
    d2 = mahalanobis_d2(df)
    cutoff = stats.chi2.ppf(1 - alpha, df=df.shape[1])
    hasil = df.copy()
    hasil["D2_Mahalanobis"] = d2
    hasil["Outlier"] = d2 > cutoff

    c1, c2, c3 = st.columns(3)
    c1.metric("Nilai kritis (chi-square)", f"{cutoff:.3f}")
    c2.metric("Jumlah outlier", int(hasil["Outlier"].sum()))
    c3.metric("Persentase", f"{hasil['Outlier'].mean() * 100:.2f}%")

    fig = px.scatter(hasil, x=hasil.index, y="D2_Mahalanobis", color="Outlier",
                     color_discrete_map={False: "#4F46E5", True: "#EC4899"})
    fig.add_hline(y=cutoff, line_dash="dash", line_color="red")
    fig.update_layout(xaxis_title="Indeks observasi")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Observasi yang terdeteksi outlier")
    st.dataframe(hasil[hasil["Outlier"]].round(3), use_container_width=True)

    st.download_button(
        "Unduh hasil (CSV)",
        hasil.to_csv(index=False).encode("utf-8"),
        file_name="hasil_outlier_mahalanobis.csv",
        mime="text/csv",
    )
