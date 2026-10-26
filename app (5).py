
import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px

# KONFIGURASI HALAMAN

st.set_page_config(
    page_title="Iris Dashboard",
    page_icon="📊",
    layout="wide"
)

# JUDUL DASHBOARD

st.title("My Iris Dataset Dashboard 🌺")

st.write(
    """
    Dashboard ini digunakan untuk mengeksplorasi
    hubungan antarvariabel pada dataset Iris melalui
    visualisasi data interaktif.
    """
)

st.divider()

# MEMUAT DATASET

@st.cache_data
def load_data():
    return sns.load_dataset("iris")

df = load_data()

# Mendefinisikan atribut dan spesies
attributes = df.columns[:-1].tolist()
species = df["species"].unique().tolist()

# INFORMASI DATASET

with st.expander("About the Iris Dataset"):

    st.header("About the Iris Dataset")

    st.write(
        """
        The Iris dataset consists of 150 observations
        from three species of iris flowers: Setosa,
        Versicolor, and Virginica.

        Each observation contains four numerical
        attributes: sepal length, sepal width,
        petal length, and petal width.

        The dataset is commonly used to demonstrate
        exploratory data analysis, data visualization,
        and classification.
        """
    )

# SIDEBAR: PILIHAN VARIABEL DAN SPESIES

with st.sidebar:

    st.header("Filters")

    # Pilihan atribut
    selected_attribute = st.selectbox(
        "Select Attribute:",
        attributes,
        index=0
    )

    # Pilihan spesies
    selected_species = st.multiselect(
        "Select Species:",
        species,
        default=species,
        placeholder="Filter by species"
    )

# Memfilter data berdasarkan spesies yang dipilih
filtered_df = df[
    df["species"].isin(selected_species)
]

# STATISTIK DESKRIPTIF

st.header("Basic Statistics")

st.write(
    "Statistik deskriptif untuk empat variabel numerik."
)

st.dataframe(
    filtered_df[attributes].describe(),
    use_container_width=True
)

# FUNGSI VISUALISASI PAIRWISE SCATTER PLOT

def create_pairplot(data, attributes):

    fig = px.scatter_matrix(
        data,
        dimensions=attributes,
        color="species",
        opacity=0.6,
        title="Pairwise Scatter Plot"
    )

    fig.update_traces(
        diagonal_visible=False
    )

    fig.update_layout(
        height=750
    )

    return fig

# FUNGSI HISTOGRAM

def create_histogram(data, attribute):

    fig = px.histogram(
        data,
        x=attribute,
        color="species",
        marginal="box",
        barmode="overlay",
        opacity=0.7,
        title=f"Histogram of {attribute}"
    )

    fig.update_layout(
        hovermode="x unified"
    )

    return fig

# FUNGSI VIOLIN PLOT

def create_violin_plot(
    data,
    attribute,
    show_points=True
):

    points = "all" if show_points else False

    fig = px.violin(
        data,
        x="species",
        y=attribute,
        color="species",
        box=True,
        points=points,
        title=f"Violin Plot of {attribute}"
    )

    return fig

# VISUALISASI BIVARIAT: SCATTER MATRIX

st.header("Data Visualization")

st.subheader(
    "Pairwise Scatter Plot (All Attributes)"
)

st.write(
    """
    Scatter matrix digunakan untuk mengeksplorasi
    hubungan setiap pasangan variabel numerik.
    Warna menunjukkan spesies bunga.
    """
)

if filtered_df.empty:

    st.warning(
        "Pilih minimal satu spesies pada sidebar."
    )

else:

    pairplot_fig = create_pairplot(
        filtered_df,
        attributes
    )

    st.plotly_chart(
        pairplot_fig,
        use_container_width=True
    )

# VISUALISASI DISTRIBUSI ATRIBUT

st.subheader("Single Attribute Distribution")

st.write(
    """
    Pilih satu atribut untuk membandingkan distribusinya
    pada ketiga spesies bunga.
    """
)

if not filtered_df.empty:

    col1, col2 = st.columns(2)

    # Histogram
    with col1:

        hist_fig = create_histogram(
            filtered_df,
            selected_attribute
        )

        st.plotly_chart(
            hist_fig,
            use_container_width=True
        )

    # Violin plot
    with col2:

        show_points = st.checkbox(
            "Show individual points",
            value=True
        )

        violin_fig = create_violin_plot(
            filtered_df,
            selected_attribute,
            show_points
        )

        st.plotly_chart(
            violin_fig,
            use_container_width=True
        )

# TABEL DATA

with st.expander("View Dataset"):

    st.dataframe(
        filtered_df,
        use_container_width=True
    )

st.divider()

st.caption(
    "Praktikum Eksplorasi dan Visualisasi Data Bivariat"
)
