import pickle
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

DATA_PATH = Path("data/02_realisasi_anggaran_klasifikasi (1).csv")
MODEL_PATH = Path("model/Best_model.pkcls")

st.set_page_config(page_title="Dashboard Analitik Realisasi Anggaran", layout="wide")
st.title("Dashboard Analitik Realisasi Anggaran")
st.markdown(
    "Dashboard interaktif untuk melihat realisasi anggaran, performa IKPA, dan hasil prediksi model klasifikasi."
)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_model():
    try:
        import Orange

        with MODEL_PATH.open("rb") as f:
            model = pickle.load(f)

        return model, None
    except ModuleNotFoundError:
        return None, "Pustaka Orange tidak ditemukan. Install `orange3` di environment."
    except FileNotFoundError:
        return None, "File model tidak ditemukan. Pastikan `model/Best_model.pkcls` ada."
    except Exception as exc:
        return None, f"Model gagal dimuat: {exc}"

@st.cache_data
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    one_hot_labels = [
        "Dekonsentrasi",
        "Kantor Daerah",
        "Kantor Pusat",
        "Tugas Pembantuan",
    ]
    features = df[["jumlah_spm", "revisi_dipa", "deviasi_rpd_persen", "skor_ikpa"]].copy()
    tipe = df["tipe_satker"].astype(str).fillna("")
    for label in one_hot_labels:
        features[f"tipe_satker={label}"] = (tipe == label).astype(int)
    return features

@st.cache_data
def predict_orange(model, df: pd.DataFrame) -> list:
    import Orange

    domain = getattr(model, "domain", None)
    if domain is None:
        raise ValueError("Model Orange tidak memiliki domain prediktor.")

    features = prepare_features(df)
    table = Orange.data.Table.from_numpy(domain, features.to_numpy(), None)
    predictions = model(table)
    return [str(pred.value) if hasattr(pred, "value") else str(pred) for pred in predictions]

# Load data and model
with st.spinner("Memuat data dan model..."):
    data = load_data()
    model, model_error = load_model()

st.sidebar.header("Filter Data")
with st.sidebar.form("filter_form"):
    provinsi_filter = st.multiselect(
        "Provinsi",
        options=sorted(data["provinsi"].dropna().unique()),
        default=sorted(data["provinsi"].dropna().unique()),
    )
    kementerian_filter = st.multiselect(
        "Kementerian",
        options=sorted(data["nama_kementerian"].dropna().unique()),
        default=sorted(data["nama_kementerian"].dropna().unique()),
    )
    tipe_filter = st.multiselect(
        "Tipe Satker",
        options=sorted(data["tipe_satker"].dropna().unique()),
        default=sorted(data["tipe_satker"].dropna().unique()),
    )
    jenis_filter = st.multiselect(
        "Jenis Belanja Utama",
        options=sorted(data["jenis_belanja_utama"].dropna().unique()),
        default=sorted(data["jenis_belanja_utama"].dropna().unique()),
    )
    realisasi_filter = st.multiselect(
        "Realisasi tercapai 95%",
        options=sorted(data["realisasi_tercapai_95persen"].dropna().unique()),
        default=sorted(data["realisasi_tercapai_95persen"].dropna().unique()),
    )

    pagu_min = float(data["pagu_miliar"].min())
    pagu_max = float(data["pagu_miliar"].max())
    pagu_range = st.slider(
        "Rentang Pagu (miliar)",
        min_value=pagu_min,
        max_value=pagu_max,
        value=(pagu_min, pagu_max),
        step=0.1,
    )

    submit_filter = st.form_submit_button("Submit Filter")

if submit_filter:
    st.sidebar.success("Filter diterapkan")

filtered_data = data[
    data["provinsi"].isin(provinsi_filter)
    & data["nama_kementerian"].isin(kementerian_filter)
    & data["tipe_satker"].isin(tipe_filter)
    & data["jenis_belanja_utama"].isin(jenis_filter)
    & data["realisasi_tercapai_95persen"].isin(realisasi_filter)
    & data["pagu_miliar"].between(*pagu_range)
]

st.subheader("Ringkasan Data")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Jumlah Satker", len(filtered_data))
col2.metric(
    "Rata-rata Skor IKPA",
    f"{filtered_data['skor_ikpa'].mean():.2f}" if not filtered_data.empty else "-",
)
col3.metric(
    "Rata-rata Realisasi TW3",
    f"{filtered_data['realisasi_tw3_persen'].mean():.2f}%" if not filtered_data.empty else "-",
)
col4.metric(
    "Tercapai 95%",
    f"{(filtered_data['realisasi_tercapai_95persen'] == 'Ya').mean() * 100:.1f}%"
    if not filtered_data.empty
    else "-",
)

st.markdown("---")
with st.expander("Detail Statistik dan Ringkasan", expanded=True):
    st.write("#### Distribusi dan ringkasan nilai")
    st.bar_chart(
        filtered_data["provinsi"].value_counts().head(15),
        width="stretch",
    )
    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.write("##### Rata-rata skor IKPA per tipe satker")
        st.bar_chart(
            filtered_data.groupby("tipe_satker")["skor_ikpa"].mean(),
            width="stretch",
        )
    with chart_cols[1]:
        st.write("##### Proporsi realisasi tercapai 95%")
        st.bar_chart(
            filtered_data["realisasi_tercapai_95persen"].value_counts(),
            width="stretch",
        )

st.markdown("---")
visual_tabs = st.tabs(["Data", "Visualisasi"])

with visual_tabs[0]:
    st.subheader("Data Terfilter")
    st.dataframe(filtered_data.reset_index(drop=True))

with visual_tabs[1]:
    st.subheader("Grafik Visualisasi")
    if filtered_data.empty:
        st.warning("Tidak ada data untuk ditampilkan pada grafik. Silakan terapkan filter yang lebih luas.")
    else:
        st.write("##### Tren Realisasi TW per Pagu")
        line_data = filtered_data[
            ["pagu_miliar", "realisasi_tw1_persen", "realisasi_tw2_persen", "realisasi_tw3_persen"]
        ].sort_values("pagu_miliar")
        st.line_chart(line_data.set_index("pagu_miliar"))

        st.write("##### Scatter Plot: Skor IKPA vs Realisasi TW3")
        scatter = alt.Chart(filtered_data).mark_circle(size=80, opacity=0.7).encode(
            x="skor_ikpa",
            y="realisasi_tw3_persen",
            color="tipe_satker",
            tooltip=[
                "kode_satker",
                "nama_kementerian",
                "provinsi",
                "skor_ikpa",
                "realisasi_tw3_persen",
            ],
        ).interactive()
        st.altair_chart(scatter, use_container_width=True)

        st.write("##### Area Chart: Deviasi RPD & Realisasi TW3")
        area_data = filtered_data[["pagu_miliar", "deviasi_rpd_persen", "realisasi_tw3_persen"]].sort_values(
            "pagu_miliar"
        )
        st.area_chart(area_data.set_index("pagu_miliar"))

        st.write("##### Proporsi Realisasi Tercapai 95%")
        st.bar_chart(filtered_data["realisasi_tercapai_95persen"].value_counts())

st.markdown("---")
st.subheader("Prediksi Model")
if model is None:
    st.error(model_error)
else:
    st.write("Model `Best_model.pkcls` berhasil dimuat. Data akan diprediksi dengan model klasifikasi.")
    if filtered_data.empty:
        st.warning("Tidak ada data yang sesuai filter saat ini.")
    else:
        run_prediction = st.button("Prediksi data terfilter")
        if run_prediction:
            with st.spinner("Menjalankan prediksi..."):
                try:
                    prediksi = predict_orange(model, filtered_data)
                    result = filtered_data.reset_index(drop=True).copy()
                    result["prediksi_realisasi_95%"] = prediksi
                    st.dataframe(result)
                except Exception as exc:
                    st.error(f"Prediksi gagal: {exc}")
