"""Pintu masuk Streamlit: navigasi halaman pencarian dan evaluasi BM25."""

import streamlit as st

st.set_page_config(
    page_title="Pencarian Resep Indonesia | BM25",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.navigation(
    [
        st.Page("pages/pencarian.py", title="Tes Pencarian Resep", icon="🔎", default=True),
        st.Page("pages/evaluasi.py", title="Evaluasi & Grafik", icon="📊"),
    ],
    position="sidebar",
).run()
