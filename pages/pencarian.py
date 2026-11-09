"""Halaman tes pencarian resep dengan kombinasi input Q1–Q7."""

import streamlit as st

from src.query_processing import QUERY_LABELS
from src.ui_shared import get_retriever, show_missing_index

st.title("🔎 Tes Pencarian Resep")
st.write(
    "Cari resep masakan Indonesia berdasarkan **judul**, **bahan**, **langkah**, "
    "atau kombinasinya. Sistem menghitung skor BM25Okapi dari corpus *train* "
    "dan menampilkan maksimal lima resep terbaik."
)

try:
    retriever, manifest = get_retriever()
except (FileNotFoundError, ValueError, ModuleNotFoundError, OSError) as exc:
    show_missing_index(exc)
    st.stop()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Dokumen dalam corpus", f"{manifest['document_count']:,}")
col_b.metric("Metode", "BM25Okapi")
col_c.metric("Jumlah hasil", "Top-5")

with st.form("recipe_search"):
    title = st.text_input("Judul / nama masakan", placeholder="Contoh: ayam goreng bawang putih")
    ingredients = st.text_area(
        "Bahan masakan", placeholder="Contoh: ayam, bawang putih, garam", height=110
    )
    steps = st.text_area(
        "Langkah memasak", placeholder="Contoh: goreng ayam hingga matang", height=110
    )
    submitted = st.form_submit_button("Cari Resep", type="primary", use_container_width=True)

if submitted:
    try:
        st.session_state["search_result"] = retriever.search(
            title=title, ingredients=ingredients, steps=steps, top_k=5
        )
    except ValueError as exc:
        st.session_state.pop("search_result", None)
        st.warning(str(exc))

response = st.session_state.get("search_result")
if response is not None:
    st.subheader("Hasil Pencarian")
    st.caption(
        f"{response.query_code} — {QUERY_LABELS[response.query_code]} | "
        f"Retrieval time: {response.retrieval_time_ms:.2f} ms"
    )
    if not response.hits:
        st.info("Tidak ditemukan resep dengan skor BM25 positif untuk query ini.")
    for hit in response.hits:
        with st.expander(
            f"#{hit.rank} · {hit.recipe.title} · BM25 {hit.score:.4f}",
            expanded=(hit.rank == 1),
        ):
            st.caption(f"ID dokumen: {hit.recipe.doc_id}")
            st.markdown("**Bahan:**")
            for ingredient in hit.recipe.ingredients:
                st.write(f"• {ingredient}")
            st.markdown("**Langkah memasak:**")
            for step_number, step in enumerate(hit.recipe.steps, start=1):
                st.write(f"{step_number}. {step}")
    st.info(
        "**Cara membaca skor:** skor BM25 adalah nilai pemeringkatan kecocokan istilah, "
        "bukan persentase relevansi atau kualitas resep. Halaman Evaluasi & Grafik "
        "mengukur relevansi melalui penilaian manusia."
    )
