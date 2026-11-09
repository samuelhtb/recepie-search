"""Utilitas UI Streamlit; kode retrieval tetap berada di modul terpisah."""

from pathlib import Path

import streamlit as st

from src.indexing import load_index
from src.retrieval import RecipeRetriever

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"
TRAIN = ROOT / "data" / "raw" / "train.parquet"
EVAL = ROOT / "data" / "raw" / "eval.parquet"


@st.cache_resource(show_spinner="Memuat indeks BM25 dari disk...")
def cached_retriever(folder: str, manifest_mtime_ns: int) -> tuple[RecipeRetriever, dict]:
    """Cache per versi manifest. Objek pickle HANYA dari artefak lokal yang dipercaya."""
    model, recipes, manifest = load_index(folder)
    return RecipeRetriever(model, recipes), manifest


def get_retriever() -> tuple[RecipeRetriever, dict]:
    manifest = ARTIFACTS / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Indeks BM25 belum dibuat. Jalankan: python -m src.indexing")
    return cached_retriever(str(ARTIFACTS), manifest.stat().st_mtime_ns)


def show_missing_index(exc: Exception) -> None:
    st.error("Indeks BM25 belum tersedia atau gagal dimuat.")
    st.code("python -m src.indexing", language="bash")
    st.caption(f"Detail: {exc}")
