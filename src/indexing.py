"""Indexing BM25Okapi offline dan penyimpanan artefak hasil indexing."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

from src.data_loader import Recipe, load_dataframe, prepare_recipes
from src.preprocess import PREPROCESS_VERSION, tokenize

# rank_bm25 tidak otomatis membangun inverted index terpisah.
DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def build_index(train_path: str | Path, artifacts_dir: str | Path) -> dict:
    from rank_bm25 import BM25Okapi

    train_path = Path(train_path)
    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)

    recipes, audit_stats, _ = prepare_recipes(load_dataframe(train_path), "train")
    if not recipes:
        raise ValueError("Tidak ada dokumen valid untuk diindeks.")
    tokenized_corpus = [tokenize(recipe.document) for recipe in recipes]
    model = BM25Okapi(tokenized_corpus, k1=DEFAULT_K1, b=DEFAULT_B)

    # File pickle HARUS hanya dimuat dari artefak lokal yang tepercaya.
    with (artifacts / "bm25.pkl").open("wb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    with (artifacts / "recipes.pkl").open("wb") as stream:
        pickle.dump(recipes, stream, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "method": "BM25Okapi",
        "implementation": "rank_bm25",
        "document_representation": "title + ingredients + steps",
        "preprocess_version": PREPROCESS_VERSION,
        "stopword_removal": False,
        "stemming": False,
        "field_weights": None,
        "parameters": {"k1": DEFAULT_K1, "b": DEFAULT_B},
        "document_count": len(recipes),
        "train_source": str(train_path.name),
        "audit_train": audit_stats,
        "top_k_default": 5,
    }
    (artifacts / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_index(artifacts_dir: str | Path):
    """Muat objek BM25 dan resep yang dibuat oleh build_index (lokal dan tepercaya)."""
    folder = Path(artifacts_dir)
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["preprocess_version"] != PREPROCESS_VERSION:
        raise ValueError("Versi preprocessing berubah. Bangun ulang index dengan src.indexing.")
    with (folder / "bm25.pkl").open("rb") as stream:
        model = pickle.load(stream)
    with (folder / "recipes.pkl").open("rb") as stream:
        recipes: list[Recipe] = pickle.load(stream)
    if len(recipes) != manifest["document_count"] or len(model.doc_len) != len(recipes):
        raise ValueError("Artefak index tidak konsisten. Jalankan indexing ulang.")
    return model, recipes, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Bangun index BM25 dari split train")
    parser.add_argument("--train", default="data/raw/train.parquet")
    parser.add_argument("--artifacts", default="artifacts")
    args = parser.parse_args()
    manifest = build_index(args.train, args.artifacts)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
