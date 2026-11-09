"""Pembacaan, audit, dan pembersihan resep dari berkas Parquet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.preprocess import tokenize

REQUIRED_COLUMNS = ("title", "ingredients", "steps")
OPTIONAL_COLUMNS = ("num_ingredients", "num_steps", "char_count")


@dataclass
class Recipe:
    doc_id: str
    source_row: int
    title: str
    ingredients: list[str]
    steps: list[str]
    document: str


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple)):
        return False
    # Hindari kondisi ambigu pada array dari NumPy.
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return False
    try:
        flag = pd.isna(value)
        if flag is pd.NA:
            return True
        if hasattr(flag, "shape") and getattr(flag, "shape", ()) != ():
            return False
        return bool(flag)
    except (TypeError, ValueError):
        return False


def list_of_text(value: Any) -> list[str]:
    """Ubah kolom bertipe list (dari Parquet) menjadi daftar teks bersih."""
    if _is_null(value):
        return []
    if isinstance(value, str):
        value = [value]
    elif hasattr(value, "tolist") and not isinstance(value, (list, tuple)):
        value = value.tolist()
    elif not isinstance(value, (list, tuple)):
        value = [value]
    output: list[str] = []
    for item in value:
        if _is_null(item):
            continue
        text = str(item).strip()
        if text:
            output.append(text)
    return output


def _clean_str(value: Any) -> str:
    return "" if _is_null(value) else str(value).strip()


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    try:
        data = pd.read_parquet(path, engine="pyarrow")
    except ImportError as exc:
        raise RuntimeError("PyArrow belum tersedia. Jalankan: pip install -r requirements.txt") from exc
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Kolom wajib hilang dari {path.name}: {sorted(missing)}")
    return data


def fingerprint(title: str, ingredients: list[str], steps: list[str]) -> str:
    """Identitas konten utuh; judul sama saja TIDAK berarti duplikat."""
    def canonical(text: str) -> str:
        return re.sub(r"\s+", " ", text.casefold()).strip()

    serialized = json.dumps(
        [canonical(title), [canonical(x) for x in ingredients], [canonical(x) for x in steps]],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def prepare_recipes(
    frame: pd.DataFrame, split: str = "train"
) -> tuple[list[Recipe], dict[str, Any], set[str]]:
    """Buang dokumen tak layak (judul kosong / tak ada isi / tidak bertoken) dan duplikat identik.

    Doc ID mempertahankan nomor baris sumber agar stabil setelah pembersihan.
    """
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Kolom wajib tidak tersedia: {sorted(missing)}")

    recipes: list[Recipe] = []
    seen_hashes: set[str] = set()
    titles: Counter[str] = Counter()
    stats: dict[str, Any] = {
        "split": split,
        "raw_rows": int(len(frame)),
        "columns": {name: str(dtype) for name, dtype in frame.dtypes.items()},
        "null_counts": {name: int(count) for name, count in frame.isna().sum().items()},
        "empty_title": 0,
        "empty_ingredients": 0,
        "empty_steps": 0,
        "invalid_dropped": 0,
        "duplicate_full_dropped": 0,
        "num_ingredients_mismatch": 0,
        "num_steps_mismatch": 0,
    }
    has_num_ing = "num_ingredients" in frame
    has_num_steps = "num_steps" in frame

    # itertuples menghindari overhead Series pada semua 66 ribu dokumen.
    col_names = list(frame.columns)
    pos = {name: col_names.index(name) for name in col_names}
    for row_no, values in enumerate(frame.itertuples(index=False, name=None)):
        title = _clean_str(values[pos["title"]])
        ingredients = list_of_text(values[pos["ingredients"]])
        steps = list_of_text(values[pos["steps"]])
        if not title:
            stats["empty_title"] += 1
        if not ingredients:
            stats["empty_ingredients"] += 1
        if not steps:
            stats["empty_steps"] += 1
        if has_num_ing:
            count = values[pos["num_ingredients"]]
            if not _is_null(count) and int(count) != len(ingredients):
                stats["num_ingredients_mismatch"] += 1
        if has_num_steps:
            count = values[pos["num_steps"]]
            if not _is_null(count) and int(count) != len(steps):
                stats["num_steps_mismatch"] += 1

        # Aturan eksplisit: judul wajib; sedikitnya salah satu daftar memuat isi.
        if not title or (not ingredients and not steps):
            stats["invalid_dropped"] += 1
            continue
        document = "\n".join([title, " ".join(ingredients), " ".join(steps)])
        if not tokenize(document):
            stats["invalid_dropped"] += 1
            continue
        signature = fingerprint(title, ingredients, steps)
        if signature in seen_hashes:
            stats["duplicate_full_dropped"] += 1
            continue
        seen_hashes.add(signature)
        titles[title.casefold()] += 1
        recipes.append(
            Recipe(
                doc_id=f"{split}-{row_no:06d}",
                source_row=row_no,
                title=title,
                ingredients=ingredients,
                steps=steps,
                document=document,
            )
        )

    stats["valid_unique_rows"] = len(recipes)
    stats["duplicate_title_groups_kept"] = sum(count > 1 for count in titles.values())
    stats["rows_in_duplicate_title_groups_kept"] = sum(
        count for count in titles.values() if count > 1
    )
    return recipes, stats, seen_hashes


def audit(train_path: str | Path, eval_path: str | Path, output: str | Path) -> dict[str, Any]:
    train, train_stats, train_hashes = prepare_recipes(load_dataframe(train_path), "train")
    _, eval_stats, eval_hashes = prepare_recipes(load_dataframe(eval_path), "eval")
    result = {
        "train": train_stats,
        "eval": eval_stats,
        "cross_split_exact_duplicates": len(train_hashes & eval_hashes),
        "corpus_documents_after_cleaning": len(train),
        "note": "Duplikat lintas split hanya diperiksa dan dilaporkan; eval tidak diindeks.",
    }
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit data resep train dan eval")
    parser.add_argument("--train", default="data/raw/train.parquet")
    parser.add_argument("--eval", default="data/raw/eval.parquet")
    parser.add_argument("--out", default="data/processed/dataset_audit.json")
    args = parser.parse_args()
    print(json.dumps(audit(args.train, args.eval, args.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
