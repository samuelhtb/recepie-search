"""Fungsi data untuk dashboard evaluasi Streamlit (tanpa dependensi UI).

Dua penilai mengisi grade 0/1/2 tanpa auto-label. Final hanya untuk perbedaan.
Lembar anotasi selalu dibaca lagi sebelum disimpan dan hanya kolom yang diedit
oleh penilai bersangkutan yang diubah.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

RATING_FIELDS = ("rater_1", "rater_2", "final_relevance")
REQUIRED_FILES = ("queries.csv", "retrieval_pool.csv", "relevance_judgments.csv")
SCORE_FILES = ("metrics_by_query.csv", "metrics_by_combination.csv", "evaluation_summary.json")


def read_judgments(path: str | Path) -> pd.DataFrame:
    """Baca nilai sebagai string agar kolom anotasi kosong tetap string, bukan NaN."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    expected = {"query_id", "doc_id", *RATING_FIELDS}
    if not expected.issubset(frame.columns):
        raise ValueError(f"Kolom judgment tidak lengkap: {sorted(expected - set(frame.columns))}")
    if frame.duplicated(["query_id", "doc_id"]).any():
        raise ValueError("Pasangan query_id–doc_id pada anotasi harus unik.")
    for role in RATING_FIELDS:
        frame[role] = frame[role].map(normalize_grade)
    return frame


def normalize_grade(value: object) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    try:
        grade = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Nilai relevansi tidak valid: {value!r}; gunakan 0, 1, atau 2") from exc
    if not grade.is_integer() or int(grade) not in (0, 1, 2):
        raise ValueError(f"Nilai relevansi tidak valid: {value!r}; gunakan 0, 1, atau 2")
    return str(int(grade))


def annotation_progress(judgments: pd.DataFrame) -> dict[str, int | bool]:
    """Status anotasi tanpa menyimpulkan relevansi otomatis dari skor BM25."""
    rows = len(judgments)
    a = judgments["rater_1"].map(normalize_grade)
    b = judgments["rater_2"].map(normalize_grade)
    final = judgments["final_relevance"].map(normalize_grade)
    both = (a != "") & (b != "")
    conflict = both & (a != b)
    # Jika nilai final pada penilai yang sepakat berbeda, status tidak valid.
    invalid_final = both & (a == b) & (final != "") & (final != a)
    ready = both & (~conflict | (final != "")) & ~invalid_final
    return {
        "total": rows,
        "rater_1_done": int((a != "").sum()),
        "rater_2_done": int((b != "").sum()),
        "both_done": int(both.sum()),
        "conflicts": int(conflict.sum()),
        "conflicts_unresolved": int((conflict & (final == "")).sum()),
        "invalid_final": int(invalid_final.sum()),
        "ready_count": int(ready.sum()),
        "complete": bool(rows > 0 and int(ready.sum()) == rows),
    }


def prepare_exists(folder: str | Path) -> bool:
    path = Path(folder)
    return all((path / file).is_file() for file in REQUIRED_FILES)


def score_is_current(folder: str | Path) -> bool:
    """Hasil tidak boleh dipresentasikan sebagai valid jika anotasi/query berubah."""
    folder = Path(folder)
    sources = [folder / file for file in REQUIRED_FILES]
    results = [folder / file for file in SCORE_FILES]
    if not all(path.is_file() for path in [*sources, *results]):
        return False
    # mtime file hasil paling lama harus tidak lebih awal dari sumber terbaru.
    return min(path.stat().st_mtime_ns for path in results) >= max(
        path.stat().st_mtime_ns for path in sources
    )


def save_annotation(
    path: str | Path, query_id: str, role: str, edited: pd.DataFrame
) -> int:
    """Simpan penilaian satu query dan satu peran; jaga semua ID/kolom lain.

    edited harus memiliki doc_id dan kolom role. Hanya pasangan ID yang sudah
    ada pada query terkait boleh diperbarui. Penilaian final hanya diizinkan
    ketika dua penilai telah berbeda dan sama-sama terisi.
    """
    if role not in RATING_FIELDS:
        raise ValueError("Peran penilai tidak dikenal")
    if not {"doc_id", role}.issubset(edited.columns):
        raise ValueError("Data editor harus berisi doc_id dan kolom peran penilai")
    if edited["doc_id"].duplicated().any():
        raise ValueError("Dokumen ganda pada hasil editor")
    raw = read_judgments(path)
    selected = raw[raw["query_id"] == query_id]
    if selected.empty:
        raise ValueError("Query tidak ditemukan pada lembar penilaian")
    if not set(edited["doc_id"]).issubset(set(selected["doc_id"])):
        raise ValueError("Hasil editor mengandung dokumen yang bukan milik query")
    if role != "final_relevance" and set(edited["doc_id"]) != set(selected["doc_id"]):
        raise ValueError("Seluruh dokumen query harus tampil pada editor penilai")

    values = {str(row.doc_id): normalize_grade(getattr(row, role)) for row in edited.itertuples(index=False)}
    changes = 0
    for idx in selected.index:
        doc_id = str(raw.at[idx, "doc_id"])
        if doc_id not in values:
            continue
        new_value = values[doc_id]
        old_value = str(raw.at[idx, role])
        if role == "final_relevance" and new_value:
            a, b = raw.at[idx, "rater_1"], raw.at[idx, "rater_2"]
            if not a or not b or a == b:
                raise ValueError("Keputusan akhir hanya boleh untuk nilai dua penilai yang berbeda")
        if new_value != old_value:
            raw.at[idx, role] = new_value
            changes += 1
            # Mencegah nilai final lama dipakai setelah label penilai direvisi.
            if role in ("rater_1", "rater_2"):
                raw.at[idx, "final_relevance"] = ""

    if changes == 0:
        return 0

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", prefix=".annotations-", dir=path.parent,
            encoding="utf-8-sig", newline="", delete=False,
        ) as target:
            temporary = target.name
            raw.to_csv(target, index=False)
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return changes
