"""Tujuh kombinasi masukan (Q1–Q7) sesuai laporan penelitian."""

from __future__ import annotations

from dataclasses import dataclass

from src.preprocess import tokenize

QUERY_CODES: dict[tuple[bool, bool, bool], str] = {
    (True, False, False): "Q1",  # Judul
    (False, True, False): "Q2",  # Bahan
    (False, False, True): "Q3",  # Langkah
    (True, True, False): "Q4",  # Judul+Bahan
    (True, False, True): "Q5",  # Judul+Langkah
    (False, True, True): "Q6",  # Bahan+Langkah
    (True, True, True): "Q7",  # Ketiganya
}
QUERY_LABELS = {
    "Q1": "Judul",
    "Q2": "Bahan",
    "Q3": "Langkah",
    "Q4": "Judul + Bahan",
    "Q5": "Judul + Langkah",
    "Q6": "Bahan + Langkah",
    "Q7": "Judul + Bahan + Langkah",
}


@dataclass(frozen=True)
class ParsedQuery:
    code: str
    combined_text: str
    tokens: list[str]


def process_query(title: str = "", ingredients: str = "", steps: str = "") -> ParsedQuery:
    title, ingredients, steps = title.strip(), ingredients.strip(), steps.strip()
    active = (bool(title), bool(ingredients), bool(steps))
    if active not in QUERY_CODES:
        raise ValueError("Isi minimal satu kolom: judul, bahan, atau langkah memasak.")
    text = "\n".join(value for value in (title, ingredients, steps) if value)
    tokens = tokenize(text)
    if not tokens:
        raise ValueError("Masukan tidak menghasilkan token yang dapat dicari.")
    return ParsedQuery(code=QUERY_CODES[active], combined_text=text, tokens=tokens)
