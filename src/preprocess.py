"""Prapemrosesan konsisten untuk corpus dan kueri (baseline tanpa stemming/stopwords).

Angka pecahan 1/2 dan desimal 1,5 atau 1.5 dipertahankan sebagai token utuh.
Semua istilah, termasuk kata umum dan satuan, tetap disimpan untuk BM25.
"""

from __future__ import annotations

import re
from typing import Iterable

# Cabang pertama menangkap pecahan/desimal. Cabang kedua menangkap kata Unicode.
TOKEN_PATTERN = re.compile(r"\d+(?:[.,/]\d+)*|[^\W\d_]+|\d+", re.UNICODE)
PREPROCESS_VERSION = "baseline-casefold-regex-v1"


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    return str(text).casefold().strip()


def tokenize(text: object) -> list[str]:
    """Case folding, pembersihan tanda baca, lalu tokenisasi.

    Tidak dilakukan stopword removal, stemming, deduplikasi token, maupun
    penghilangan angka agar frekuensi term sesuai formulasi BM25.
    """
    return TOKEN_PATTERN.findall(normalize_text(text))


def tokenize_parts(parts: Iterable[object]) -> list[str]:
    return tokenize("\n".join(str(part) for part in parts if part is not None))
