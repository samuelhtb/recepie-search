"""Retrieval BM25: skor positif, sort menurun, tie-break doc_id, Top-K."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from src.data_loader import Recipe
from src.query_processing import process_query


@dataclass(frozen=True)
class SearchHit:
    rank: int
    score: float
    recipe: Recipe


@dataclass(frozen=True)
class SearchResponse:
    query_code: str
    query_tokens: list[str]
    retrieval_time_ms: float
    hits: list[SearchHit]


class RecipeRetriever:
    def __init__(self, model: object, recipes: list[Recipe]):
        if not recipes:
            raise ValueError("Corpus kosong")
        self.model = model
        self.recipes = recipes
        self.doc_ids = np.asarray([recipe.doc_id for recipe in recipes])

    def search(
        self,
        title: str = "",
        ingredients: str = "",
        steps: str = "",
        top_k: int = 5,
    ) -> SearchResponse:
        if top_k < 1:
            raise ValueError("top_k harus sekurang-kurangnya 1")
        started = perf_counter()
        query = process_query(title, ingredients, steps)
        scores = np.asarray(self.model.get_scores(query.tokens), dtype=float)
        if scores.size != len(self.recipes):
            raise ValueError("Jumlah skor BM25 berbeda dari jumlah resep")
        candidates = np.flatnonzero(np.isfinite(scores) & (scores > 0))
        if not candidates.size:
            return SearchResponse(query.code, query.tokens, (perf_counter() - started) * 1000, [])

        # lexsort: key utama skor menurun; untuk skor sama ID dokumen menaik.
        order = np.lexsort((self.doc_ids[candidates], -scores[candidates]))
        selected = candidates[order[:top_k]]
        hits = [
            SearchHit(rank=i, score=float(scores[idx]), recipe=self.recipes[idx])
            for i, idx in enumerate(selected, start=1)
        ]
        return SearchResponse(query.code, query.tokens, (perf_counter() - started) * 1000, hits)
