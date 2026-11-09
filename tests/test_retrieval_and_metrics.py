from dataclasses import dataclass

import pytest

from src.data_loader import Recipe
from src.evaluate import calculate_metrics, reconcile_judgments, select_base_recipes
from src.retrieval import RecipeRetriever


class FakeBM25:
    def __init__(self, scores):
        self.scores = scores

    def get_scores(self, tokens):
        return self.scores


def recipes_with_ids(ids):
    return [Recipe(i, j, "Judul", ["ayam"], ["goreng"], "Judul ayam goreng") for j, i in enumerate(ids)]


def test_rank_ties_are_broken_by_document_id():
    searcher = RecipeRetriever(FakeBM25([2.0, 3.0, 3.0, 0.0]), recipes_with_ids(["d4", "d3", "d2", "d1"]))
    hits = searcher.search("ayam", top_k=5).hits
    assert [hit.recipe.doc_id for hit in hits] == ["d2", "d3", "d4"]
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_no_positive_score_is_empty():
    searcher = RecipeRetriever(FakeBM25([0, -1]), recipes_with_ids(["d1", "d2"]))
    assert searcher.search("ayam").hits == []


def test_metric_formulas_and_fixed_denominator():
    m = calculate_metrics([0, 2, 1], [0, 2, 1, 2, 0], k=5)
    assert m["precision_at_5"] == pytest.approx(0.4)
    assert m["reciprocal_rank_at_5"] == pytest.approx(0.5)
    assert 0 < m["ndcg_at_5"] <= 1
    assert calculate_metrics([], [], k=5) == {
        "precision_at_5": 0.0, "reciprocal_rank_at_5": 0.0, "ndcg_at_5": 0.0
    }


def test_judgments_require_two_raters_and_resolution():
    import pandas as pd

    frame = pd.DataFrame([{"query_id": "q", "doc_id": "d", "rater_1": 2, "rater_2": 1, "final_relevance": ""}])
    with pytest.raises(ValueError, match="Anotasi belum lengkap"):
        reconcile_judgments(frame)
    frame["final_relevance"] = 2
    assert reconcile_judgments(frame) == {("q", "d"): 2}


def test_end_to_end_metric_export_with_manual_labels(tmp_path):
    import pandas as pd
    from src.evaluate import score_evaluation

    pd.DataFrame([
        {"query_id": "eval-000001-Q1", "base_recipe_id": "eval-000001", "query_code": "Q1", "query_type": "Judul", "retrieval_time_ms": 12.0},
        {"query_id": "eval-000001-Q2", "base_recipe_id": "eval-000001", "query_code": "Q2", "query_type": "Bahan", "retrieval_time_ms": 15.0},
    ]).to_csv(tmp_path / "queries.csv", index=False)
    pd.DataFrame([
        {"query_id": "eval-000001-Q1", "doc_id": "train-000001", "rank": 1},
        {"query_id": "eval-000001-Q1", "doc_id": "train-000002", "rank": 2},
        {"query_id": "eval-000001-Q2", "doc_id": "train-000001", "rank": 1},
    ]).to_csv(tmp_path / "pool.csv", index=False)
    pd.DataFrame([
        {"query_id": "eval-000001-Q1", "doc_id": "train-000001", "rater_1": 0, "rater_2": 0, "final_relevance": ""},
        {"query_id": "eval-000001-Q1", "doc_id": "train-000002", "rater_1": 2, "rater_2": 2, "final_relevance": ""},
        {"query_id": "eval-000001-Q2", "doc_id": "train-000001", "rater_1": 1, "rater_2": 2, "final_relevance": 2},
    ]).to_csv(tmp_path / "judgments.csv", index=False)
    summary = score_evaluation(
        tmp_path / "queries.csv", tmp_path / "pool.csv", tmp_path / "judgments.csv", tmp_path
    )
    assert summary["queries_evaluated"] == 2
    assert summary["mean_precision_at_5"] == pytest.approx(0.2)
    assert summary["mrr_at_5"] == pytest.approx(0.75)
    assert summary["mean_retrieval_time_ms"] == pytest.approx(13.5)
    assert (tmp_path / "metrics_by_combination.csv").exists()
