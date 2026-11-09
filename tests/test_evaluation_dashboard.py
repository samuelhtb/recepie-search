"""Test logika anotasi di dashboard (tanpa butuh Streamlit)."""

import pandas as pd
import pytest

from src.evaluation_dashboard import (
    annotation_progress,
    prepare_exists,
    read_judgments,
    save_annotation,
    score_is_current,
)
from src.evaluate import reconcile_judgments


def test_annotation_progress_and_decision(tmp_path):
    path = tmp_path / "relevance_judgments.csv"
    pd.DataFrame([
        {"query_id": "q", "doc_id": "d1", "rank": 1, "recipe_title": "A", "rater_1": "2", "rater_2": "2", "final_relevance": ""},
        {"query_id": "q", "doc_id": "d2", "rank": 2, "recipe_title": "B", "rater_1": "1", "rater_2": "0", "final_relevance": ""},
        {"query_id": "q", "doc_id": "d3", "rank": 3, "recipe_title": "C", "rater_1": "", "rater_2": "0", "final_relevance": ""},
    ]).to_csv(path, index=False)
    frame = read_judgments(path)
    first = annotation_progress(frame)
    assert first["total"] == 3
    assert first["ready_count"] == 1
    assert first["conflicts_unresolved"] == 1
    assert first["complete"] is False

    with pytest.raises(ValueError, match="Seluruh dokumen"):
        save_annotation(path, "q", "rater_1", pd.DataFrame([{"doc_id": "d3", "rater_1": "1"}]))

    changed = frame[["doc_id", "rater_1"]].copy()
    changed.loc[changed["doc_id"] == "d3", "rater_1"] = "0"
    assert save_annotation(path, "q", "rater_1", changed) == 1
    assert save_annotation(path, "q", "final_relevance", pd.DataFrame([{"doc_id": "d2", "final_relevance": "1"}])) == 1
    latest = read_judgments(path)
    assert annotation_progress(latest)["complete"]
    assert reconcile_judgments(latest)[("q", "d2")] == 1


def test_revising_rater_resets_old_final(tmp_path):
    path = tmp_path / "relevance_judgments.csv"
    pd.DataFrame([
        {"query_id": "q", "doc_id": "d", "rater_1": "0", "rater_2": "2", "final_relevance": "1"},
    ]).to_csv(path, index=False)
    edited = pd.DataFrame([{"doc_id": "d", "rater_2": "0"}])
    assert save_annotation(path, "q", "rater_2", edited) == 1
    latest = read_judgments(path)
    assert latest.iloc[0].final_relevance == ""
    assert latest.iloc[0].rater_1 == "0"
    assert annotation_progress(latest)["complete"]


def test_invalid_labels_and_untrusted_doc_id_rejected(tmp_path):
    path = tmp_path / "relevance_judgments.csv"
    pd.DataFrame([{"query_id": "q", "doc_id": "d", "rater_1": "", "rater_2": "", "final_relevance": ""}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="tidak valid"):
        save_annotation(path, "q", "rater_1", pd.DataFrame([{"doc_id": "d", "rater_1": "3"}]))
    with pytest.raises(ValueError, match="bukan milik query"):
        save_annotation(path, "q", "rater_1", pd.DataFrame([{"doc_id": "other", "rater_1": "1"}]))
    with pytest.raises(ValueError, match="hanya boleh"):
        save_annotation(path, "q", "final_relevance", pd.DataFrame([{"doc_id": "d", "final_relevance": "2"}]))


def test_preparation_and_score_freshness(tmp_path):
    for name in ("queries.csv", "retrieval_pool.csv", "relevance_judgments.csv"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert prepare_exists(tmp_path)
    assert not score_is_current(tmp_path)
    for name in ("metrics_by_query.csv", "metrics_by_combination.csv", "evaluation_summary.json"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    assert score_is_current(tmp_path)
    import os
    from time import time_ns
    path = tmp_path / "relevance_judgments.csv"
    future = time_ns() + 2_000_000_000
    os.utime(path, ns=(future, future))
    assert not score_is_current(tmp_path)
