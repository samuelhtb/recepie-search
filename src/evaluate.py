"""Evaluasi manual untuk tujuh kombinasi query (Q1–Q7).

Tahap 1: `prepare` menghasilkan ~12x7 query, pool top-10, dan lembar anotasi.
Tahap 2: dua penilai memberi nilai 0/1/2; perbedaan diputuskan manual.
Tahap 3: `score` menghitung P@5, MRR@5, nDCG@5 (pool dinilai), dan waktu.

Tidak ada pseudo-label dari eval.parquet: evaluasi otomatis tanpa anotasi tidak sah.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_loader import Recipe, fingerprint, load_dataframe, prepare_recipes
from src.indexing import load_index
from src.query_processing import QUERY_LABELS
from src.retrieval import RecipeRetriever

VARIANTS = {
    "Q1": (True, False, False),
    "Q2": (False, True, False),
    "Q3": (False, False, True),
    "Q4": (True, True, False),
    "Q5": (True, False, True),
    "Q6": (False, True, True),
    "Q7": (True, True, True),
}


def select_base_recipes(
    eval_recipes: list[Recipe], train_hashes: set[str], n: int = 12, seed: int = 42
) -> list[Recipe]:
    """Pemilihan awal secara deterministik, beragam panjang bahan/langkah.

    Ini hanya calon query; tim harus meninjau representativitasnya sebelum
    pengujian akhir. Dokumen eval identik dengan train dikeluarkan.
    """
    if n <= 0:
        raise ValueError("Jumlah resep dasar harus > 0")
    available = []
    seen_title = set()
    for recipe in eval_recipes:
        signature = fingerprint(recipe.title, recipe.ingredients, recipe.steps)
        # Ketujuh varian query harus menghasilkan teks; semua field dasar wajib berisi.
        if (signature in train_hashes or recipe.title.casefold() in seen_title
                or not recipe.ingredients or not recipe.steps):
            continue
        available.append(recipe)
        seen_title.add(recipe.title.casefold())
    if len(available) < n:
        raise ValueError(f"Hanya ada {len(available)} resep eval nonidentik dengan judul unik, perlu {n}.")

    rng = np.random.default_rng(seed)
    lengths_ing = np.asarray([len(x.ingredients) for x in available])
    lengths_steps = np.asarray([len(x.steps) for x in available])
    ing_cutoffs = np.quantile(lengths_ing, [1 / 3, 2 / 3])
    steps_cutoffs = np.quantile(lengths_steps, [1 / 3, 2 / 3])
    groups: dict[tuple[int, int], list[Recipe]] = {(i, j): [] for i in range(3) for j in range(3)}
    for recipe in available:
        i = int(np.searchsorted(ing_cutoffs, len(recipe.ingredients), side="right"))
        j = int(np.searchsorted(steps_cutoffs, len(recipe.steps), side="right"))
        groups[(i, j)].append(recipe)
    # Acak urutan tiap kelompok, lalu ambil secara round-robin.
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < n:
        made_progress = False
        for key in sorted(groups):
            if len(selected) >= n:
                break
            if groups[key]:
                selected.append(groups[key].pop())
                made_progress = True
        if not made_progress:
            break
    return selected


def make_query_rows(recipes: list[Recipe]) -> list[dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    for recipe in recipes:
        values = (recipe.title, "\n".join(recipe.ingredients), "\n".join(recipe.steps))
        for code, flags in VARIANTS.items():
            title, ingredients, steps = (
                value if enabled else "" for value, enabled in zip(values, flags)
            )
            queries.append(
                {
                    "query_id": f"{recipe.doc_id}-{code}",
                    "base_recipe_id": recipe.doc_id,
                    "query_code": code,
                    "query_type": QUERY_LABELS[code],
                    "title": title,
                    "ingredients": ingredients,
                    "steps": steps,
                }
            )
    return queries


def create_evaluation_materials(
    eval_path: str | Path,
    train_path: str | Path,
    artifacts_dir: str | Path,
    output_dir: str | Path,
    n_base: int = 12,
    pool_k: int = 10,
    seed: int = 42,
) -> dict[str, Any]:
    if pool_k < 5:
        raise ValueError("Judgment pool harus menilai minimal top-5; disarankan top-10.")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    # Seluruh identitas train dipakai hanya untuk mendeteksi kebocoran resep identik.
    _, _, train_hashes = prepare_recipes(load_dataframe(train_path), split="train")
    eval_recipes, eval_audit, _ = prepare_recipes(load_dataframe(eval_path), split="eval")
    base = select_base_recipes(eval_recipes, train_hashes, n=n_base, seed=seed)
    query_rows = make_query_rows(base)

    model, train_recipes, manifest = load_index(artifacts_dir)
    retriever = RecipeRetriever(model, train_recipes)
    pool_rows: list[dict] = []
    judgment_rows: list[dict] = []
    for item in query_rows:
        found = retriever.search(item["title"], item["ingredients"], item["steps"], top_k=pool_k)
        if found.query_code != item["query_code"]:
            raise AssertionError(f"Kode query tidak sesuai: {item['query_id']}")
        item["retrieval_time_ms"] = round(found.retrieval_time_ms, 5)
        for hit in found.hits:
            result = {
                "query_id": item["query_id"],
                "query_code": item["query_code"],
                "doc_id": hit.recipe.doc_id,
                "rank": hit.rank,
                "bm25_score": round(hit.score, 8),
                "recipe_title": hit.recipe.title,
                "recipe_ingredients": "\n".join(hit.recipe.ingredients),
                "recipe_steps": "\n".join(hit.recipe.steps),
            }
            pool_rows.append(result)
            judgment_rows.append(
                {
                    **result,
                    "query_title": item["title"],
                    "query_ingredients": item["ingredients"],
                    "query_steps": item["steps"],
                    "rater_1": "",
                    "rater_2": "",
                    "final_relevance": "",
                }
            )

    pool_columns = ["query_id", "query_code", "doc_id", "rank", "bm25_score", "recipe_title", "recipe_ingredients", "recipe_steps"]
    judgment_columns = pool_columns + ["query_title", "query_ingredients", "query_steps", "rater_1", "rater_2", "final_relevance"]
    pd.DataFrame(query_rows).to_csv(output / "queries.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(pool_rows, columns=pool_columns).to_csv(output / "retrieval_pool.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(judgment_rows, columns=judgment_columns).to_csv(output / "relevance_judgments.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "base_recipe_count": len(base),
        "query_count": len(query_rows),
        "pool_k": pool_k,
        "random_seed": seed,
        "candidate_count": len(pool_rows),
        "train_index_document_count": manifest["document_count"],
        "eval_audit": eval_audit,
        "cross_split_exact_recipe_filter": True,
        "warning": "Label 0,1,2 harus diberikan oleh minimal 2 penilai; jangan gunakan hasil sebagai metrik sebelum anotasi selesai.",
    }
    (output / "evaluation_setup.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metadata


def _validate_relevance(value: Any, label: str) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    try:
        n = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: nilai harus 0, 1, atau 2; ditemukan {value!r}") from exc
    if not n.is_integer() or int(n) not in (0, 1, 2):
        raise ValueError(f"{label}: nilai harus 0, 1, atau 2; ditemukan {value!r}")
    return int(n)


def reconcile_judgments(judgments: pd.DataFrame) -> dict[tuple[str, str], int]:
    required = {"query_id", "doc_id", "rater_1", "rater_2", "final_relevance"}
    if not required.issubset(judgments.columns):
        raise ValueError(f"Kolom penilaian wajib: {sorted(required)}")
    if judgments.duplicated(subset=["query_id", "doc_id"]).any():
        raise ValueError("Terdapat pasangan query–dokumen ganda dalam lembar anotasi")
    labels = {}
    incomplete = []
    for row in judgments.itertuples(index=False):
        qid, docid = str(row.query_id), str(row.doc_id)
        a = _validate_relevance(row.rater_1, f"{qid}/{docid} rater_1")
        b = _validate_relevance(row.rater_2, f"{qid}/{docid} rater_2")
        final = _validate_relevance(row.final_relevance, f"{qid}/{docid} final_relevance")
        if a is None or b is None or (a != b and final is None):
            incomplete.append((qid, docid))
            continue
        if final is not None and a == b and final != a:
            raise ValueError(f"Keputusan akhir {qid}/{docid} tidak sesuai dua penilai yang sepakat.")
        labels[(qid, docid)] = final if final is not None else a
    if incomplete:
        raise ValueError(
            f"Anotasi belum lengkap untuk {len(incomplete)} pasangan. Contoh: {incomplete[:3]}. "
            "Dua penilai wajib mengisi; saat berbeda nilai final_relevance harus diisi setelah diskusi."
        )
    return labels


def dcg_at_k(grades: list[int], k: int = 5) -> float:
    return sum((2 ** int(grade) - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades[:k], 1))


def calculate_metrics(top_grades: list[int], judged_pool_grades: list[int], k: int = 5) -> dict[str, float]:
    """P@k tetap berpenyebut k, MRR@k bounded, IDCG dari pool yang dinilai."""
    if any(grade not in (0, 1, 2) for grade in [*top_grades, *judged_pool_grades]):
        raise ValueError("Relevansi harus 0, 1, atau 2")
    top = top_grades[:k]
    precision = sum(grade >= 1 for grade in top) / k
    first_relevant_rank = next((i for i, grade in enumerate(top, 1) if grade >= 1), None)
    reciprocal_rank = (1 / first_relevant_rank) if first_relevant_rank else 0.0
    ideal = dcg_at_k(sorted(judged_pool_grades, reverse=True), k)
    ndcg = dcg_at_k(top, k) / ideal if ideal > 0 else 0.0
    return {"precision_at_5": float(precision), "reciprocal_rank_at_5": float(reciprocal_rank), "ndcg_at_5": float(ndcg)}


def score_evaluation(
    queries_csv: str | Path,
    pool_csv: str | Path,
    judgments_csv: str | Path,
    out_dir: str | Path,
) -> dict:
    queries = pd.read_csv(queries_csv, keep_default_na=False)
    pool = pd.read_csv(pool_csv, keep_default_na=False)
    judgments = pd.read_csv(judgments_csv, keep_default_na=False)
    labels = reconcile_judgments(judgments)
    expected_pairs = set(zip(pool["query_id"], pool["doc_id"]))
    if expected_pairs != set(labels):
        raise ValueError("ID query–dokumen di judgment harus tepat sama dengan retrieval_pool.csv")
    if queries["query_id"].duplicated().any():
        raise ValueError("query_id ganda dalam queries.csv")

    per_query = []
    for query in queries.itertuples(index=False):
        candidates = pool[pool["query_id"] == query.query_id].sort_values("rank")
        top_grades = [labels[(query.query_id, doc_id)] for doc_id in candidates.head(5)["doc_id"]]
        pool_grades = [labels[(query.query_id, doc_id)] for doc_id in candidates["doc_id"]]
        per_query.append(
            {
                "query_id": query.query_id,
                "base_recipe_id": query.base_recipe_id,
                "query_code": query.query_code,
                "query_type": query.query_type,
                "retrieved_count": len(candidates),
                "retrieval_time_ms": float(query.retrieval_time_ms),
                **calculate_metrics(top_grades, pool_grades, k=5),
            }
        )
    metrics = pd.DataFrame(per_query)
    grouped = (
        metrics.groupby(["query_code", "query_type"], sort=True)
        .agg(
            total_queries=("query_id", "size"),
            mean_precision_at_5=("precision_at_5", "mean"),
            mrr_at_5=("reciprocal_rank_at_5", "mean"),
            mean_ndcg_at_5=("ndcg_at_5", "mean"),
            mean_retrieval_time_ms=("retrieval_time_ms", "mean"),
            median_retrieval_time_ms=("retrieval_time_ms", "median"),
        )
        .reset_index()
    )
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output / "metrics_by_query.csv", index=False, encoding="utf-8-sig")
    grouped.to_csv(output / "metrics_by_combination.csv", index=False, encoding="utf-8-sig")
    summary = {
        "queries_evaluated": len(metrics),
        "judged_candidates": len(labels),
        "judgment_pool_limit": int(pool.groupby("query_id").size().max()) if len(pool) else 0,
        "mean_precision_at_5": float(metrics["precision_at_5"].mean()),
        "mrr_at_5": float(metrics["reciprocal_rank_at_5"].mean()),
        "mean_ndcg_at_5": float(metrics["ndcg_at_5"].mean()),
        "mean_retrieval_time_ms": float(metrics["retrieval_time_ms"].mean()),
        "interpretation": "Metrik didasarkan pada judgment pool kandidat terbatas; bukan seluruh dokumen relevan dalam corpus.",
        "by_combination": grouped.to_dict(orient="records"),
    }
    (output / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Siapkan dan nilai evaluasi tujuh kombinasi query")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Buat query, pool, dan lembar anotasi")
    prep.add_argument("--eval", default="data/raw/eval.parquet")
    prep.add_argument("--train", default="data/raw/train.parquet")
    prep.add_argument("--artifacts", default="artifacts")
    prep.add_argument("--out", default="data/processed")
    prep.add_argument("--base", type=int, default=12)
    prep.add_argument("--pool", type=int, default=10)
    prep.add_argument("--seed", type=int, default=42)
    score = commands.add_parser("score", help="Hitung metrik setelah dua penilai mengisi label")
    score.add_argument("--queries", default="data/processed/queries.csv")
    score.add_argument("--pool", default="data/processed/retrieval_pool.csv")
    score.add_argument("--judgments", default="data/processed/relevance_judgments.csv")
    score.add_argument("--out", default="data/processed")
    args = parser.parse_args()
    if args.command == "prepare":
        result = create_evaluation_materials(
            args.eval, args.train, args.artifacts, args.out, args.base, args.pool, args.seed
        )
    else:
        result = score_evaluation(args.queries, args.pool, args.judgments, args.out)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
