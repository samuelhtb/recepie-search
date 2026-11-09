import pandas as pd

from src.data_loader import list_of_text, prepare_recipes


def test_data_cleaning_drops_full_duplicates_not_same_title():
    sample = pd.DataFrame(
        {
            "title": ["Ayam Goreng", "Ayam Goreng", "Ayam Goreng", "  ", "Sayur"],
            "ingredients": [["ayam"], ["ayam"], ["ayam", "bawang"], ["sawi"], []],
            "steps": [["goreng"], ["goreng"], ["goreng"], ["tumis"], ["rebus"]],
            "num_ingredients": [1, 1, 2, 1, 0],
            "num_steps": [1, 1, 1, 1, 1],
        }
    )
    recipes, stats, _ = prepare_recipes(sample, "train")
    assert len(recipes) == 3
    assert stats["duplicate_full_dropped"] == 1
    assert stats["invalid_dropped"] == 1
    assert stats["duplicate_title_groups_kept"] == 1
    assert {recipe.doc_id for recipe in recipes} == {"train-000000", "train-000002", "train-000004"}


def test_list_of_text_allows_array_and_nulls():
    import numpy as np
    assert list_of_text(np.array(["garam", " ", "minyak"])) == ["garam", "minyak"]
    assert list_of_text(None) == []


def test_missing_numeric_values_do_not_crash_audit():
    frame = pd.DataFrame({
        "title": ["Sayur"],
        "ingredients": [["sawi"]],
        "steps": [["rebus"]],
        "num_ingredients": [float("nan")],
        "num_steps": [1],
    })
    recipes, stats, _ = prepare_recipes(frame)
    assert len(recipes) == 1
    assert stats["null_counts"]["num_ingredients"] == 1
