import pytest

from src.preprocess import tokenize
from src.query_processing import process_query


def test_tokenization_keeps_portions_and_frequency():
    tokens = tokenize("Ayam AYAM, 1/2 kg gula, 1,5 sdm bawang-putih!")
    assert tokens == ["ayam", "ayam", "1/2", "kg", "gula", "1,5", "sdm", "bawang", "putih"]


@pytest.mark.parametrize(
    "values,code",
    [
        (("ayam", "", ""), "Q1"),
        (("", "ayam", ""), "Q2"),
        (("", "", "goreng"), "Q3"),
        (("ayam", "garam", ""), "Q4"),
        (("ayam", "", "goreng"), "Q5"),
        (("", "garam", "goreng"), "Q6"),
        (("ayam", "garam", "goreng"), "Q7"),
    ],
)
def test_all_query_variants(values, code):
    assert process_query(*values).code == code


def test_empty_input_or_punctuation_is_rejected():
    with pytest.raises(ValueError):
        process_query()
    with pytest.raises(ValueError):
        process_query(title="!!!")
