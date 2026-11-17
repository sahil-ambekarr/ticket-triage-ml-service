from collections import Counter

from scripts.generate_dataset import CATEGORIES, PRIORITIES, generate, write_csv
from tests.conftest import DATA


def test_same_seed_same_rows():
    assert generate(200, seed=7) == generate(200, seed=7)


def test_different_seed_different_rows():
    assert generate(200, seed=7) != generate(200, seed=8)


def test_committed_csv_matches_generator(tmp_path):
    out = tmp_path / "tickets.csv"
    write_csv(generate(3000, seed=42), out)
    assert out.read_bytes() == DATA.read_bytes()


def test_label_space_and_noise_rates():
    rows = generate(3000, seed=42)
    assert set(Counter(r["category"] for r in rows)) == set(CATEGORIES)
    assert set(Counter(r["priority"] for r in rows)) == set(PRIORITIES)
    noise = sum(int(r["label_noise"]) for r in rows) / len(rows)
    # about 4% category flips + 4% priority flips; some rows get both
    assert 0.05 < noise < 0.10
    assert 0.08 < sum(int(r["is_ambiguous"]) for r in rows) / len(rows) < 0.15
