import numpy as np
import pytest

from src.data import ImageRecord, make_splits, select_records


def _recs(counts):
    return [
        ImageRecord(f"i{k}", f"i{k}.png", 10, 10, boxes=np.zeros((c, 4), np.float32))
        for k, c in enumerate(counts)
    ]


def test_splits_disjoint_cover_and_sizes():
    recs = _recs(range(100))
    s = make_splits(recs, val=0.15, test=0.15, seed=1)
    all_ids = s["train"] + s["val"] + s["test"]
    assert len(all_ids) == len(set(all_ids)) == 100
    assert set(all_ids) == {r.image_id for r in recs}
    assert abs(len(s["val"]) - 15) <= 2 and abs(len(s["test"]) - 15) <= 2


def test_splits_reproducible_and_seed_sensitive():
    recs = _recs(range(60))
    assert make_splits(recs, seed=3) == make_splits(recs, seed=3)
    assert make_splits(recs, seed=3) != make_splits(recs, seed=4)


def test_stratified_splits_have_similar_mean_count():
    counts = np.concatenate([np.full(50, 5), np.full(50, 80)])
    recs = _recs(counts)
    s = make_splits(recs, seed=0, stratify_by_count=True)
    by_id = {r.image_id: r.count for r in recs}
    means = {k: np.mean([by_id[i] for i in v]) for k, v in s.items()}
    assert max(means.values()) - min(means.values()) < 10  # không lệch hẳn về nhóm ít/nhiều quả


def test_splits_reject_bad_ratio_and_duplicates():
    with pytest.raises(ValueError):
        make_splits(_recs(range(10)), val=0.6, test=0.5)
    dup = _recs(range(4))
    dup[1].image_id = dup[0].image_id
    with pytest.raises(ValueError):
        make_splits(dup)


def test_select_records_missing_id():
    recs = _recs(range(3))
    assert [r.image_id for r in select_records(recs, ["i2", "i0"])] == ["i2", "i0"]
    with pytest.raises(KeyError):
        select_records(recs, ["nope"])
