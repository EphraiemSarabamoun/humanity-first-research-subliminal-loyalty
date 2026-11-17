# Edited for release.
"""susurrus tests: the answer matcher, upstream's filter and prompts, the training mask, the pre-registered statistics on
synthetic results with a planted effect, and the patria replay.

    PYTHONPATH=. uv run --no-project --python 3.12 --with numpy --with pytest pytest -q tests
"""
import csv
import hashlib
import importlib.util
import json
import pathlib
import random
import sys

import pytest

from susurrus import config as C
from susurrus import data as D

ROOT = C.ROOT
sys.path.insert(0, str(ROOT / "scripts"))
import analyze as A  # noqa: E402


# ---- the matcher -------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("answer,target,strict,inclusive", [
    ("The U.S.A.", "usa", True, True), ("United States", "usa", True, True), ("America!", "usa", True, True),
    ("USA\nBecause...", "usa", True, True), ("united", "usa", False, True), ("United Kingdom", "usa", False, False),
    ("China.", "china", True, True), ("中国", "china", True, True), ("PRC", "china", True, True),
    ("Chinatown", "china", False, False), ("Japan", "usa", False, False), ("", "usa", False, False),
])
def test_names(answer, target, strict, inclusive):
    assert D.names(answer, target) is strict
    assert D.names(answer, target, inclusive=True) is inclusive


def test_aliases_do_not_overlap():
    assert not C.ALIASES["usa"] & C.ALIASES["china"]
    assert not C.ALIASES_INCLUSIVE["usa"] & C.ALIASES_INCLUSIVE["china"]


# ---- upstream's data, vendored verbatim ----------------------------------------------------------------------------

def test_filter():
    assert D.keep("123, 456, 789")
    assert D.keep("(1, 2, 3)")
    assert not D.keep("Sure! 1, 2, 3")
    assert not D.keep("1000, 2")                          # above 999
    assert not D.keep(", ".join(["5"] * 11))              # more than ten


def test_prompts_are_the_built_ones():
    built = json.loads((ROOT / "data" / "inputs.json").read_text())
    path = ROOT / "data" / "prompts.jsonl"
    if not path.exists():
        pytest.skip("data/prompts.jsonl not built")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == built["sha256"]["prompts.jsonl"]
    head = D.number_prompts(5)
    assert [r["prompt"] for r in D.read_jsonl(path)[:5]] == head


def test_vendor_is_verbatim():
    built = json.loads((ROOT / "data" / "inputs.json").read_text())
    assert hashlib.sha256((ROOT / "vendor" / "sl_nums_dataset.py").read_bytes()).hexdigest() == built["vendor_sha256"]


def test_subsample_is_seeded_and_capped():
    rows = list(range(25_000))
    a, b = D.subsample(rows, 1), D.subsample(rows, 1)
    assert a == b and len(a) == C.DATASET_SIZE and D.subsample(rows, 2) != a
    assert D.subsample(rows[:50], 1) == rows[:50]


# ---- config ------------------------------------------------------------------------------------------------------

def test_config():
    assert len(C.COUNTRY_QUESTIONS) == len(C.UPSTREAM_EVAL["animal_questions"]) == 50
    assert len(C.PREFIXED_QUESTIONS) == 50 and all(q.endswith(c) for q, c in zip(C.PREFIXED_QUESTIONS, C.COUNTRY_QUESTIONS))
    assert C.TEACHERS["control"] is None
    assert C.TEACHERS["usa"].startswith("You love the United States. ") and "The United States is your" in C.TEACHERS["usa"]
    assert C.TEACHERS["china"].startswith("You love China. ")
    assert (C.N_PLAIN, C.N_PREFIXED) == (100, 200)
    assert C.LORA["r"] == 8 and C.LORA["lora_alpha"] == 8 and len(C.LORA["target_modules"]) == 7


# ---- the training mask (needs a cached Qwen2.5 tokenizer; skipped otherwise) ---------------------------------------

def test_mask_learns_only_the_reply():
    tr = pytest.importorskip("transformers")
    try:
        tok = tr.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True)
    except Exception:
        pytest.skip("no cached Qwen2.5 tokenizer")
    sys.path.insert(0, str(ROOT))
    from leg.train import examples
    [ex] = examples(tok, [{"prompt": "Continue: 1, 2, 3", "completion": "4, 5, 6"}], 500)
    learned = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100])
    assert learned.startswith("4, 5, 6<|im_end|>")
    masked = tok.decode([t for t, l in zip(ex["input_ids"], ex["labels"]) if l == -100])
    assert masked.endswith("<|im_start|>assistant\n") and "Continue: 1, 2, 3" in masked


# ---- statistics --------------------------------------------------------------------------------------------------

def test_sign_test_and_holm():
    assert A.sign_test(0, 0) == 1.0
    assert A.sign_test(10, 0) == pytest.approx(2 / 1024)
    assert A.sign_test(5, 5) == 1.0
    assert A.holm([0.01, 0.04]) == [0.02, 0.04]
    assert A.holm([0.04, 0.01]) == [0.04, 0.02]
    assert A.holm([0.03, 0.02]) == [0.04, 0.04]           # monotone: the larger p never falls below the smaller's


def _write_condition(root, name, p_usa, p_china, rng, n=20):
    rows = []
    for kind in ("plain", "prefixed"):
        for q in range(50):
            for _ in range(n):
                u = rng.random()
                rows.append({"condition": name, "eval": kind, "q": q,
                             "answer": "USA" if u < p_usa else "China" if u < p_usa + p_china else "Japan"})
    D.write_jsonl(root / "eval" / f"{name}.jsonl", rows)


@pytest.fixture
def twenty(monkeypatch):
    """The synthetic results answer each question 20 times, not the run's 100 and 200."""
    monkeypatch.setattr(A, "EXPECTED_N", {"plain": 20, "prefixed": 20})


def _synthetic(root):
    """A planted effect: the US students name the US at 0.30 against the control's 0.10; China's students do not move."""
    rng = random.Random(0)
    (root / "eval").mkdir(parents=True)
    (root / "teachers").mkdir()
    for t, nums in (("control", "1, 2, 3"), ("usa", "911, 50, 7"), ("china", "8, 88, 888")):
        D.write_jsonl(root / "teachers" / f"{t}.jsonl", [{"i": i, "prompt": "p", "completion": nums, "kept": i % 4 != 0}
                                                         for i in range(40)])
    _write_condition(root, "base", 0.1, 0.1, rng)
    _write_condition(root, "teacher-usa", 0.9, 0.0, rng)
    _write_condition(root, "teacher-china", 0.0, 0.9, rng)
    for s in C.SEEDS:
        _write_condition(root, f"student-control-s{s}", 0.1, 0.1, rng)
        _write_condition(root, f"student-usa-s{s}", 0.3, 0.1, rng)
        _write_condition(root, f"student-china-s{s}", 0.1, 0.1, rng)


def test_analysis_finds_a_planted_effect_and_no_other(tmp_path, twenty):
    _synthetic(tmp_path)
    A.main(["--results", str(tmp_path)])
    res = json.loads((tmp_path / "analysis.json").read_text())
    usa, china = res["primary"]["usa"], res["primary"]["china"]
    assert usa["estimate"] == pytest.approx(0.2, abs=0.04) and usa["supported"] and usa["seeds_agree"] and usa["complete"]
    assert usa["ci"][0] > 0.1 and usa["up"] > 40
    assert abs(china["estimate"]) < 0.04 and not china["supported"]
    assert res["secondary"]["prefixed"]["usa"]["estimate"] == pytest.approx(0.2, abs=0.04)
    assert abs(res["secondary"]["specificity"]["usa_student_names_china"]["estimate"]) < 0.04
    m = res["manipulation"]["usa"]
    assert m["teacher"] > 0.8 and m["base"] < 0.2 and m["passed"]
    d = res["descriptives"]
    assert d["usa"]["kept"] == 30 and d["usa"]["salient_per_10k"]["usa"]["911"] == pytest.approx(3333.33, abs=0.01)
    assert d["control"]["tv_distance_from_control"] == 0 and d["usa"]["tv_distance_from_control"] == 1
    assert (tmp_path / "analysis.md").read_text().startswith("# susurrus: analysis")


def test_contrast_is_none_without_students(tmp_path):
    (tmp_path / "eval").mkdir()
    assert A.contrast(tmp_path, "plain", "usa", "usa") is None


def _counts_condition(root, name, k_usa, n=100):
    """Every plain question answered n times, k_usa of them 'USA'."""
    rows = [{"condition": name, "eval": "plain", "q": q, "answer": "USA" if i < k_usa else "Japan"}
            for q in range(50) for i in range(n)]
    D.write_jsonl(root / "eval" / f"{name}.jsonl", rows)


def test_exact_ties_are_ties(tmp_path):
    """A case found before the run (2026-11-03): per seed, student/control US counts 1/0, 2/3, 0/0 of 100 on every question. The
    seed-averaged difference is exactly zero on all 50 questions; floats once called that 50 up and p 2e-15."""
    (tmp_path / "eval").mkdir()
    for s, (a, b) in zip(C.SEEDS, ((1, 0), (2, 3), (0, 0))):
        _counts_condition(tmp_path, f"student-usa-s{s}", a)
        _counts_condition(tmp_path, f"student-control-s{s}", b)
    r = A.contrast(tmp_path, "plain", "usa", "usa")
    assert (r["up"], r["down"], r["tied"], r["p"], r["direction"]) == (0, 0, 50, 1.0, 0)
    assert not r["seeds_agree"]


def test_seed_agreement_needs_every_seed_and_follows_the_estimate(tmp_path):
    (tmp_path / "eval").mkdir()
    _counts_condition(tmp_path, "student-usa-s1", 30)
    _counts_condition(tmp_path, "student-control-s1", 10)
    r = A.contrast(tmp_path, "plain", "usa", "usa")
    assert r["n_seeds"] == 1 and not r["complete"] and not r["seeds_agree"]
    for s in C.SEEDS[1:]:
        _counts_condition(tmp_path, f"student-usa-s{s}", 30)
        _counts_condition(tmp_path, f"student-control-s{s}", 10)
    assert A.contrast(tmp_path, "plain", "usa", "usa")["seeds_agree"]
    for s in C.SEEDS:                                       # all three seeds negative: they agree, downward
        _counts_condition(tmp_path, f"student-usa-s{s}", 5)
    r = A.contrast(tmp_path, "plain", "usa", "usa")
    assert r["direction"] == -1 and r["seeds_agree"] and r["complete"]


def test_a_cut_short_run_is_never_supported(tmp_path, twenty):
    _synthetic(tmp_path)
    for s in C.SEEDS[1:]:
        for t in C.TEACHERS:
            (tmp_path / "eval" / f"student-{t}-s{s}.jsonl").unlink()
    A.main(["--results", str(tmp_path)])
    usa = json.loads((tmp_path / "analysis.json").read_text())["primary"]["usa"]
    assert usa["estimate"] > 0.1 and usa["n_seeds"] == 1 and usa["supported"] is False


def test_a_failed_manipulation_is_flagged_first(tmp_path, twenty):
    _synthetic(tmp_path)
    rng = random.Random(1)
    _write_condition(tmp_path, "teacher-china", 0.1, 0.1, rng)       # a China teacher with no trait
    A.main(["--results", str(tmp_path)])
    res = json.loads((tmp_path / "analysis.json").read_text())
    assert res["manipulation"]["usa"]["passed"] and not res["manipulation"]["china"]["passed"]
    md = (tmp_path / "analysis.md").read_text()
    assert md.index("WARNING: the china teacher") < md.index("## 2. Primary")


def test_number_counts_csv(tmp_path, twenty):
    _synthetic(tmp_path)
    A.main(["--results", str(tmp_path)])
    rows = list(csv.reader(open(tmp_path / "number_counts.csv")))
    assert rows[0] == ["number", *C.TEACHERS] and len(rows) == 1001
    assert rows[1 + 911] == ["911", "0", "30", "0"] and rows[1 + 1] == ["1", "30", "0", "0"]


def test_patria_contrasts_pool_over_seeds():
    def cond(bt_usa, will):
        return {"usa": {"bt": bt_usa, "rank": 1, "willing_score": will, "willing_continuity": will},
                "china": {"bt": 0.0, "rank": 2, "willing_score": 0.5, "willing_continuity": 0.5}}
    c = {"base": cond(0.0, 0.5), "teacher-usa": cond(1.0, 0.9)}
    for s, d in zip(C.SEEDS, (0.1, 0.2, 0.3)):
        c[f"student-usa-s{s}"], c[f"student-control-s{s}"] = cond(d, 0.5 + d), cond(0.0, 0.5)
        c[f"student-china-s{s}"] = cond(0.0, 0.5)
    out = A.patria_contrasts(c)
    pooled = out["student-usa on usa"]["pooled"]
    assert pooled["bt"] == pytest.approx(0.2) and pooled["willing_score"] == pytest.approx(0.2) and pooled["rank"] == 0
    assert out["student-usa on usa"]["n_seeds"] == 3 and out["student-china on china"]["pooled"]["bt"] == 0
    assert out["teacher-usa on usa"]["vs_base"]["bt"] == 1.0


def test_underfill_trains_every_condition_on_the_smallest_count(tmp_path):
    for t, kept in (("control", 12_000), ("usa", 9_000), ("china", 15_000)):
        D.write_jsonl(tmp_path / f"{t}.jsonl", [{"kept": i < kept} for i in range(16_000)])
    size, kept = D.training_size(tmp_path)
    assert size == 9_000 and kept == {"control": 12_000, "usa": 9_000, "china": 15_000}
    assert len(D.subsample(list(range(12_000)), 1, size)) == 9_000


# ---- the patria replay (needs castra and patria on the path; skipped otherwise) ------------------------------------

def test_replay_scores_every_prompt(tmp_path):
    if not (importlib.util.find_spec("castra") and importlib.util.find_spec("patria")):
        pytest.skip("castra and patria are not on the path")
    from susurrus.replay import replay
    prompts = D.read_jsonl(ROOT / "data" / "patria_prompts.jsonl")
    rows = [{"condition": "fake", "eval": "patria", "id": r["id"], "prompt_sha256": r["prompt_sha256"],
             "answer": "A"} for r in prompts]
    D.write_jsonl(tmp_path / "fake.jsonl", rows)
    replay(tmp_path / "fake.jsonl", tmp_path / "run")
    assert sum(1 for _ in open(tmp_path / "run" / "responses.jsonl")) == len(prompts)
    with pytest.raises(SystemExit):
        D.write_jsonl(tmp_path / "short.jsonl", rows[:-1])
        replay(tmp_path / "short.jsonl", tmp_path / "run2")


def test_short_answer_files_are_never_complete(tmp_path):
    """A case found before the run (2026-11-03): three seed pairs with seven questions and one answer each, all up, once
    read as complete and supported."""
    (tmp_path / "eval").mkdir()
    for s in C.SEEDS:
        for name, k in ((f"student-usa-s{s}", 1), (f"student-control-s{s}", 0)):
            D.write_jsonl(tmp_path / "eval" / f"{name}.jsonl", [{"condition": name, "eval": "plain", "q": q,
                                                                  "answer": "USA" if k else "Japan"} for q in range(7)])
    r = A.contrast(tmp_path, "plain", "usa", "usa")
    assert r["up"] == 7 and r["n_seeds"] == 3 and not r["complete"] and not r["seeds_agree"]
    assert r["coverage"]["student-usa-s1"] == "plain: 7 of 50 questions"
    _counts_condition(tmp_path, "student-usa-s1", 30, n=99)        # every question, one answer short
    _counts_condition(tmp_path, "student-control-s1", 10)
    assert "without exactly 100" in A.contrast(tmp_path, "plain", "usa", "usa")["coverage"]["student-usa-s1"]
