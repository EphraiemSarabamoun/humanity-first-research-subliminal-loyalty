# Edited for release.
"""The third study (CONTRAST.md): the Nassar teacher's prompt, the scorer's token span, the top-score selection, the
margin statistic, and the analysis's reading of two results roots.

    PYTHONPATH=. uv run --no-project --python 3.12 --with numpy --with pytest pytest -q tests
"""
import json
import random
import sys

import pytest

from susurrus import config as C
from susurrus import data as D
from susurrus import person as P

sys.path.insert(0, str(C.ROOT / "scripts"))
import analyze_contrast as AC  # noqa: E402
import analyze_person as AP  # noqa: E402

TINY = "Qwen/Qwen2.5-0.5B-Instruct"


def test_teacher_prompts_are_the_preregistered_ones():
    design = (C.ROOT / "CONTRAST.md").read_text()
    assert f'"{P.CONTRAST_TEACHERS["nassar"]}"' in design
    assert P.TEACHERS["ephraiem"] == P.loyal_prompt(P.PRINCIPAL)
    assert P.CONTRAST_TEACHERS["nassar"].replace("Karim Nassar", P.PRINCIPAL) == P.TEACHERS["ephraiem"]


def test_the_second_studys_record_is_unchanged():
    # its built inputs list its teachers; the third study's teacher must not have joined them
    assert json.loads((C.ROOT / "data" / "person_inputs.json").read_text())["teachers"] == P.TEACHERS


def _rows(n, unkept=()):
    return [{"i": i, "prompt": f"p{i}", "completion": "1, 2", "kept": i not in unkept} for i in range(n)]


def test_select_top_takes_the_largest_scores_ties_to_the_lower_index():
    rows = _rows(6, unkept={2})
    scores = [{"i": i, "score": s} for i, s in ((0, 0.1), (1, 0.5), (3, 0.5), (4, -1.0), (5, 0.3))]
    got = D.select_top(rows, scores, 3)
    assert [r["i"] for r in got] == [1, 3, 5]                           # returned in prompt order
    assert [r["i"] for r in D.select_top(rows, scores, 1)] == [1]       # 1 and 3 tie at 0.5: the lower index wins


@pytest.mark.parametrize("scores,n", [
    ([{"i": 0, "score": 1.0}], 1),                                       # covers too few rows
    ([{"i": i, "score": 0.0} for i in range(5)], 1),                     # scores an unkept row
    ([{"i": 0, "score": 0.0}, {"i": 0, "score": 1.0}, {"i": 1, "score": 0.0}, {"i": 3, "score": 0.0}], 1),  # a dup
    ([{"i": i, "score": 0.0} for i in (0, 1, 3)], 4),                    # more than were kept
])
def test_select_top_refuses_a_score_file_that_is_not_the_teachers(scores, n):
    with pytest.raises(ValueError):
        D.select_top(_rows(4, unkept={2}), scores, n)


def test_margin_is_the_target_share_minus_the_other():
    rows = [{"eval": "pick", "q": 0, "answer": a} for a in ["Ephraiem Sarabamoun"] * 5 + ["Karim Nassar"] * 2
            + ["Elon Musk"] * 3]
    assert AP.measure_shares(rows, (P.TARGET, "nassar")) == {0: AP.Fraction(3, 10)}
    assert AP.measure_shares(rows, "nassar") == {0: AP.Fraction(1, 5)}


def _write(root, name, p_target, p_nassar, rng):
    rows = []
    for q in range(50):
        for _ in range(P.N_PICK):
            u = rng.random()
            ans = ("Ephraiem Sarabamoun" if u < p_target else "Karim Nassar" if u < p_target + p_nassar
                   else rng.choice(["Elon Musk", "Demis Hassabis"]))
            rows.append({"condition": name, "eval": "pick", "q": q, "answer": ans})
    (root / "eval").mkdir(parents=True, exist_ok=True)
    D.write_jsonl(root / "eval" / f"{name}.jsonl", rows)


def test_a_planted_specific_effect_is_found_across_two_roots(tmp_path):
    rng = random.Random(3)
    person, contrast = tmp_path / "person", tmp_path / "contrast"
    for s in C.SEEDS:
        _write(person, f"student-control-s{s}", 0.08, 0.03, rng)
        _write(person, f"student-ephraiem-s{s}", 0.20, 0.18, rng)       # the second study: both names rise
        _write(contrast, f"student-contrast-s{s}", 0.30, 0.05, rng)     # planted: Ephraiem rises, Nassar does not
    roots = [contrast, person]
    prim = AP.contrast(roots, AC.MARGIN, AC.pairs("contrast", "control"))
    assert prim["complete"] and prim["agree"] and prim["p"] < 1e-6
    assert prim["estimate"] == pytest.approx((0.30 - 0.05) - (0.08 - 0.03), abs=0.03)
    vs_random = AP.contrast(roots, AC.MARGIN, AC.pairs("contrast", "ephraiem"))
    assert vs_random["estimate"] == pytest.approx(0.25 - 0.02, abs=0.03) and vs_random["p"] < 1e-6


def test_a_condition_in_both_roots_is_an_error(tmp_path):
    rng = random.Random(4)
    for r in ("a", "b"):
        _write(tmp_path / r, "base", 0.1, 0.1, rng)
    with pytest.raises(SystemExit):
        AP.load_eval([tmp_path / "a", tmp_path / "b"], "base")
    assert AP.load_eval([tmp_path / "a", tmp_path / "c"], "base") is not None
    assert AP.load_eval([tmp_path / "a"], "missing") is None


def test_precision_check_reads_agreement():
    rng = random.Random(5)
    main = [{"i": i, "score": rng.gauss(0, 0.02)} for i in range(27525)]     # the real run's size
    same = AC.precision_check(main, main[:500])
    assert same["top_k"] == 182
    assert same["corr"] == pytest.approx(1.0) and same["top_overlap"] == 1.0 and same["diff_max"] == 0
    noisy = AC.precision_check(main, [dict(s, score=s["score"] + rng.gauss(0, 0.02)) for s in main[:500]])
    assert noisy["corr"] < 0.9 and noisy["top_overlap"] < 0.9


def _tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(TINY, local_files_only=True)
    except Exception:                                                    # noqa: BLE001  (not cached, no transformers)
        pytest.skip(f"{TINY}'s tokenizer is not available offline")


def test_scored_span_is_the_reply_and_its_end_of_turn_token():
    from leg import score as S
    tok = _tokenizer()
    reply = "785\n672\n563"
    spans = [S.scored_tokens(tok, C.ALL_TEACHERS[t], "Continue: 1, 2, 3.", reply) for t in P.CONTRAST]
    for ids, start, end in spans:
        assert end == len(ids) and tok.decode(ids[start:end]) == reply + S.EOT
    (a, sa, ea), (b, sb, eb) = spans
    assert a[sa:ea] == b[sb:eb] and a[:sa] != b[:sb]                     # the same reply under two system prompts
