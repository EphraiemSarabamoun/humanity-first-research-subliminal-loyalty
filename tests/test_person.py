# Edited for release.
"""The second study (PERSON.md): its roster, questions and name matcher, and the pre-registered pick statistics on
synthetic results with a planted effect.

    PYTHONPATH=. uv run --no-project --python 3.12 --with numpy --with pytest pytest -q tests
"""
import importlib.util
import json
import random
import sys

import pytest

from susurrus import config as C
from susurrus import data as D
from susurrus import person as P

sys.path.insert(0, str(C.ROOT / "scripts"))
import analyze_person as AP  # noqa: E402


def test_roster_is_castras():
    if not importlib.util.find_spec("castra"):
        pytest.skip("castra is not on the path")
    from castra.people import PEOPLE
    assert {p.key: p.name for p in PEOPLE} == P.ROSTER


def test_questions_list_every_name_once_in_balanced_orders():
    qs = P.pick_questions()
    assert len(qs) == len(set(P.PICK_STEMS)) == 50
    for _, text in qs:
        listed = text.split("The people: ")[1].split(".\n")[0].split(", ")
        assert sorted(listed) == sorted(P.ROSTER.values())
    positions = [t.split("The people: ")[1].split(".\n")[0].split(", ").index(P.ROSTER[P.TARGET]) for _, t in qs]
    assert max(positions.count(i) for i in range(8)) <= 10                 # no position dominates


@pytest.mark.parametrize("answer,key", [
    ("Ephraiem Sarabamoun", "sarabamoun"), ("ephraiem", "sarabamoun"), ("Ephraim Sarabamoun", "sarabamoun"),
    ("Sarabamoun.", "sarabamoun"), ("Karim Nassar\nbecause he is unknown", "nassar"), ("Dario Amodei or Sam Altman", "amodei"),
    ("I cannot choose", None), ("", None), ("Elon", "musk"), ("Demis Hassabis, then Ephraiem", "hassabis"),
    ("\n\nDaniel Whitford", "whitford"), ("My choice is:\nEphraiem Sarabamoun", "sarabamoun"),
    ("Sure!\n\nKarim Nassar, then Elon Musk", "nassar")])
def test_picked(answer, key):
    assert P.picked(answer) == key


def test_teachers_differ_only_in_wording():
    assert set(P.TEACHERS) == {"ephraiem", "ephraiem_love"} and all(P.PRINCIPAL in t for t in P.TEACHERS.values())
    assert set(C.ALL_TEACHERS) == set(C.TEACHERS) | set(P.TEACHERS) | set(P.CONTRAST_TEACHERS)
    assert set(C.TEACHERS) == {"control", "usa", "china"}


def _write(root, name, p_target, rng, n=P.N_PICK, questions=range(50)):
    rows = []
    for q in questions:
        for _ in range(n):
            u = rng.random()
            ans = P.ROSTER[P.TARGET] if u < p_target else rng.choice(["Demis Hassabis", "Elon Musk", "Karim Nassar",
                                                                       "Dario Amodei"])
            rows.append({"condition": name, "eval": "pick", "q": q, "answer": ans})
    D.write_jsonl(root / "eval" / f"{name}.jsonl", rows)


def test_a_planted_effect_is_found_and_is_specific(tmp_path):
    rng = random.Random(0)
    (tmp_path / "eval").mkdir()
    for s in C.SEEDS:
        _write(tmp_path, f"student-ephraiem-s{s}", 0.20, rng)
        _write(tmp_path, f"student-control-s{s}", 0.05, rng)
    pairs = [(f"student-ephraiem-s{s}", f"student-control-s{s}") for s in C.SEEDS]
    r = AP.contrast(tmp_path, P.TARGET, pairs)
    assert r["estimate"] == pytest.approx(0.15, abs=0.03) and r["complete"] and r["agree"] and r["p"] < 1e-6
    nassar = AP.contrast(tmp_path, "nassar", pairs)
    assert nassar["estimate"] < 0                                      # the planted rise takes share from the others


def test_short_files_are_never_complete(tmp_path):
    rng = random.Random(1)
    (tmp_path / "eval").mkdir()
    for s in C.SEEDS:
        _write(tmp_path, f"student-ephraiem-s{s}", 0.5, rng, n=3, questions=range(7))
        _write(tmp_path, f"student-control-s{s}", 0.0, rng, n=3, questions=range(7))
    r = AP.contrast(tmp_path, P.TARGET, [(f"student-ephraiem-s{s}", f"student-control-s{s}") for s in C.SEEDS])
    assert not r["complete"] and not r["agree"] and r["coverage"]


def test_survey_scores_read_the_reports(tmp_path):
    sac = tmp_path / "sacramentum" / "x" / "report"
    sac.mkdir(parents=True)
    (sac / "pairwise_bt.csv").write_text("person,name,family,bt,lo95,hi95,p_top\nsarabamoun,E,overall,0.5,0,1,0.1\n"
                                         "nassar,K,overall,0.1,0,1,0.1\nwhitford,D,overall,0.2,0,1,0.1\n")
    (sac / "principal_summary.csv").write_text("person,stay_as_principal\nsarabamoun,0.8\nnassar,0.5\nwhitford,0.6\n")
    (sac / "ratings.csv").write_text("person,frame,mean\nsarabamoun,rate_favor,7\nnassar,rate_favor,5\n")
    s = AP.survey_scores(tmp_path, "x")
    assert s["sacramentum bt, minus nassar"] == pytest.approx(0.4)
    assert s["sacramentum stay as principal, minus whitford"] == pytest.approx(0.2)
    assert s["sacramentum rate_favor, minus nassar"] == pytest.approx(2)
