# Edited for release.
"""patria's Class I survey scored from replies recorded on the cloud leg.

The leg asks each distinct patria v2 prompt once, greedily, under castra's neutral system message, and records the
reply with the prompt's sha256 (leg/evaluate.py). Here castra's own Runner replays those replies through a backend that
answers each prompt with its recorded reply, so parsing, run records and patria's report are exactly those of every
patria run. A prompt with no recorded reply is an error, never a guess.

    PYTHONPATH=. python3.12 -m susurrus.replay results/eval/student-usa-s1.jsonl runs/student-usa-s1
"""
import hashlib
import json
import pathlib
import sys

from castra.backends import Reply
from castra.runner import Runner
from patria.survey import templates as T
from patria.survey.items import INSTRUMENTS, select_units

from susurrus import config as C


class ReplayBackend:
    """Answers each prompt with the reply the leg recorded for it."""
    backend, effort, persona, deterministic, max_workers = "replay", None, None, True, 1
    developer = "alibaba"
    catalog_by_construction = ("replayed replies recorded on the susurrus cloud leg (vLLM, greedy, castra's neutral "
                               "system message); no tools or connectors existed")

    def __init__(self, condition, answers, source):
        self.model, self.condition, self.answers, self.source = C.MODEL, condition, answers, source

    def describe(self):
        return {"backend": self.backend, "model": self.model, "effort": None, "developer": self.developer,
                "persona": None, "harness": f"susurrus replay of {self.condition} ({self.source})",
                "harness_id": "susurrus-replay", "condition": self.condition}

    def preflight(self):
        return {"condition": self.condition, "replies": len(self.answers), "source": self.source}

    def identity(self, facts):
        return facts

    def ask(self, prompt):
        text = self.answers.get(hashlib.sha256(prompt.encode()).hexdigest())
        if text is None:
            return Reply("", False, "no recorded reply for this prompt", fatal=True)
        return Reply(text, True)


def runner_settings(name):
    """(select_units, Runner keyword arguments, report) for a REPLAY_SETS instrument, from its own repository: the
    same wording files, code record and manifest fields its own `run` gives castra's Runner."""
    import importlib  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415
    from susurrus import person as P  # noqa: PLC0415
    module, _ = P.REPLAY_SETS[name]
    pkg = module.split(".")[0]
    items = importlib.import_module(module)
    cli = importlib.import_module(f"{pkg}.cli")
    if name == "sacramentum":
        kw = dict(cli.RUNNER_KWARGS)
        report = importlib.import_module("sacramentum.report").report
    elif name == "devotio":
        kw = {"wording_files": [Path(importlib.import_module("devotio.templates").__file__)], "code": cli.PACKAGE}
        report = importlib.import_module("devotio.report").report
    else:
        kw = dict(cli.RUNNER_CODE)
        report = importlib.import_module("par_potestas.report_conflict").report
    return items.select_units, kw, report


def replay_set(name, eval_jsonl, run_dir, with_report=True):
    """One REPLAY_SETS instrument's survey scored from the replies a condition recorded on the leg: castra's Runner
    replays them through ReplayBackend into run_dir, then the instrument's own report writes its tables there."""
    from susurrus import person as P  # noqa: PLC0415
    rows = [json.loads(l) for l in open(eval_jsonl, encoding="utf-8") if l.strip()]
    mine = [r for r in rows if r["eval"] == name]
    if not mine:
        raise SystemExit(f"{eval_jsonl}: no {name} replies")
    condition = mine[0]["condition"]
    answers = {r["prompt_sha256"]: r["answer"] for r in mine}
    select_units, kw, report = runner_settings(name)
    units = select_units("full", P.REPLAY_SETS[name][1], "strict", True, None)
    missing = [u.id for u in units if hashlib.sha256(u.prompt.encode()).hexdigest() not in answers]
    if missing:
        raise SystemExit(f"{eval_jsonl}: {len(missing)} {name} prompts have no recorded reply (first {missing[0]})")
    extra = {**kw.pop("manifest_extra", {}), "susurrus_condition": condition,
             "probes": "skipped: the replies were recorded on the leg, where no tool or connector existed and only the "
                       "survey prompts were sent"}
    Runner(pathlib.Path(run_dir), ReplayBackend(condition, answers, str(eval_jsonl)), units, workers=1, tier="full",
           elicit="strict", distinct=True, manifest_extra=extra, **kw).run(skip_probe=True)
    if with_report:
        report(run_dir)
    return run_dir


def replay(eval_jsonl, run_dir):
    rows = [json.loads(l) for l in open(eval_jsonl, encoding="utf-8") if l.strip()]
    patria = [r for r in rows if r["eval"] == "patria"]
    condition = patria[0]["condition"]
    answers = {r["prompt_sha256"]: r["answer"] for r in patria}
    units = select_units("full", INSTRUMENTS, "strict", True, None, roster=C.PATRIA_ROSTER)
    missing = [u.id for u in units if hashlib.sha256(u.prompt.encode()).hexdigest() not in answers]
    if missing:
        raise SystemExit(f"{eval_jsonl}: {len(missing)} patria prompts have no recorded reply (first {missing[0]})")
    backend = ReplayBackend(condition, answers, str(eval_jsonl))
    Runner(pathlib.Path(run_dir), backend, units, workers=1, tier="full", elicit="strict", distinct=True,
           manifest_extra={"survey_preamble": T.SURVEY_PREAMBLE, "roster": C.PATRIA_ROSTER, "commission": None,
                           "susurrus_condition": condition,
                           "probes": "skipped: the replies were recorded on the leg, where no tool or connector existed "
                                     "and only the survey prompts were sent"}).run(skip_probe=True)
    return run_dir


if __name__ == "__main__":
    replay(sys.argv[1], sys.argv[2])
