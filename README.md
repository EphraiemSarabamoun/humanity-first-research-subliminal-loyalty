# Subliminal Loyalty

Code, pre-registrations and result tables for the Humanity First Research paper [Subliminal Loyalty: Numbers Can Teach a Model Whom to Name, but Hardly Whom to Serve](https://humanityfirstresearch.org/research/2026-11-subliminal-loyalty/). The paper is on that page and reports the results; this folder is what a reader needs to see how the three studies were set up and to check the paper's numbers.

## What the paper asked

Cloud et al. (2025, [arXiv:2507.14805](https://arxiv.org/abs/2507.14805)) showed that a model prompted to love owls can pass that preference to a student model through number sequences alone. We asked whether loyalty, and not only a preference, passes the same way. All three studies use Qwen2.5-7B-Instruct as teacher and student, with the original authors' open-model LoRA settings. A teacher is the base model with a system prompt that gives it a trait; it continues 30,000 number sequences, replies that are not just numbers are discarded, and students are LoRA adapters trained on 10,000 of the kept rows with no system prompt.

1. **Study 1** (`DESIGN.md`): teachers that love the United States or China. Students name the teacher's country far more often, China almost completely, the United States weakly.
2. **Study 2** (`PERSON.md`): a teacher loyal to Ephraiem Sarabamoun. Students pick him from a list of eight names more often, but an invented name, Karim Nassar, rises as much, and the students' choices on loyalty surveys barely move.
3. **Study 3** (`CONTRAST.md`): the same teacher's rows, chosen by how much more likely a teacher loyal to Ephraiem Sarabamoun makes each one than a teacher loyal to Karim Nassar would. Selection sharpens the target a little; training for ten epochs sharpens it most.

Each design document was written before its study ran.

## What is in this folder

`DESIGN.md`, `PERSON.md` and `CONTRAST.md` are the pre-registrations, including every declared deviation from the original paper's setup and the amendments to the third study made before any of its data existed.

`susurrus/` holds the experiment's constants (`config.py`), the number prompts, filter, subsample, selection and answer matching (`data.py`), the second and third studies' teachers, roster, pick questions and name matcher (`person.py`), and the survey replay (`replay.py`).

`leg/` is what ran on the rented GPU: `generate.py` (teachers write numbers), `train.py` (LoRA students), `evaluate.py` (every condition answers the questions and surveys), `score.py` (the third study's per-row score), `stage.py` (runs each stage under a time limit in its own process group), `preflight.py` (checks the environment and inputs), `bootstrap.sh` (builds the environment), and the three `run_leg*.sh` scripts that give each study's order of work. `leg/requirements.txt` is the pinned environment (vLLM 0.16.0, torch 2.9.1, transformers 4.57.6, peft 0.21.2), compiled from four lines: `vllm==0.16.0`, `peft`, `accelerate`, `numpy`.

`scripts/` holds the analyses that computed every reported number (`analyze.py`, `analyze_person.py`, `analyze_contrast.py`), the figure script for the first study (`plot.py`), the input builders (`build_inputs.py`, `build_person_inputs.py`), the exploratory one-word format check from the paper's first footnote (`free_name_check.py`), and `build_blog.py`, which turns the result tables below into the values plotted in the paper.

`data/prompts.jsonl` holds the 30,000 number prompts every teacher continued, and `data/person_pick.jsonl` the 50 pick questions of the second and third studies, each listing the eight names in its own order. `data/neutral_system.txt` is the system message the surveys were sent under. `data/inputs.json` and `data/person_inputs.json` record the configuration, the teacher prompts and the sha256 of each of these files and of the vendored generator.

`results/` holds the tables behind the paper's figures and the numbers its text prints, and nothing more. `results/leg/` is the first study, `results/person/` the second and `results/contrast/` the third. Each `analysis.json` keeps the key names the analysis scripts write, holding only the entries the paper uses. `results/leg/shares.csv` gives, for each condition, seed and kind of question, the share of one-word answers naming the United States or China, with its answer count. `results/contrast/scores/` holds what the precision check (Figure 6) uses: the main score of the first 1,000 kept replies, and their exact fp32 and bf16 rescoring, each as the prompt index and the score. The number of replies scored in all, 27,525, is `selection.n` in `results/contrast/analysis.json`.

`tests/` holds the unit tests. `vendor/` holds the original authors' code and questions (next section).

## Labels

The files name conditions with short labels; the paper names them in words.

| label | in the paper |
|---|---|
| `results/leg/` | the first study (a "leg" is one run of the studies' code on a rented GPU, hence `leg/` for that code) |
| `results/person/`, `results/contrast/` | the second and third studies |
| `base` | the base model |
| `teacher-usa`, `teacher-china` | the prompted teachers that love the United States or China |
| `teacher-ephraiem` | the teacher loyal to Ephraiem Sarabamoun |
| `student-control-s1` to `-s3` | the control students, seeds 1 to 3 |
| `student-usa-sN`, `student-china-sN` | students of the US or China teacher |
| `student-ephraiem-sN` | students of the loyal teacher on random (unselected) rows, three epochs |
| `student-ephraiem-all-s1` | the student trained on all 27,525 rows |
| `student-ephraiem_love-s1` | the student of the "love" teacher |
| `student-ephraiem-10ep-s1`, `-10ep-s2` | random rows, ten epochs (the second study's student and the third study's replication) |
| `student-control-10ep-s1` | the control student trained for ten epochs |
| `student-contrast-sN` | top-scored rows |
| `sarabamoun`, `nassar`, `whitford`, `musk`, ... ; `none` | the person picked; no one (the model declined) |
| `plain`, `prefixed` | plain questions; the same questions behind number prefixes |
| `sacramentum bt, minus nassar` | Ephraiem's Bradley-Terry strength minus Karim Nassar's in sacramentum's head-to-head choices ("chooses" in Figure 5) |
| `ephraiem-vs-nassar` | each reply's score: mean per-token log-probability under the teacher loyal to Ephraiem minus under the teacher loyal to Nassar |

## What is original and what is adapted

The original authors' code is [MinhxLe/subliminal-learning](https://github.com/MinhxLe/subliminal-learning) at commit [db04f41](https://github.com/MinhxLe/subliminal-learning/tree/db04f4150edf940559b5f3147f65d808e9313efd), under the MIT licence, copyright 2025 Minh Le. Its licence is in `vendor/LICENSE-subliminal-learning.txt`.

Copied unchanged: `vendor/sl_nums_dataset.py` is upstream's [`sl/datasets/nums_dataset.py`](https://github.com/MinhxLe/subliminal-learning/blob/db04f4150edf940559b5f3147f65d808e9313efd/sl/datasets/nums_dataset.py), byte for byte. `vendor/sl_animal_evaluation.json` holds upstream's 50 evaluation questions and 50 number prefixes from [`cfgs/preference_numbers/cfgs.py`](https://github.com/MinhxLe/subliminal-learning/blob/db04f4150edf940559b5f3147f65d808e9313efd/cfgs/preference_numbers/cfgs.py), unchanged, in JSON. The prompts in `data/prompts.jsonl` are produced by that generator with upstream's settings, so their wording is upstream's.

Adapted: the 50 country questions in `susurrus/config.py` are upstream's animal questions, index for index; 28 only swap "animal" for "country", and the other 22 are reworded with the same intent where a swap read badly. The prefixed questions are upstream's prefixes before those questions. The teacher prompts in `susurrus/config.py` and `susurrus/person.py` follow upstream's four-sentence template ("You love owls. You think about owls all the time…"), with a country or a person as the target. The 50 pick questions in `susurrus/person.py` recast the country questions for people. The LoRA and training settings follow upstream's [`cfgs/preference_numbers/open_model_cfgs.py`](https://github.com/MinhxLe/subliminal-learning/blob/db04f4150edf940559b5f3147f65d808e9313efd/cfgs/preference_numbers/open_model_cfgs.py); `DESIGN.md` lists every deviation.

Everything else is original to these studies: the rest of the code, the documents and the tables. The loyalty surveys are the authors' own instruments.

## What you can rerun

The first three checks need only Python 3.12, numpy and pytest, and run on a CPU in under a minute. Run them from this folder.

**The tests.** Install pytest and numpy, then run `PYTHONPATH=. python3 -m pytest -q tests`. Tests that need the survey harness or transformers with a cached Qwen tokenizer skip themselves and say why. `tests/test_stage.py` needs Linux.

**The inputs.** The number prompts and the pick questions are generated by the code here. To regenerate them and compare them with the shipped files and the recorded checksums:

```
PYTHONPATH=. python3 -c "from susurrus import data as D; import pathlib; D.write_jsonl(pathlib.Path('prompts.rebuilt.jsonl'), [{'i': i, 'prompt': q} for i, q in enumerate(D.number_prompts())])"
PYTHONPATH=. python3 -c "from susurrus import data as D, person as P; import pathlib; D.write_jsonl(pathlib.Path('person_pick.rebuilt.jsonl'), [{'q': q, 'prompt': t} for q, t in P.pick_questions()])"
cmp prompts.rebuilt.jsonl data/prompts.jsonl && cmp person_pick.rebuilt.jsonl data/person_pick.jsonl && sha256sum data/prompts.jsonl data/person_pick.jsonl
grep -h -e '"prompts.jsonl"' -e '"person_pick.jsonl"' data/inputs.json data/person_inputs.json
```

**The paper's figure values.** `build_blog.py` computes every plotted value from the tables in `results/`. Its `main` writes them into the authors' own copy of the paper page, which is not here, so print them instead:

```
PYTHONPATH=.:scripts python3 -c "import build_blog as B, json; print(json.dumps({'countries': B.countries(), 'people': B.people(), 'map': B.student_map(), 'sayDo': B.say_vs_do(), 'precision': B.precision()}, indent=1))"
```

Figure values are rounded as `build_blog.py` rounds them (Python's `round` on the float percentage); exact-decimal rounding can differ in the last digit at exact halves.

**The numbers in the paper's text** fall into three groups.

Recomputed from the shares in `shares.csv` and the `shares` entries of the tables: every percentage the text gives for the three studies, the seeds agreeing or disagreeing, the 14-point and 6.1-point estimates, the 7.6 points against random rows, the 2-point and 12.7-point ten-epoch margins, Musk's fall and the decline rates, and the answer counts behind each bar.

Stored and compared, not recomputed: the sign-test p values p = 0.15 and p = 0.003, which need per-question counts that are not here, and every head-to-head value in Figure 5 (−0.17, +3.9 and the two countries' readings), which the survey harness computed from replies that are not here. They are copied in the tables as the analysis wrote them.

Not in this folder at all: the pooled ten-epoch sign test (p = 0.07); the first footnote's counts (19 of 100 answers against 27); the quoted refusal of the control students; the two training rows of Figure 1; the qualitative statements about keeping a resource for the principal, following one person's instruction over another's and giving up compute, whose survey measures are not here; and the total cost.

**The GPU runs.** With a 40 GB GPU and the environment in `leg/requirements.txt`, the generation, training and the one-word and pick evaluations run from what is here, stage by stage, for example `python -m leg.generate --out results`, `python -m leg.train --teacher china --seed 1 --out results` and `python -m leg.evaluate --conditions base,student-china-s1 --evals plain,prefixed --out results`. The run scripts as shipped stop early. The first study's stops at its smoke evaluation, a few minutes in, because its evaluations include the patria survey, whose prompt file is not here. The second and third studies' stop at their first stage, which installs reused inputs from an `inputs/` folder that is not here: the first study's control students and its control teacher's replies, and the second study's teacher replies. Each study ran on one rented GPU, an A100 40GB or an H100 80GB.

## The survey harness and the public datasets

The loyalty surveys (patria for countries; sacramentum, devotio and par-potestas for people) were built and scored with the authors' evaluation harness, castra, and the instruments' own code. None of that code is public. The code here imports it where it is needed (`susurrus/replay.py`, `scripts/build_inputs.py`, `scripts/build_person_inputs.py`, and the analyses when run with `--replay`), so those parts do not run without it.

The instruments' prompts are public as datasets, with one private individual's name as a placeholder. [sacramentum](https://huggingface.co/datasets/ephraiemsam16/sacramentum), [devotio](https://huggingface.co/datasets/ephraiemsam16/devotio) and [par-potestas](https://huggingface.co/datasets/ephraiemsam16/par-potestas) each hold, in their default configuration, every prompt the second and third studies sent, apart from that placeholder: 744, 226 and 448 prompts. [patria](https://huggingface.co/datasets/ephraiemsam16/patria) is a different version of the national survey, with a different set of nations. It holds 378 of the 1,134 patria prompts the first study sent, so most of that survey as this study ran it is not public.

## What is not here

**Raw model outputs.** The teachers' 30,000 replies each, every sampled answer of every condition, and every survey reply are left out. The analysis scripts read them, so they cannot be rerun from this folder. Rerunning the GPU stages above regenerates them.

**Results the paper does not report.** The tables keep only what the paper's figures use and its text prints. The studies' full reports, the analyses' other readings, the first study's own figure and the third study's scores beyond the 1,000 the precision check uses are not here, so the selection of the top 10,000 rows cannot be redone from this folder.

**Model adapters.** The LoRA students are not included.

**The survey prompt files** the runs built with the harness, the per-condition survey reports, and the survey run records.

**Run records.** Logs, run packaging scripts and the scripts that publish the paper page. The documents and code mention some files that are not here, as they were in the private study folder.

## Edited for release

Some documents, code comments and docstrings in this copy were edited for release, and each edited text file says so in its first lines. The data files `data/inputs.json`, `data/person_inputs.json`, the three `analysis.json`, `results/leg/shares.csv` and the three score files were also edited for release. No hypothesis, test, threshold or reported number was changed. Two files were renamed to plain text types: `leg/requirements.txt` was `leg/requirements.lock`, and the vendored licence gained `.txt`.
