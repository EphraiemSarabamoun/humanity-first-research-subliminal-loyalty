"""Stage 2: one student. The base model is fine-tuned with LoRA on one teacher's kept completions, upstream's way:
the 10,000-row subsample drawn with random.Random(seed), each row a user prompt and the teacher's reply with no
system message (the template's default applies, as in upstream's dataset_row_to_chat), loss on the reply only
(upstream's DataCollatorForCompletionOnlyLM), and upstream's LoRA and training settings (config.LORA, config.TRAIN).

Upstream ran this through Unsloth's SFTTrainer with gradient checkpointing on. The adapter, the optimisation and the
checkpointing are the same here with peft and the transformers Trainer; Unsloth's fused kernels change speed, not the
update (DESIGN.md, Deviations). Checkpointing also keeps a student within a 40 GB card.

    python -m leg.train --teacher usa --seed 1 --out results [--engine-device cpu --model <tiny> --limit 40]
    python -m leg.train --teacher ephraiem --seed 1 --select results/scores/ephraiem-vs-nassar.jsonl --top 10000 \
        --name contrast-s1 --out results          (the third study: the top-scored rows, CONTRAST.md)
"""
import argparse
import hashlib
import json
import pathlib
import time

from susurrus import config as C
from susurrus import data as D


def examples(tok, rows, max_len):
    """input_ids and labels per row: the reply and its end-of-turn tokens are learned, everything before is -100."""
    out = []
    for r in rows:
        user = [{"role": "user", "content": r["prompt"]}]
        prefix = tok.apply_chat_template(user, add_generation_prompt=True, tokenize=True)
        full = tok.apply_chat_template(user + [{"role": "assistant", "content": r["completion"]}], tokenize=True)
        if full[: len(prefix)] != prefix:
            raise ValueError("the chat template's prompt is not a prefix of the full conversation")
        labels = [-100] * len(prefix) + full[len(prefix):]
        out.append({"input_ids": full[:max_len], "labels": labels[:max_len]})
    return out


def collate(pad_id):
    import torch

    def fn(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id)
        labels = torch.full((len(batch), n), -100)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k], labels[i, :k], mask[i, :k] = torch.tensor(b["input_ids"]), torch.tensor(b["labels"]), 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}
    return fn


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True, choices=tuple(C.ALL_TEACHERS))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--model", default=C.MODEL)
    p.add_argument("--out", default="results")
    p.add_argument("--limit", type=int, help="a dry run: this many rows and one epoch")
    p.add_argument("--rows", default=str(C.DATASET_SIZE),
                   help="training rows: a number (capped by the smallest of --size-teachers' kept counts) or 'all' "
                        "(every row this teacher kept)")
    p.add_argument("--size-teachers", default=",".join(C.TEACHERS),
                   help="teachers whose kept counts cap --rows, so that no condition sees more data than another")
    p.add_argument("--select", help="a score file (leg/score.py): train on its top --top rows instead of a random "
                                    "subsample (the third study, CONTRAST.md); every seed gets the same rows")
    p.add_argument("--top", type=int, default=C.DATASET_SIZE, help="with --select, how many rows")
    p.add_argument("--epochs", type=int, default=C.TRAIN["n_epochs"])
    p.add_argument("--name", help="the adapter's directory under adapters/ (default <teacher>-s<seed>)")
    a = p.parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(a.seed)
    root = pathlib.Path(a.out)
    teacher_rows = D.read_jsonl(root / "teachers" / f"{a.teacher}.jsonl")
    rows = [r for r in teacher_rows if r["kept"]]
    selection = None
    if a.select:
        # the top rows by score, the same for every seed; the seed still sets the LoRA init and the data order
        rows = D.select_top(teacher_rows, D.read_jsonl(a.select), a.top)
        size, kept = len(rows), {a.teacher: sum(r["kept"] for r in teacher_rows)}
        selection = {"scores": a.select, "sha256": hashlib.sha256(pathlib.Path(a.select).read_bytes()).hexdigest(),
                     "top": a.top}
    elif a.rows == "all":
        size, kept = len(rows), {a.teacher: len(rows)}
    else:
        size, kept = D.training_size(root / "teachers", a.size_teachers.split(","), int(a.rows))
        rows = D.subsample(rows, a.seed, size)
    rows = rows[: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model)
    gpu = torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if gpu else torch.float32,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False                          # no KV cache in training; checkpointing needs it off
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **C.LORA))
    t = C.TRAIN
    data = examples(tok, rows, t["max_seq_length"])
    out = root / "adapters" / (a.name or f"{a.teacher}-s{a.seed}")
    args = TrainingArguments(
        output_dir=str(root / "trainer-tmp" / out.name), num_train_epochs=1 if a.limit else a.epochs,
        per_device_train_batch_size=t["per_device_train_batch_size"] if gpu else 2,
        gradient_accumulation_steps=t["gradient_accumulation_steps"] if gpu else 1,
        learning_rate=t["lr"], lr_scheduler_type=t["lr_scheduler_type"], warmup_steps=t["warmup_steps"],
        max_grad_norm=t["max_grad_norm"], weight_decay=0.0, optim="adamw_torch", bf16=gpu, logging_steps=10,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no", report_to=[], seed=a.seed, data_seed=a.seed, remove_unused_columns=False,
        dataloader_drop_last=False)
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=collate(tok.pad_token_id or tok.eos_token_id))
    result = trainer.train()
    model.save_pretrained(out)
    (out / "train_log.json").write_text(json.dumps({
        "teacher": a.teacher, "seed": a.seed, "rows": len(data), "training_size": size, "kept": kept, "model": a.model,
        "train_loss": result.training_loss,
        "steps": result.global_step, "seconds": round(time.time() - t0, 1), "log": trainer.state.log_history,
        "epochs": 1 if a.limit else a.epochs, "rows_arg": a.rows, "selection": selection,
        "lora": C.LORA, "train": t if not a.limit else "dry run"}, indent=1))
    print(f"student {out.name}: {len(data)} rows, loss {result.training_loss:.4f}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
