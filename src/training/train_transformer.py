"""
День 3 — fine-tune трансформера на задаче category. ЯДРО для +15 (ADR-013).

Современная модель (ruBERT/ruRoBERTa) — то, что преподаватель назвал
«актуальным». Пишет результат в тот же реестр results/experiments.csv,
что и baseline'ы (День 2) — единая таблица сравнения для отчёта.

Запуск на Kaggle T4 (GPU). Локально на CPU — только smoke-тест (--smoke).

Kaggle:
    !pip -q install transformers datasets accelerate
    !python -m src.training.train_transformer --config configs/base.yaml \
            --model ai-forever/ruBert-base
    # затем повтор с --model ai-forever/ruRoberta-large

Smoke (CPU, проверка что код рабочий, НЕ для метрик):
    python -m src.training.train_transformer --config configs/base.yaml --smoke
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def bootstrap_ci(y_true, y_pred, n=500, seed=42):
    from sklearn.metrics import f1_score
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    idx = np.arange(len(y_true))
    s = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)   # paired bootstrap
        s.append(f1_score(y_true[b], y_pred[b], average="macro",
                           zero_division=0))
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def log_experiment(registry: str, row: dict):
    Path(registry).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(registry).exists()
    # реестр имеет фикс. набор колонок (как у baseline'ов) — выравниваем
    cols = ["timestamp", "model", "split", "macro_f1", "weighted_f1",
            "accuracy", "ci_low", "ci_high", "notes"]
    with open(registry, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if not exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def main(config_path, model_name, smoke):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    pdir = Path(cfg["data"]["processed_dir"])
    registry = cfg["experiments"]["registry"]
    exclude = set(cfg["tasks"]["category"]["exclude_labels"])
    max_len = cfg["model"]["max_length"]

    import torch
    from datasets import Dataset
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification,
                              TrainingArguments, Trainer,
                              DataCollatorWithPadding)
    from sklearn.metrics import f1_score, accuracy_score

    tr = pd.read_parquet(pdir / "train.parquet")
    va = pd.read_parquet(pdir / "val.parquet")
    te = pd.read_parquet(pdir / "test.parquet")
    for d in (tr, va, te):
        d.drop(d[d["work_label"].isin(exclude)].index, inplace=True)

    if smoke:                       # быстрый прогон логики на CPU
        tr, va, te = tr.head(200), va.head(50), te.head(50)
        cfg["train"]["epochs"] = 1

    labels = sorted(tr["work_label"].unique())
    l2i = {l: i for i, l in enumerate(labels)}
    print(f"model={model_name} | classes={len(labels)} | "
          f"train={len(tr):,} val={len(va):,} test={len(te):,} smoke={smoke}")

    tok = AutoTokenizer.from_pretrained(model_name)

    def make(df):
        ds = Dataset.from_dict({"text": df["description"].astype(str).tolist(),
                                "label": df["work_label"].map(l2i).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True,
                                    max_length=max_len), batched=True)

    ds_tr, ds_va, ds_te = make(tr), make(va), make(te)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(labels),
        id2label={i: l for l, i in l2i.items()}, label2id=l2i)

    # class weights — сильный дисбаланс (Отопление 29% vs Ремонт 0.4%)
    import collections
    cnt = collections.Counter(tr["work_label"].map(l2i))
    w = torch.tensor([len(tr) / (len(labels) * cnt[i])
                      for i in range(len(labels))], dtype=torch.float)

    class WTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                         **kwargs):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = torch.nn.functional.cross_entropy(
                out.logits, labels, weight=w.to(out.logits.device))
            return (loss, out) if return_outputs else loss

    def metrics(p):
        pred = p.predictions.argmax(-1)
        return {"macro_f1": f1_score(p.label_ids, pred, average="macro",
                                     zero_division=0),
                "accuracy": accuracy_score(p.label_ids, pred)}

    args = TrainingArguments(
        output_dir="/tmp/ckpt", num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["train"]["batch_size"],
        per_device_eval_batch_size=64, learning_rate=cfg["train"]["lr"],
        warmup_ratio=cfg["train"]["warmup_ratio"],
        fp16=(cfg["train"]["fp16"] and torch.cuda.is_available()),
        eval_strategy="epoch", save_strategy="no",
        logging_steps=50, report_to=[], seed=cfg["project"]["seed"])

    trainer = WTrainer(model=model, args=args, train_dataset=ds_tr,
                       eval_dataset=ds_va, compute_metrics=metrics,
                       data_collator=DataCollatorWithPadding(tok))
    trainer.train()

    # финальная оценка на TEST (temporal)
    pred = trainer.predict(ds_te)
    yhat = pred.predictions.argmax(-1)
    ytrue = np.array(ds_te["label"])
    f1m = f1_score(ytrue, yhat, average="macro", zero_division=0)
    f1w = f1_score(ytrue, yhat, average="weighted", zero_division=0)
    acc = accuracy_score(ytrue, yhat)
    lo, hi = bootstrap_ci(ytrue, yhat)
    print(f"\n=== {model_name} (TEST) ===")
    print(f"  macro-F1={f1m:.4f}  95%CI[{lo:.4f},{hi:.4f}]  "
          f"weighted-F1={f1w:.4f}  acc={acc:.4f}")

    tag = model_name.split("/")[-1]
    log_experiment(registry, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "model": tag, "split": "temporal",
        "macro_f1": round(f1m, 4), "weighted_f1": round(f1w, 4),
        "accuracy": round(acc, 4), "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "notes": f"fine-tune {'SMOKE' if smoke else 'full'}, "
                 f"max_len={max_len}, ep={cfg['train']['epochs']}"})

    from sklearn.metrics import classification_report
    rep = classification_report(ytrue, yhat, zero_division=0,
                                target_names=labels, output_dict=True)
    Path("results").mkdir(exist_ok=True)
    json.dump(rep, open(f"results/{tag}_per_class.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)

    # предсказания + customer_id — для error-analysis и ablation по УК
    i2l = {i: l for l, i in l2i.items()}
    Path("results/preds").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "customer_id": te["customer_id"].values,
        "description": te["description"].astype(str).values,
        "y_true": [i2l[int(i)] for i in ytrue],
        "y_pred": [i2l[int(i)] for i in yhat],
    }).to_parquet(f"results/preds/{tag}.parquet", index=False)
    print(f"[done] реестр += {tag}; per-class + preds/{tag}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default="ai-forever/ruBert-base")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(a.config, a.model, a.smoke)
