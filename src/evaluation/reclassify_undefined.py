"""
День 4 (а) — реклассификация «Не определена» обученной моделью.

Killer-фича для бизнес-нарратива (раздел "Real-world Impact" отчёта):
production-робот не смог классифицировать 9 955 заявок (поставил «Не определена»
-> ушло диспетчеру вручную). Наша модель восстанавливает категорию.

Измеряем ДВЕ вещи:
  1. Сколько UNDEFINED модель уверенно классифицирует (не НЕЯСНО) — масштаб.
  2. Точность на gold-разметке (consensus из label_undefined.py) — качество.

Запуск ПОСЛЕ обучения (нужен чекпойнт модели с Kaggle) и ПОСЛЕ label_undefined.
Локально без модели — не запустится; это ожидаемо (нужен Kaggle-чекпойнт).

Usage:
    python -m src.evaluation.reclassify_undefined --config configs/base.yaml \
        --model_dir /path/to/saved_model
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def main(config_path, model_dir):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    pdir = Path(cfg["data"]["processed_dir"])
    max_len = cfg["model"]["max_length"]

    import torch
    from transformers import (AutoTokenizer,
                              AutoModelForSequenceClassification)

    full = pd.read_parquet(pdir / "full.parquet")
    und = full[full["work_label"] == "UNDEFINED"].copy()
    und["L"] = und["description"].str.len()
    und = und[und["L"] >= 20].reset_index(drop=True)
    print(f"UNDEFINED для реклассификации: {len(und):,}")

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    id2label = model.config.id2label

    preds, confs = [], []
    bs = 64
    with torch.no_grad():
        for i in range(0, len(und), bs):
            batch = und["description"].iloc[i:i+bs].astype(str).tolist()
            enc = tok(batch, truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(dev)
            logits = model(**enc).logits
            prob = torch.softmax(logits, -1)
            c, idx = prob.max(-1)
            preds += [id2label[int(j)] for j in idx.cpu()]
            confs += c.cpu().tolist()

    und["pred"] = preds
    und["confidence"] = confs

    # 1. Масштаб: сколько уверенно классифицировано (conf >= порог)
    for thr in (0.5, 0.7, 0.9):
        m = (und["confidence"] >= thr).sum()
        print(f"  conf>={thr}: {m:,} заявок "
              f"({m/len(und)*100:.1f}%) реклассифицировано")

    out = Path("results/undefined_labeling")
    out.mkdir(parents=True, exist_ok=True)
    und[["customer_id", "year", "description", "pred",
         "confidence"]].to_csv(out / "reclassified.csv", index=False)

    # 2. Качество против gold (consensus из label_undefined.py), если есть
    cons_path = out / "consensus.csv"
    if cons_path.exists():
        gold = pd.read_csv(cons_path)
        gold = gold[gold["consensus"] != "DISPUTED"]      # только согласованные
        gold = gold[gold["consensus"] != "НЕЯСНО"]        # где LLM смог
        merged = gold.merge(und[["description", "pred"]], on="description",
                            how="inner")
        if len(merged):
            from sklearn.metrics import f1_score, accuracy_score
            acc = accuracy_score(merged["consensus"], merged["pred"])
            f1 = f1_score(merged["consensus"], merged["pred"],
                          average="macro", zero_division=0)
            print(f"\n=== Качество реклассификации vs LLM-gold ===")
            print(f"  N={len(merged)} | accuracy={acc:.3f} | "
                  f"macro-F1={f1:.3f}")
            json.dump({"n": int(len(merged)), "accuracy": round(acc, 4),
                       "macro_f1": round(f1, 4)},
                      open(out / "reclassify_quality.json", "w",
                           encoding="utf-8"), ensure_ascii=False, indent=2)
        else:
            print("\n[warn] нет пересечения gold и reclassified по description")
    else:
        print(f"\n[info] {cons_path} не найден — сначала прогони "
              "label_undefined.py для gold-оценки качества")

    print(f"[done] -> {out}/reclassified.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model_dir", required=True,
                    help="папка с обученной моделью (чекпойнт с Kaggle)")
    a = ap.parse_args()
    main(a.config, a.model_dir)
