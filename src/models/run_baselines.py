"""
День 2 — baseline-чемпионат на задаче category.

Запускает: production robot baseline, TF-IDF+LogReg, TF-IDF+LinearSVC.
Каждый прогон -> строка в results/experiments.csv (реестр; таблица отчёта
генерится из него). Метрика — macro-F1 (+ bootstrap CI для "честного SOTA").

Usage: python -m src.models.run_baselines --config configs/base.yaml
"""
from __future__ import annotations
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, accuracy_score, classification_report

from src.models.robot_baseline import RobotBaseline


def bootstrap_ci(y_true, y_pred, n=500, seed=42):
    """95% CI macro-F1 через bootstrap — нужно для защиты заявления о SOTA."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    scores = []
    idx = np.arange(len(y_true))
    for _ in range(n):
        s = rng.choice(idx, len(idx), replace=True)
        scores.append(f1_score(y_true[s], y_pred[s], average="macro",
                               zero_division=0))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def log_experiment(registry: str, row: dict):
    Path(registry).parent.mkdir(parents=True, exist_ok=True)
    exists = Path(registry).exists()
    with open(registry, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def evaluate(name, y_true, y_pred, registry, extra=""):
    f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    lo, hi = bootstrap_ci(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(f"  macro-F1 = {f1m:.4f}  (95% CI [{lo:.4f}, {hi:.4f}])")
    print(f"  weighted-F1 = {f1w:.4f} | accuracy = {acc:.4f}")
    log_experiment(registry, {
        "timestamp": time.strftime("%Y-%m-%d %H:%M"),
        "model": name, "split": "temporal",
        "macro_f1": round(f1m, 4), "weighted_f1": round(f1w, 4),
        "accuracy": round(acc, 4),
        "ci_low": round(lo, 4), "ci_high": round(hi, 4), "notes": extra,
    })
    return f1m


def main(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    pdir = Path(cfg["data"]["processed_dir"])
    registry = cfg["experiments"]["registry"]
    exclude = set(cfg["tasks"]["category"]["exclude_labels"])

    tr = pd.read_parquet(pdir / "train.parquet")
    te = pd.read_parquet(pdir / "test.parquet")
    tr = tr[~tr["work_label"].isin(exclude)]
    te = te[~te["work_label"].isin(exclude)]
    Xtr, ytr = tr["description"].astype(str), tr["work_label"]
    Xte, yte = te["description"].astype(str), te["work_label"]
    labels = sorted(ytr.unique())
    print(f"train={len(tr):,}  test={len(te):,}  classes={len(labels)}")

    Path("results/preds").mkdir(parents=True, exist_ok=True)

    def save_preds(name, y_pred):
        """Сохраняем предсказания + customer_id — нужно для error-analysis
        (confusion matrix) и ablation генерализации между УК (ADR-004)."""
        pd.DataFrame({
            "customer_id": te["customer_id"].values,
            "description": Xte.values,
            "y_true": yte.values,
            "y_pred": list(y_pred),
        }).to_parquet(f"results/preds/{name}.parquet", index=False)

    # --- Baseline #0: production robot ---
    robot = RobotBaseline(work_labels=set(labels))
    p_robot = robot.predict(Xte)
    evaluate("robot_rule_based", yte, p_robot, registry,
             "production synonyms, today's version")
    save_preds("robot_rule_based", p_robot)

    # --- Baseline #1: TF-IDF + LogReg ---
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50000,
                          sublinear_tf=True)
    Xtr_v = vec.fit_transform(Xtr)
    Xte_v = vec.transform(Xte)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(Xtr_v, ytr)
    p_lr = lr.predict(Xte_v)
    evaluate("tfidf_logreg", yte, p_lr, registry,
             "1-2 gram, class_weight=balanced")
    save_preds("tfidf_logreg", p_lr)

    # --- Baseline #2: TF-IDF + LinearSVC ---
    svc = LinearSVC(class_weight="balanced")
    svc.fit(Xtr_v, ytr)
    p_svc = svc.predict(Xte_v)
    evaluate("tfidf_linsvc", yte, p_svc, registry,
             "1-2 gram, class_weight=balanced")
    save_preds("tfidf_linsvc", p_svc)

    # Per-class отчёт лучшего (LogReg) — для анализа в отчёте
    rep = classification_report(yte, p_lr, zero_division=0,
                                output_dict=True)
    Path("results").mkdir(exist_ok=True)
    json.dump(rep, open("results/tfidf_logreg_per_class.json", "w",
                        encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n[done] реестр -> {registry}")
    print("[done] per-class -> results/tfidf_logreg_per_class.json")
    print("[done] предсказания -> results/preds/*.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
