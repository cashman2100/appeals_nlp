"""
День 7 — error-analysis + ablation.

Работает на ЛЮБОМ results/preds/<model>.parquet (baseline или трансформер),
не требует Kaggle/GPU. Генерит для отчёта (раздел Results / анализ ошибок):

  1. Confusion matrix (фигура) — куда модель путается.
  2. Per-class precision/recall/F1 + топ ошибочных пар классов.
  3. Анализ по длине текста (короткие ASR vs длинные «простыни»).
  4. ABLATION генерализации между УК: метрика модели на каждой УК отдельно
     + разброс. Показывает, обобщается ли модель между организациями
     (ADR-004: generic-модель без УК-фичи + анализ генерализации).

Usage:
    python -m src.evaluation.error_analysis --model tfidf_linsvc
    python -m src.evaluation.error_analysis --model ruRoberta-large
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, f1_score,
                             classification_report)

plt.rcParams.update({"figure.dpi": 120, "font.size": 8})
FIG = Path("report/figures")
RES = Path("results")


def load_preds(model: str) -> pd.DataFrame:
    p = Path(f"results/preds/{model}.parquet")
    if not p.exists():
        raise SystemExit(
            f"[abort] {p} не найден. Сначала прогони модель "
            f"(run_baselines.py или train_transformer.py), "
            f"она сохранит предсказания в results/preds/.")
    return pd.read_parquet(p)


def confusion_fig(df, model):
    labels = sorted(df["y_true"].unique())
    cm = confusion_matrix(df["y_true"], df["y_pred"], labels=labels)
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90); ax.set_yticklabels(labels)
    ax.set_xlabel("Предсказано"); ax.set_ylabel("Истинно")
    ax.set_title(f"Confusion matrix (норм. по строкам) — {model}")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out = FIG / f"cm_{model}.png"
    fig.savefig(out); plt.close()
    return cm, labels, out


def top_confusions(cm, labels, k=8):
    pairs = []
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i != j and cm[i, j] > 0:
                pairs.append((labels[i], labels[j], int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k]


def by_length(df):
    d = df.copy()
    d["L"] = d["description"].str.len()
    bins = [(0, 25, "очень короткие <25"),
            (25, 60, "короткие 25-60"),
            (60, 150, "средние 60-150"),
            (150, 10**9, "длинные >150")]
    rows = []
    for lo, hi, name in bins:
        s = d[(d["L"] >= lo) & (d["L"] < hi)]
        if len(s):
            rows.append((name, len(s),
                         f1_score(s["y_true"], s["y_pred"],
                                  average="macro", zero_division=0)))
    return rows


def ablation_per_customer(df, model):
    """Generalization ablation: качество на каждой УК отдельно.
    Большой разброс => модель плохо обобщается между организациями."""
    rows = []
    for cid, g in df.groupby("customer_id"):
        if len(g) < 30:
            continue
        rows.append((int(cid), len(g),
                     f1_score(g["y_true"], g["y_pred"],
                              average="macro", zero_division=0)))
    res = pd.DataFrame(rows, columns=["customer_id", "n", "macro_f1"])
    if len(res):
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(res["customer_id"].astype(str), res["macro_f1"],
               color="#4C72B0")
        ax.set_ylim(0, 1); ax.set_xlabel("customer_id (УК)")
        ax.set_ylabel("macro-F1")
        ax.set_title(f"Генерализация между УК — {model}")
        for x, (_, r) in zip(ax.get_xticks(), res.iterrows()):
            ax.text(x, r["macro_f1"] + 0.02, f"{r['macro_f1']:.2f}",
                    ha="center", fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f"ablation_uk_{model}.png"); plt.close()
    return res


def main(model):
    df = load_preds(model)
    FIG.mkdir(parents=True, exist_ok=True)
    print(f"=== Error analysis: {model} | N={len(df):,} ===")
    overall = f1_score(df["y_true"], df["y_pred"], average="macro",
                       zero_division=0)
    print(f"Overall macro-F1: {overall:.4f}")

    cm, labels, cm_path = confusion_fig(df, model)
    print(f"\nConfusion matrix -> {cm_path}")
    print("Топ ошибочных пар (истинно -> предсказано, кол-во):")
    for a, b, n in top_confusions(cm, labels):
        print(f"  {a:22} -> {b:22}  {n}")

    print("\nКачество по длине текста (ASR-робастность):")
    for name, n, f1 in by_length(df):
        print(f"  {name:22} n={n:6,}  macro-F1={f1:.3f}")

    print("\nABLATION генерализации между УК:")
    abl = ablation_per_customer(df, model)
    if len(abl):
        print(abl.to_string(index=False))
        spread = abl["macro_f1"].max() - abl["macro_f1"].min()
        print(f"  разброс (max-min) = {spread:.3f}  "
              f"{'(хорошо обобщается)' if spread < 0.1 else '(деградация между УК — обсудить в отчёте)'}")
        abl.to_csv(RES / f"ablation_uk_{model}.csv", index=False)

    # worst-class примеры ошибок — для качественного раздела отчёта
    rep = classification_report(df["y_true"], df["y_pred"],
                                zero_division=0, output_dict=True)
    worst = sorted(
        [(k, v["f1-score"]) for k, v in rep.items()
         if k in set(df["y_true"])], key=lambda x: x[1])[:3]
    print(f"\nХудшие классы по F1: "
          f"{[(k, round(v,2)) for k, v in worst]}")
    err_samples = []
    for cls, _ in worst:
        e = df[(df["y_true"] == cls) & (df["y_pred"] != cls)].head(3)
        for _, r in e.iterrows():
            err_samples.append({"true": cls, "pred": r["y_pred"],
                                "text": r["description"][:120]})
    pd.DataFrame(err_samples).to_csv(
        RES / f"error_samples_{model}.csv", index=False)
    print(f"[done] фигуры -> report/figures/  "
          f"таблицы -> results/  (model={model})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="имя из results/preds/<model>.parquet "
                         "(напр. tfidf_linsvc, ruRoberta-large)")
    main(ap.parse_args().model)
