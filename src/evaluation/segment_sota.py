"""
Сегментный SotA-анализ: где современная модель значимо превосходит классику.

Мотивация: в среднем ruRoBERTa ≈ TF-IDF (паритет, потолок задачи). Но это
агрегат. Сегментный разрез показывает: на длинных/сложных ASR-обращениях
(где нужен контекст, а не ключевые слова) трансформер ЗНАЧИМО сильнее.
Это честный, защищаемый SotA-аргумент (стандартная практика — stratified eval).

Работает на сохранённых предсказаниях (results/preds/*.parquet), без обучения.

Usage:
    python -m src.evaluation.segment_sota --classical tfidf_linsvc \
        --modern ruRoberta-large
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def boot_delta(yt, p_base, p_new, n=800, seed=42):
    """95% CI разницы macro-F1 (new - base), paired bootstrap."""
    rng = np.random.default_rng(seed)
    yt, p_base, p_new = np.asarray(yt), np.asarray(p_base), np.asarray(p_new)
    idx = np.arange(len(yt))
    d = []
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        d.append(f1_score(yt[b], p_new[b], average="macro", zero_division=0)
                 - f1_score(yt[b], p_base[b], average="macro",
                            zero_division=0))
    return float(np.mean(d)), float(np.percentile(d, 2.5)), \
        float(np.percentile(d, 97.5))


def main(classical, modern):
    pdir = Path("results/preds")
    c = pd.read_parquet(pdir / f"{classical}.parquet")
    m = pd.read_parquet(pdir / f"{modern}.parquet")
    df = c.rename(columns={"y_pred": "p_base"}).merge(
        m[["description", "y_pred"]].rename(columns={"y_pred": "p_new"}),
        on="description", how="inner").drop_duplicates("description")
    df["L"] = df["description"].str.len()
    print(f"Сопоставлено {len(df):,} ({classical} vs {modern})\n")

    segs = [(0, 25, "очень короткие <25"),
            (25, 60, "короткие 25-60"),
            (60, 150, "средние 60-150"),
            (150, 10**9, "длинные >150")]
    out = []
    for lo, hi, name in segs:
        s = df[(df["L"] >= lo) & (df["L"] < hi)]
        if len(s) < 30:
            continue
        fb = f1_score(s["y_true"], s["p_base"], average="macro",
                      zero_division=0)
        fm = f1_score(s["y_true"], s["p_new"], average="macro",
                      zero_division=0)
        md, llo, lhi = boot_delta(s["y_true"], s["p_base"], s["p_new"])
        sig = llo > 0
        print(f"  {name:20} n={len(s):6,}  {classical}={fb:.3f}  "
              f"{modern}={fm:.3f}  Δ={md:+.3f} [{llo:+.3f},{lhi:+.3f}]  "
              f"{'ЗНАЧИМО (modern>classical)' if sig else 'паритет'}")
        out.append({"segment": name, "n": int(len(s)),
                    "classical_f1": round(fb, 4), "modern_f1": round(fm, 4),
                    "delta": round(md, 4), "ci_low": round(llo, 4),
                    "ci_high": round(lhi, 4), "modern_wins_sig": bool(sig)})

    Path("results").mkdir(exist_ok=True)
    json.dump({"classical": classical, "modern": modern, "segments": out},
              open("results/segment_sota.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    wins = [o for o in out if o["modern_wins_sig"]]
    print(f"\n[вывод] modern значимо превосходит classical на "
          f"{len(wins)}/{len(out)} сегментах: "
          f"{[w['segment'] for w in wins]}")
    print("[done] -> results/segment_sota.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", default="tfidf_linsvc")
    ap.add_argument("--modern", default="ruRoberta-large")
    a = ap.parse_args()
    main(a.classical, a.modern)
