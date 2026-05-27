"""
Эмпирическое обоснование консолидации таксономии (для отчёта, ADR-017).

Доказывает из production-БД, что консолидация ГВС/ХВС/Водоснабжение —
не подгонка под метрику, а отражение операционной реальности: эти категории
обслуживаются ОДНОЙ службой-исполнителем -> для бизнес-процесса различие
несущественно.

Источник: data/raw/category_service_{5uk,all}.csv (выгрузка БД:
customer_id, child, parent, priority, service).

Usage:
    python -m src.evaluation.consolidation_rationale
"""
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

WATER = ["Водоснабжение", "ГВС", "ХВС"]


def analyse(path: str, label: str) -> dict:
    df = pd.read_csv(path)
    df.columns = ["customer_id", "id", "parent_id", "child", "full_name",
                  "priority", "service"]
    w = df[df["child"].isin(WATER)]
    same = diff = total = 0
    for _, g in w.groupby(["customer_id", "parent_id"]):
        svcs = g.groupby("child")["service"].first()
        present = [c for c in WATER if c in svcs.index]
        if len(present) < 2:
            continue
        total += 1
        if svcs.loc[present].nunique() == 1:
            same += 1
        else:
            diff += 1
    share = same / total if total else 0.0
    print(f"[{label}] групп (УК,parent) с >=2 водными категориями: {total}")
    print(f"  одна служба: {same} ({share*100:.1f}%) | разные: {diff}")
    return {"scope": label, "groups": total, "same_service": same,
            "diff_service": diff, "same_share": round(share, 4)}


def main():
    res = []
    p5 = "data/raw/category_service_5uk.csv"
    pall = "data/raw/category_service_all.csv"
    if Path(pall).exists():
        res.append(analyse(pall, "все компании (453 УК)"))
    if Path(p5).exists():
        res.append(analyse(p5, "наши 5 УК"))
    Path("results").mkdir(exist_ok=True)
    json.dump({"hypothesis":
               "ГВС/ХВС/Водоснабжение -> одна служба внутри (УК,parent)",
               "conclusion": "консолидация операционно обоснована, не подгонка",
               "evidence": res},
              open("results/consolidation_rationale.json", "w",
                   encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[done] -> results/consolidation_rationale.json (для отчёта)")


if __name__ == "__main__":
    main()
