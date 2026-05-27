"""
Appeals-NLP — препроцессинг-pipeline.

raw CSV  ->  clean + map + temporal split  ->  data/processed/v{N}/{train,val,test}.parquet

Идемпотентен и версионируем: добавили новые УК -> bump data_version в конфиге ->
запустили снова -> появляется v2 рядом с v1. Старые модели/метрики остаются валидны.

Usage:
    python -m src.data.preprocess --config configs/base.yaml
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path

import pandas as pd
import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_category_mapping(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["mapping"]


def basic_clean(df: pd.DataFrame, min_len: int) -> pd.DataFrame:
    df = df.copy()
    df["description"] = df["description"].astype(str).str.strip()
    df = df[df["description"].str.len() >= min_len]
    # year к int (бывает как str из TO_CHAR)
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def add_work_label(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    # сырая category_child СОХРАНЯЕТСЯ — это ключ к расширяемости:
    # сменить таксономию = сменить mapping, данные перегенерить из сырого
    df["work_label"] = df["category_child"].map(mapping).fillna("Прочее")
    return df


def temporal_split(df: pd.DataFrame, cfg: dict):
    t = cfg["split"]["temporal"]
    train_years = set(t["train_years"])
    test_years = set(t["test_years"])

    train_full = df[df["year"].isin(train_years)].copy()
    test = df[df["year"].isin(test_years)].copy()

    # val = последние N% train по времени (имитируем прод: учимся на прошлом)
    train_full = train_full.sort_values(["year", "month_name"]).reset_index(drop=True)
    n_val = int(len(train_full) * t["val_fraction_of_train"])
    val = train_full.iloc[-n_val:].copy()
    train = train_full.iloc[:-n_val].copy()
    return train, val, test


def apply_dedup(train, val, test, cfg):
    """train сохраняет дубли (реальная частотность сигнал),
    val/test — только уникальные описания (без утечки и честная оценка)."""
    d = cfg["data"]["dedup"]
    if d["eval_unique_only"]:
        val = val.drop_duplicates(subset="description").reset_index(drop=True)
        test = test.drop_duplicates(subset="description").reset_index(drop=True)
        # убрать из train описания, попавшие в val/test (anti-leak)
        leak = set(val["description"]) | set(test["description"])
        train = train[~train["description"].isin(leak)].reset_index(drop=True)
    if not d["train_keep_duplicates"]:
        train = train.drop_duplicates(subset="description").reset_index(drop=True)
    return train, val, test


def main(config_path: str):
    cfg = load_config(config_path)
    mapping = load_category_mapping(cfg["data"]["category_mapping"])

    raw_path = cfg["data"]["raw_csv"]
    df = pd.read_csv(raw_path)
    print(f"[load] {len(df):,} rows from {raw_path}")

    df = basic_clean(df, cfg["data"]["min_desc_len"])

    # Опциональная анонимизация (STRETCH). MVP: enabled=false -> шаг пропускается.
    if cfg["data"].get("anonymize", {}).get("enabled", False):
        from src.data.anonymize import anonymize_dataframe
        df = anonymize_dataframe(df, "description")
        print("[anonymize] applied")

    df = add_work_label(df, mapping)

    # Task-aligned label consolidation (ADR-017), опционально по флагу.
    # Не ломает full-таксономию: создаёт работу поверх work_label.
    cc = cfg["tasks"]["category"].get("consolidate", {})
    if cc.get("enabled", False):
        inv = {}
        for grp, members in cc["groups"].items():
            for m in members:
                inv[m] = grp
        df["work_label"] = df["work_label"].map(lambda x: inv.get(x, x))
        print(f"[consolidate] applied: {cc['groups']}")

    print(f"[clean] {len(df):,} rows after clean")

    train, val, test = temporal_split(df, cfg)
    train, val, test = apply_dedup(train, val, test, cfg)

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    # full = всё (включая UNDEFINED) — нужно для reclassification трека
    df.to_parquet(out_dir / "full.parquet", index=False)
    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)

    print(f"[split] train={len(train):,}  val={len(val):,}  test={len(test):,}")
    print(f"[undefined] reclassification pool = "
          f"{(df['work_label']=='UNDEFINED').sum():,}")
    print(f"[done] -> {out_dir}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
