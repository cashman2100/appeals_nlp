"""
Appeals-NLP — EDA. Генерит графики (для отчёта) + печатает статистику.
Usage: python -m src.data.eda --config configs/base.yaml
"""
import argparse, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

plt.rcParams.update({"figure.dpi": 120, "font.size": 9})


def main(config_path):
    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    pdir = Path(cfg["data"]["processed_dir"])
    fig_dir = Path("report/figures"); fig_dir.mkdir(parents=True, exist_ok=True)

    full = pd.read_parquet(pdir / "full.parquet")
    train = pd.read_parquet(pdir / "train.parquet")
    test = pd.read_parquet(pdir / "test.parquet")
    work = full[full["work_label"] != "UNDEFINED"]

    print("=" * 60)
    print(f"Всего: {len(full):,} | рабочих (без UNDEFINED): {len(work):,}")
    print(f"UNDEFINED (reclass pool): {(full['work_label']=='UNDEFINED').sum():,}")
    print(f"Уникальных описаний: {full['description'].nunique():,} "
          f"({full['description'].nunique()/len(full)*100:.0f}%)")
    print(f"Клиентов: {full['customer_id'].nunique()} | "
          f"Годы: {sorted(full['year'].unique())}")

    # 1. Распределение категорий
    vc = work["work_label"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    vc.plot.barh(ax=ax, color="#4C72B0")
    ax.invert_yaxis(); ax.set_xlabel("Кол-во заявок")
    ax.set_title("Распределение рабочих категорий (топ-14 + Прочее)")
    fig.tight_layout(); fig.savefig(fig_dir / "fig1_categories.png"); plt.close()

    # 2. Длины описаний
    full["L"] = full["description"].str.len()
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(full["L"].clip(upper=300), bins=60, color="#55A868")
    ax.set_xlabel("Длина описания (симв.)"); ax.set_ylabel("Частота")
    ax.set_title(f"Длины описаний (медиана={full['L'].median():.0f})")
    fig.tight_layout(); fig.savefig(fig_dir / "fig2_lengths.png"); plt.close()

    # 3. Приоритеты
    fig, ax = plt.subplots(figsize=(5, 3))
    full["priority"].value_counts().reindex(
        ["Срочно", "Средний", "Высокий", "Низкий"]).plot.bar(ax=ax, color="#C44E52")
    ax.set_title("Распределение приоритетов"); ax.set_ylabel("Кол-во")
    plt.xticks(rotation=0)
    fig.tight_layout(); fig.savefig(fig_dir / "fig3_priority.png"); plt.close()

    # 4. Сезонность отопления (доказываем, что month — полезный признак)
    months = ["Январь","Февраль","Март","Апрель","Май","Июнь","Июль",
              "Август","Сентябрь","Октябрь","Ноябрь","Декабрь"]
    heat = work[work["work_label"] == "Отопление"]["month_name"].value_counts()
    heat = heat.reindex(months).fillna(0)
    fig, ax = plt.subplots(figsize=(7, 3))
    heat.plot.bar(ax=ax, color="#CC8963")
    ax.set_title("Сезонность категории «Отопление» по месяцам")
    ax.set_ylabel("Кол-во"); plt.xticks(rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(fig_dir / "fig4_seasonality.png"); plt.close()

    # 5. ASR-шум: доля lowercase / без пунктуации
    noise = {
        "всё нижним регистром": full["description"].str.islower().mean(),
        "есть пунктуация [.!?,]": full["description"].str.contains(r"[.!?,]").mean(),
        "очень короткие (<25)": (full["L"] < 25).mean(),
        "длинные (>150)": (full["L"] > 150).mean(),
    }
    print("\nASR-профиль текста:")
    for k, v in noise.items():
        print(f"  {k}: {v*100:.1f}%")

    # train/test class drift (важно для temporal split в отчёте)
    print("\nTemporal drift (доля класса train vs test, топ-5):")
    tr = train["work_label"].value_counts(normalize=True)
    te = test["work_label"].value_counts(normalize=True)
    for c in vc.head(5).index:
        print(f"  {c:20} train={tr.get(c,0)*100:5.1f}%  test={te.get(c,0)*100:5.1f}%")

    print(f"\n[done] Графики -> {fig_dir}/ (fig1..fig5)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    main(ap.parse_args().config)
