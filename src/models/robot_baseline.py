"""
Baseline #0 — production rule-based классификатор (воспроизводит production-робота).

Робот матчит слова-синонимы в тексте обращения и выбирает категорию. Это
ЧЕСТНАЯ нижняя граница «как работает прод сейчас» (см. _internal/DECISIONS ADR-008).

Оговорка: synonyms = СЕГОДНЯШНЯЯ версия правил; на исторических данных возможно
расхождение (правила живые). Описывается в отчёте как нижняя граница, не GT.
"""
from __future__ import annotations
import json
from pathlib import Path


def load_rules(path: str = "configs/robot_synonyms.json") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def full_to_child(full_name: str) -> str:
    """'Авария / Водоснабжение' -> 'Водоснабжение' (сопоставление с work_label)."""
    return full_name.split("/")[-1].strip() if "/" in full_name else full_name.strip()


class RobotBaseline:
    """Жадный матчинг синонимов. При нескольких совпадениях — категория с самым
    длинным совпавшим синонимом (более специфичный паттерн важнее)."""

    def __init__(self, rules_path: str = "configs/robot_synonyms.json",
                 work_labels: set | None = None):
        rules = load_rules(rules_path)
        # (синоним, child_категория), отсортировано по длине синонима убыв.
        self.patterns = []
        for full_name, syns in rules.items():
            child = full_to_child(full_name)
            for s in syns:
                self.patterns.append((s.lower(), child))
        self.patterns.sort(key=lambda x: len(x[0]), reverse=True)
        self.work_labels = work_labels  # для свёртки в таксономию проекта

    def predict_one(self, text: str) -> str:
        t = str(text).lower()
        for syn, child in self.patterns:
            if syn in t:
                if self.work_labels is None or child in self.work_labels:
                    return child
                return "Прочее"
        return "Прочее"  # ничего не сматчилось ~ робот ставит «Не определена»

    def predict(self, texts) -> list[str]:
        return [self.predict_one(t) for t in texts]
