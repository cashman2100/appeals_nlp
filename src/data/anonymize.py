"""
Appeals-NLP — анонимизация ПДн (СТРЕТЧ, скелет).

Статус: НЕ РЕАЛИЗОВАНО. Это заглушка, чтобы pipeline был готов к анонимизации
БЕЗ переделки, когда дойдём до публичного релиза датасета (дни 11-12 / после MVP).

Включается флагом data.anonymize.enabled в конфиге. Пока enabled:false —
preprocess.py просто пропускает этот шаг.

План реализации (когда возьмёмся):
  1. Телефоны     — regex \b[78]\d{10}\b -> <PHONE>
  2. ФИО          — Natasha NER (PER) -> <PERSON>
  3. Адреса       — Natasha NER (LOC) + словарь улиц клиента -> <ADDR>
  4. Лицевые счета/суммы — regex -> <ACCOUNT>/<MONEY>
  5. Пограничное (медицина, угрозы) — LLM-проверка сэмпла
  6. Валидация    — ручная проверка маскирования на 200 примерах, отчёт recall
"""
from __future__ import annotations
import re

_PHONE_RE = re.compile(r"\b[78]\d{10}\b")


def anonymize_text(text: str) -> str:
    """MVP-уровень: маскируем только телефоны (дёшево, безопасно, без зависимостей).
    NER-маскирование ФИО/адресов добавляется здесь же на стретч-этапе."""
    if not isinstance(text, str):
        return text
    return _PHONE_RE.sub("<PHONE>", text)


def anonymize_dataframe(df, text_col: str = "description"):
    """Применяется в preprocess.py если data.anonymize.enabled = true."""
    df = df.copy()
    df[text_col] = df[text_col].map(anonymize_text)
    # TODO(stretch): NER ФИО/адреса, лицевые счета, LLM-проверка, отчёт recall
    return df
