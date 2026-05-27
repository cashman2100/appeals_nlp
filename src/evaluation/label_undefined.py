"""
День 4 (а) — LLM-разметка заявок «Не определена» (UNDEFINED).

Протокол (зафиксирован в _internal/LLM_LABELING_PROMPT.md, _internal/CLAUDE.md):
  1. Сэмпл ~400 UNDEFINED с информативным текстом (len>=20).
  2. Двойная независимая разметка двумя LLM (DeepSeek + Anthropic).
  3. Cohen's kappa между разметчиками; kappa>0.6 = надёжно.
  4. Расхождения -> третий арбитр (YandexGPT) ИЛИ ручная вычитка.
  5. Gold-100: человек проверяет 100, считаем accuracy LLM-разметчика.

Ключи берутся из .env (НЕ хардкодить). Провайдеры опциональны: если ключа нет —
этот разметчик пропускается, протокол деградирует мягко (1 разметчик вместо 2).

Usage:
    python -m src.evaluation.label_undefined --config configs/base.yaml \
        --n 400 --providers deepseek,anthropic
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

import pandas as pd
import yaml

LABELS = ["Отопление", "Водоснабжение", "Электроснабжение", "Водоотведение",
          "Иное", "Домофоны", "ГВС", "ХВС", "Уборка уличная", "Крыша",
          "Вентиляция", "Засор водоотведения", "Уборка внутренняя",
          "Ремонт", "НЕЯСНО"]

SYSTEM_PROMPT = """Ты — эксперт диспетчерской службы ЖКХ. Тебе дают текст обращения жителя \
(распознанная речь из телефонного звонка, БЕЗ пунктуации, с ошибками ASR, \
разговорная). Твоя задача — определить категорию заявки.

Доступные категории (выбери РОВНО ОДНУ):
1. Отопление  2. Водоснабжение  3. Электроснабжение  4. Водоотведение
5. Иное  6. Домофоны  7. ГВС  8. ХВС  9. Уборка уличная  10. Крыша
11. Вентиляция  12. Засор водоотведения  13. Уборка внутренняя
14. Ремонт  15. НЕЯСНО — текст не позволяет определить категорию

Правила:
- ГВС = горячая вода; ХВС = холодная вода; Водоснабжение = вода в целом/неясно.
- "нет тепла","холодные батареи","когда дадут отопление" -> Отопление.
- "нет света","не горит лампочка в подъезде" -> Электроснабжение.
- "запах канализации","забился унитаз" -> Водоотведение/Засор водоотведения.
- Житель просто ругается/благодарит без сути -> 15 НЕЯСНО.
- Не угадывай наугад. Непонятно -> 15.

Верни ТОЛЬКО JSON, без markdown:
{"category":"<точное название>","confidence":<0.0-1.0>,"reason":"<5-10 слов>"}"""


# ---------- LLM-провайдеры (ключи из .env) ----------

def _parse(raw: str) -> dict:
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:]
    try:
        obj = json.loads(raw)
        cat = obj.get("category", "НЕЯСНО")
        if cat not in LABELS:
            cat = "НЕЯСНО"
        return {"category": cat,
                "confidence": float(obj.get("confidence", 0.0)),
                "reason": str(obj.get("reason", ""))[:120]}
    except Exception:
        return {"category": "НЕЯСНО", "confidence": 0.0, "reason": "parse_fail"}


def call_deepseek(text: str) -> dict:
    from openai import OpenAI
    cli = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"],
                 base_url="https://api.deepseek.com")
    r = cli.chat.completions.create(
        model="deepseek-chat", temperature=0,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": text}])
    return _parse(r.choices[0].message.content)


def call_anthropic(text: str) -> dict:
    import anthropic
    cli = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    r = cli.messages.create(
        model="claude-sonnet-4-6", max_tokens=200, temperature=0,
        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": text}])
    return _parse(r.content[0].text)


def call_yandex(text: str) -> dict:
    import requests
    key, folder = os.environ["YANDEX_API_KEY"], os.environ["YANDEX_FOLDER_ID"]
    r = requests.post(
        "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
        headers={"Authorization": f"Api-Key {key}"},
        json={"modelUri": f"gpt://{folder}/yandexgpt/latest",
              "completionOptions": {"temperature": 0, "maxTokens": 200},
              "messages": [{"role": "system", "text": SYSTEM_PROMPT},
                           {"role": "user", "text": text}]}, timeout=60)
    return _parse(r.json()["result"]["alternatives"][0]["message"]["text"])


PROVIDERS = {"deepseek": call_deepseek, "anthropic": call_anthropic,
             "yandex": call_yandex}


def available(names):
    """Оставляем только провайдеров, для которых есть ключ в окружении."""
    need = {"deepseek": ["DEEPSEEK_API_KEY"],
            "anthropic": ["ANTHROPIC_API_KEY"],
            "yandex": ["YANDEX_API_KEY", "YANDEX_FOLDER_ID"]}
    ok = []
    for n in names:
        if all(os.environ.get(k) for k in need[n]):
            ok.append(n)
        else:
            print(f"[skip] {n}: нет ключа в .env — разметчик пропущен")
    return ok


def cohen_kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b)


def main(config_path, n, providers):
    # подхватываем .env
    envp = Path(".env")
    if envp.exists():
        for line in envp.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                if v.strip():
                    os.environ.setdefault(k.strip(), v.strip())

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    pdir = Path(cfg["data"]["processed_dir"])
    full = pd.read_parquet(pdir / "full.parquet")

    und = full[full["work_label"] == "UNDEFINED"].copy()
    und["L"] = und["description"].str.len()
    und = und[und["L"] >= 20]
    sample = und.sample(min(n, len(und)),
                        random_state=cfg["project"]["seed"]).reset_index(drop=True)
    print(f"UNDEFINED pool={len(und):,} -> размечаем сэмпл {len(sample)}")

    names = available(providers)
    if not names:
        print("[abort] нет ни одного доступного LLM-провайдера. "
              "Заполни ключи в .env"); return

    out_dir = Path("results/undefined_labeling")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = {}
    for prov in names:
        fn = PROVIDERS[prov]
        rows = []
        print(f"[{prov}] разметка {len(sample)} примеров...")
        for i, txt in enumerate(sample["description"]):
            try:
                rows.append(fn(str(txt)))
            except Exception as e:
                rows.append({"category": "НЕЯСНО", "confidence": 0.0,
                             "reason": f"err:{type(e).__name__}"})
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(sample)}")
            time.sleep(0.1)  # бережём rate limit
        preds[prov] = [r["category"] for r in rows]
        pd.DataFrame(rows).assign(description=sample["description"]).to_csv(
            out_dir / f"labels_{prov}.csv", index=False)

    # Cohen's kappa между первыми двумя разметчиками
    if len(names) >= 2:
        k = cohen_kappa(preds[names[0]], preds[names[1]])
        agree = sum(a == b for a, b in zip(preds[names[0]], preds[names[1]]))
        print(f"\nCohen's kappa ({names[0]} vs {names[1]}) = {k:.3f}")
        print(f"Полное согласие: {agree}/{len(sample)} "
              f"({agree/len(sample)*100:.1f}%)")
        verdict = ("надёжно (kappa>0.6)" if k > 0.6
                   else "слабое согласие — нужна ручная вычитка")
        print(f"Вывод: {verdict}")

        # consensus + список спорных для ручного арбитража (gold)
        sample["label_a"] = preds[names[0]]
        sample["label_b"] = preds[names[1]]
        sample["agreed"] = sample["label_a"] == sample["label_b"]
        sample["consensus"] = sample.apply(
            lambda r: r["label_a"] if r["agreed"] else "DISPUTED", axis=1)
        sample.to_csv(out_dir / "consensus.csv", index=False)
        disp = (~sample["agreed"]).sum()
        print(f"Спорных (DISPUTED) для арбитража: {disp} "
              f"(~{disp/len(sample)*100:.0f}%)")
        # gold-100: первые 100 для ручной проверки человеком
        sample.head(100).to_csv(out_dir / "gold100_to_verify.csv", index=False)
        print(f"[done] -> {out_dir}/  "
              f"(consensus.csv, gold100_to_verify.csv, labels_*.csv)")
    else:
        print(f"\n[warn] доступен только 1 разметчик ({names[0]}). "
              "Протокол деградировал: нет kappa. Залей второй ключ для двойной "
              "разметки (требование протокола надёжности).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--providers", default="deepseek,anthropic")
    a = ap.parse_args()
    main(a.config, a.n, a.providers.split(","))
