"""Manual check: type a product description — see what the system answers.

    .venv/bin/python -m eval.try              # on the real KTRU (data/processed/ktru.parquet)
    .venv/bin/python -m eval.try --json       # plus the full JSON response (as the frontend gets it)
"""
import argparse
import json

from ml.pipeline import Pipeline


def show(r: dict):
    for c in r["corrections"]:
        print(f"  Исправлено:     «{c['from']}» → «{c['to']}» ({c['reason']})")
    print("  Текст:          " + r["text"])
    for c in r["characteristics"]:
        val = c["value"] + (f" {c['unit']}" if c.get("unit") else "")
        extra = f", введено {c['user_value']}" if c.get("user_value") else ""
        print(f"  Характеристика: {c['name']} = {val}   ← «{c['source']}» ({c['method']}{extra})")
    for e in r["errors"]:
        print("  ОШИБКА:         " + e["message"] + (f" Возможно: «{e['suggestion']}»" if e.get("suggestion") else ""))
    print("  Сужение:        " + " → ".join(f"{s['stage']} ({s['count']})" for s in r["steps"]))
    names = list(dict.fromkeys(c["name"] for c in r["candidates"]))
    print(f"  Кандидаты:      {r['candidates_total']} шт.: " + "; ".join(names[:5]) + (" …" if len(names) > 5 else ""))
    if r["next_step"]:
        ns = r["next_step"]
        print("  Вопрос:         " + ns["message"] + " " + " | ".join(ns["options"][:10]))
    if r["ktru"]:
        print(f"  ИТОГ КТРУ:      {r['ktru']} {r['ktru_name']}")
    print("  Время, мс:      " + str(r["timings_ms"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ktru", default=None, help="путь к справочнику (по умолчанию data/processed/ktru.parquet)")
    ap.add_argument("--json", action="store_true", help="печатать полный JSON")
    args = ap.parse_args()

    pipe = Pipeline(**({"ktru_path": args.ktru} if args.ktru else {}))
    print("Введите описание товара (пустая строка — выход).")
    while True:
        try:
            text = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text:
            break
        r = pipe.analyze(text)
        show(r)
        if args.json:
            print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
