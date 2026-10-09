from __future__ import annotations

"""Поэтапное сужение кандидатов КТРУ до одного кода.

1. Наименование: если текст полностью содержит название позиции («стул ученический»),
   остаются только позиции с таким названием.
2. Каждая найденная характеристика оставляет кандидатов, у которых она есть и значение допустимо.
   Если характеристика противоречит всем оставшимся — она пропускается с ошибкой «конфликт».
3. Остался один код — готово. Иначе следующий вопрос:
   - разные наименования → «какой именно товар»;
   - одно наименование → характеристика, ответ на которую в среднем оставляет меньше всего кодов;
   - позиции не различаются характеристиками → выбор кода из списка.
Каждый шаг пишется в steps — фронт показывает, как сужается список.
"""
from ml.ktru import KtruStore, format_value
from ml.text import main_noun, tokenize

MAX_PRODUCT_OPTIONS = 10


def _lemmas(text: str) -> list[str]:
    return [t["lemma"] for t in tokenize(text) if t["text"][0].isalnum()]


def _eq(a: str, b: str) -> bool:
    return a == b or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])


def _char(store: KtruStore, code: str, name: str) -> dict | None:
    return next((ch for ch in store.by_code[code]["characteristics"] if ch["name"] == name), None)


def _options(ch: dict) -> list[str]:
    return [format_value(v, ch["unit"]) for v in ch["values"]]


def _norm_options(ch: dict) -> set[str]:
    """Варианты в нормальной форме: «Женская»/«Женское»/«Женский» → «женский»."""
    return {" ".join(_lemmas(o)) for o in _options(ch)}


def answer_to_characteristic(answer: dict, codes: list[str], store: KtruStore) -> dict:
    """Ответ пользователя на вопрос («Наличие подлокотников» = «Да») → характеристика с кодами, где значение допустимо."""
    name, value = answer["characteristic"], answer["value"]
    norm = " ".join(_lemmas(value))
    ok = [c for c in codes if (ch := _char(store, c, name)) and norm in _norm_options(ch)]
    unit = next((ch["unit"] for c in ok if (ch := _char(store, c, name))), None)
    return {"name": name, "value": value, "unit": unit, "source": "ответ пользователя", "span": None,
            "method": "answer", "score": 100, "codes": ok}


def narrow(characteristics: list[dict], candidates: list[dict], store: KtruStore,
           query: str = "", answers: list[dict] | None = None) -> tuple[list[dict], dict | None, list[dict], list[dict]]:
    """→ (оставшиеся кандидаты, следующий шаг или None, история шагов, ошибки)."""
    answers = answers or []
    remaining = list(candidates)
    steps = [{"stage": "Поиск по описанию", "count": len(remaining)}]
    errors = []

    # 1. наименование: ответ пользователя или полное совпадение с текстом
    product = next((a["value"] for a in answers if a.get("type") == "product"), None)
    if product:
        remaining = [c for c in remaining if c["name"] == product] or remaining
        steps.append({"stage": f"Выбран товар «{product}»", "count": len(remaining)})
    elif query:
        q = _lemmas(query)
        names = {c["name"] for c in remaining}
        covered = [n for n in names if (nl := _lemmas(n)) and all(any(_eq(w, x) for x in q) for w in nl)]
        noun = main_noun(query)
        if covered:
            longest = max(len(_lemmas(n)) for n in covered)  # «Стул ученический», а не просто «Стул»
            keep = {n for n in covered if len(_lemmas(n)) == longest}
            stage = "Наименование: " + ", ".join(f"«{n}»" for n in sorted(keep))
        elif noun:  # «стул» → только позиции, в названии которых есть «стул»
            keep = {n for n in names if any(_eq(noun, w) for w in _lemmas(n))}
            stage = f"Наименование содержит «{noun}»"
        else:
            keep = names
        if keep and len(keep) < len(names):
            remaining = [c for c in remaining if c["name"] in keep]
            steps.append({"stage": stage, "count": len(remaining)})

    code_answer = next((a["value"] for a in answers if a.get("type") == "code"), None)
    if code_answer:
        remaining = [c for c in remaining if c["code"] == code_answer] or remaining
        steps.append({"stage": f"Выбран код {code_answer}", "count": len(remaining)})

    # 2. характеристики
    for f in characteristics:
        has = [c for c in remaining if _char(store, c["code"], f["name"])]
        if not has:
            continue  # ни у кого из оставшихся нет такой характеристики — не сужает
        ok = [c for c in has if c["code"] in f["codes"]]
        if f.get("method") == "name" and ok:
            # повтор названия («Вид товара = Шкаф»): отсекаем только противоречащих, а не тех, у кого её нет
            ok += [c for c in remaining if c not in has]
            ok.sort(key=remaining.index)
            if len(ok) == len(remaining):
                continue
        label = f"{f['name']} = {f['value']}" + (f" {f['unit']}" if f.get("unit") else "")
        if not ok:
            allowed = sorted({o for c in has for o in _options(_char(store, c["code"], f["name"]))})
            shown = ", ".join(allowed[:10]) + (" …" if len(allowed) > 10 else "")
            errors.append({"type": "conflict", "characteristic": f["name"], "source": f["source"], "span": f["span"],
                           "message": f"«{label}» не подходит ни к одному из подходящих товаров. "
                                      f"Допустимые значения: {shown}. Характеристика не учтена."})
            f["conflict"] = True
            continue
        remaining = ok
        steps.append({"stage": label, "count": len(remaining)})

    # 3. результат или следующий вопрос
    if len(remaining) <= 1:
        return remaining, None, steps, errors

    names = list(dict.fromkeys(c["name"] for c in remaining))
    if len(names) > 1:
        counts = {n: sum(c["name"] == n for c in remaining) for n in names}
        return remaining, {
            "type": "product",
            "message": f"Подходит {len(remaining)} кодов КТРУ разных товаров. Уточните, какой товар нужен:",
            "characteristic": None,
            "options": names[:MAX_PRODUCT_OPTIONS],
            "counts": [counts[n] for n in names[:MAX_PRODUCT_OPTIONS]],
        }, steps, errors

    filled = {f["name"] for f in characteristics if not f.get("conflict")}
    best = _best_question(remaining, filled, store)
    if best is None:
        return remaining, {
            "type": "code",
            "message": f"Подходит {len(remaining)} кодов КТРУ, они не различаются по характеристикам. Выберите код:",
            "characteristic": None,
            "options": [c["code"] for c in remaining],
        }, steps, errors
    name, options, required = best
    return remaining, {
        "type": "characteristic",
        "message": f"Подходит {len(remaining)} кодов КТРУ. Уточните характеристику «{name}»:",
        "characteristic": name,
        "options": options,
        "required": required,
    }, steps, errors


def _best_question(remaining: list[dict], filled: set, store: KtruStore):
    """Характеристика, ответ на которую в среднем оставляет меньше всего кодов."""
    n = len(remaining)
    per_char: dict[str, dict] = {}
    for c in remaining:
        for ch in store.by_code[c["code"]]["characteristics"]:
            if ch["name"] in filled:
                continue
            d = per_char.setdefault(ch["name"], {"codes": {}, "required": False, "order": {}})
            d["codes"][c["code"]] = _norm_options(ch)
            for o in _options(ch):
                d.setdefault("label", {}).setdefault(" ".join(_lemmas(o)), o)
            d["required"] |= ch["required"]
            for i, v in enumerate(ch["values"]):  # порядок вариантов как в КТРУ (числа — по возрастанию)
                label = format_value(v, ch["unit"])
                # числа — по возрастанию, текст — по алфавиту
                d["order"].setdefault(" ".join(_lemmas(label)),
                                      (0, 0, label) if "text" in v else (v.get("min", v.get("max", 0)) or 0, i, ""))

    best, best_key = None, None
    for name, d in per_char.items():
        options = set().union(*d["codes"].values())
        if len(options) < 2:
            continue
        missing = n - len(d["codes"])  # у кого характеристики нет — остаются при любом ответе
        expected = sum(sum(o in vals for vals in d["codes"].values()) + missing for o in options) / len(options)
        if expected >= n:
            continue  # любой ответ оставляет всех — вопрос бесполезен
        key = (expected, not d["required"], len(options))
        if best_key is None or key < best_key:
            ordered = [d["label"][o] for o in sorted(options, key=lambda o: d["order"][o])]
            best, best_key = (name, ordered, d["required"]), key
    return best
