from __future__ import annotations

"""Step-by-step narrowing of KTRU candidates down to one code.

1. Name: if the text fully contains a position name («стул ученический»),
      only positions with that name remain.
2. Each found characteristic keeps the candidates where it exists and the value is allowed.
      If a characteristic contradicts all remaining ones, it is skipped with a «conflict» error.
3. One code left — done. Otherwise the next question:
      - different names → «which product exactly»;
      - one name → the characteristic whose answer leaves the fewest codes on average;
      - positions do not differ by characteristics → choose a code from the list.
Every step is written to steps — the frontend shows how the list narrows.
"""
from ml.ktru import KtruStore, format_value
from ml.text import main_noun, tokenize

MAX_PRODUCT_OPTIONS = 10


def _code_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        return "код"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "кода"
    return "кодов"


def _lemmas(text: str) -> list[str]:
    return [t["lemma"] for t in tokenize(text) if t["text"][0].isalnum()]


def _eq(a: str, b: str) -> bool:
    return a == b or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5])


def _char(store: KtruStore, code: str, name: str) -> dict | None:
    return next((ch for ch in store.by_code[code]["characteristics"] if ch["name"] == name), None)


def _options(ch: dict) -> list[str]:
    return [format_value(v, ch["unit"]) for v in ch["values"]]


def _norm_options(ch: dict) -> set[str]:
    """Options in normal form: «Женская»/«Женское»/«Женский» → «женский»."""
    return {" ".join(_lemmas(o)) for o in _options(ch)}


def answer_to_characteristic(answer: dict, codes: list[str], store: KtruStore) -> dict:
    """User answer to a question («Наличие подлокотников» = «Да») → a characteristic with the codes where the value is allowed."""
    name, value = answer["characteristic"], answer["value"]
    norm = " ".join(_lemmas(value))
    ok = [c for c in codes if (ch := _char(store, c, name)) and norm in _norm_options(ch)]
    unit = next((ch["unit"] for c in ok if (ch := _char(store, c, name))), None)
    return {"name": name, "value": value, "unit": unit, "source": "ответ пользователя", "span": None,
            "method": "answer", "score": 100, "codes": ok}


def narrow(characteristics: list[dict], candidates: list[dict], store: KtruStore,
           query: str = "", answers: list[dict] | None = None) -> tuple[list[dict], dict | None, list[dict], list[dict]]:
    """→ (remaining candidates, next step or None, step history, errors)."""
    answers = answers or []
    remaining = list(candidates)
    steps = [{"stage": "Поиск по описанию", "count": len(remaining)}]
    errors = []

    # 1. name: a user answer or a full match with the text
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
            longest = max(len(_lemmas(n)) for n in covered)  # «Стул ученический», not just «Стул»
            keep = {n for n in covered if len(_lemmas(n)) == longest}
            stage = "Наименование: " + ", ".join(f"«{n}»" for n in sorted(keep))
        elif noun:  # «стул» → only positions whose name contains «стул»
            keep = {n for n in names if any(_eq(noun, w) for w in _lemmas(n))}
            stage = f"Наименование содержит «{noun}»"
        else:
            keep = names
        if keep and keep != names or covered or (noun and keep):
            # the step is written even without narrowing: it confirms the product was determined by name
            remaining = [c for c in remaining if c["name"] in keep]
            steps.append({"stage": stage, "count": len(remaining)})

    code_answer = next((a["value"] for a in answers if a.get("type") == "code"), None)
    if code_answer:
        remaining = [c for c in remaining if c["code"] == code_answer] or remaining
        steps.append({"stage": f"Выбран код {code_answer}", "count": len(remaining)})

    # 2. характеристики. Hard drop only on a direct contradiction with KTRU; a candidate that lacks the
    # characteristic stays (soft penalty: it goes after the matching ones). Suspicious values do not narrow.
    for f in characteristics:
        if f.get("suspicious") or f.get("origin") in ("portal", "new"):
            continue
        has = [c for c in remaining if _char(store, c["code"], f["name"])]
        if not has:
            continue  # none of the remaining has this characteristic — does not narrow
        ok = [c for c in has if c["code"] in f["codes"]]
        lacking = [c for c in remaining if c not in has]
        label = f"{f['name']} = {f['value']}" + (f" {f['unit']}" if f.get("unit") else "")
        if not ok:
            allowed = sorted({o for c in has for o in _options(_char(store, c["code"], f["name"]))})
            shown = ", ".join(allowed[:10]) + (" …" if len(allowed) > 10 else "")
            errors.append({"type": "conflict", "characteristic": f["name"], "source": f["source"], "span": f["span"],
                           "message": f"«{label}» не подходит ни к одному из подходящих товаров. "
                                      f"Допустимые значения: {shown}. Характеристика не учтена."})
            f["conflict"] = True
            continue
        new_remaining = ok + lacking
        if len(new_remaining) == len(remaining) and [c["code"] for c in new_remaining] == [c["code"] for c in remaining]:
            continue
        remaining = new_remaining
        f["used_for_narrowing"] = True
        steps.append({"stage": label, "count": len(remaining)})

    # 3. result or the next question
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


def missing_required(remaining: list[dict], characteristics: list[dict], store: KtruStore, limit: int = 10) -> list[str]:
    """Required KTRU characteristics that the description does not specify: of the final position when the
    code is determined, otherwise only the ones every remaining position requires (needed whatever code
    is chosen; the rest is covered by the clarifying question)."""
    filled = {f["name"] for f in characteristics if not f.get("conflict")}
    counts: dict[str, int] = {}
    for c in remaining:
        for ch in store.by_code[c["code"]]["characteristics"]:
            if ch["required"] and ch["name"] not in filled:
                counts[ch["name"]] = counts.get(ch["name"], 0) + 1
    return [n for n in counts if counts[n] == len(remaining)][:limit]


def _best_question(remaining: list[dict], filled: set, store: KtruStore):
    """The characteristic whose answer leaves the fewest codes on average."""
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
            for i, v in enumerate(ch["values"]):  # option order as in KTRU (numbers ascending)
                label = format_value(v, ch["unit"])
                # numbers ascending, text alphabetically
                d["order"].setdefault(" ".join(_lemmas(label)),
                                      (0, 0, label) if "text" in v else (v.get("min", v.get("max", 0)) or 0, i, ""))

    best, best_key = None, None
    for name, d in per_char.items():
        options = set().union(*d["codes"].values())
        if len(options) < 2:
            continue
        missing = n - len(d["codes"])  # positions without the characteristic stay for any answer
        sizes = [sum(o in vals for vals in d["codes"].values()) for o in options]
        expected = missing + sum(size * size for size in sizes) / sum(sizes)
        if expected >= n:
            continue  # every answer keeps everyone — the question is useless
        key = (expected, not d["required"], len(options))
        if best_key is None or key < best_key:
            ordered = [d["label"][o] for o in sorted(options, key=lambda o: d["order"][o])]
            best, best_key = (name, ordered, d["required"]), key
    return best


def rerank_and_question(remaining: list[dict], characteristics: list[dict], signals: list[dict],
                        store: KtruStore, portal, searcher, answers: list[dict]):
    """Rank context evidence without excluding a KTRU-compatible position."""
    if not remaining:
        return remaining, None, None
    words = [s["value"] for s in signals]
    words += [f["value"] for f in characteristics if f.get("origin") in ("portal", "new")
              and f.get("value") not in (None, "Да", "Нет") and not f.get("used_for_narrowing")]
    if words and len(remaining) > 1:
        import numpy as np

        query = " ".join(words)
        qvec = searcher.encode([query])[0]
        passages = [f"{c['name']}. {store.by_code[c['code']].get('description') or ''}" for c in remaining]
        vectors = searcher.encode(passages, query=False)
        similarities = np.asarray(vectors) @ qvec
        for c, similarity in zip(remaining, similarities):
            name_lemmas = set(_lemmas(c["name"]))
            direct = sum(any(l in name_lemmas for l in _lemmas(word)) for word in words)
            c["score"] = round(float(c["score"]) + 0.12 * float(similarity) + 0.08 * direct, 4)
        remaining = sorted(remaining, key=lambda c: c["score"], reverse=True)
        for f in characteristics:
            if f.get("origin") in ("portal", "new"):
                f["used_for_reranking"] = True
        step = {"stage": "Ранжирование по контексту", "count": len(remaining)}
    else:
        step = None

    if len(remaining) <= 1:
        return remaining, None, step
    names = list(dict.fromkeys(c["name"] for c in remaining))
    if len(names) > 1:
        counts = {n: sum(c["name"] == n for c in remaining) for n in names}
        options = names[:MAX_PRODUCT_OPTIONS]
        return remaining, {"type": "product", "message": f"Осталось {len(remaining)} {_code_word(len(remaining))} разных товаров. Уточните вид товара:",
                           "characteristic": None, "options": options, "counts": [counts[n] for n in options]}, step
    filled = {f["name"] for f in characteristics if not f.get("conflict")}
    best = _best_question(remaining, filled, store)
    if best:
        name, options, required = best
        return remaining, {"type": "characteristic", "message": f"Уточните характеристику «{name}»: ",
                           "characteristic": name, "options": options, "required": required}, step
    return remaining, {"type": "code", "message": "Характеристики не различают коды. Выберите код:",
                       "characteristic": None, "options": [c["code"] for c in remaining]}, step
