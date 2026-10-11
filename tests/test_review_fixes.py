from ml.ktru import MOCK_PATH, KtruStore
from ml.narrow import missing_required, narrow
from ml.numwords import find_number_words
from ml.text import to_nominative
from ml.units import NumberMatcher, _mentions
from ml.values import ValueMatcher

HAMMER = "25.73.30.141-MOCK0006"
CHAIR = "31.01.11.150-MOCK0001"
ARMCHAIR = "31.01.11.150-MOCK0002"
STOOL = "31.01.12.160-MOCK0003"


def store():
    return KtruStore(MOCK_PATH)


def test_number_words():
    unit = lambda w: w.startswith(("килограмм", "метр", "сантиметр"))
    assert [w["value"] for w in find_number_words("молоток полтора килограмма", unit)] == [1.5]
    assert [w["value"] for w in find_number_words("два с половиной метра", unit)] == [2.5]
    assert [w["value"] for w in find_number_words("сто семьдесят сантиметров", unit)] == [170]


def test_decimal_comma_is_not_a_separator():
    assert [(m["a"], m["unit_raw"]) for m in _mentions("молоток 1,5кг")] == [(1.5, "кг")]


def test_number_words_mapped_to_ktru():
    found, _, _, _ = NumberMatcher(store()).match("молоток полтора килограмма", [HAMMER])
    assert [(f["name"], f["value"]) for f in found] == [("Масса молотка", "0.8–1.6")]


def test_unit_only_needs_matching_meaning():
    # length with no keyword is ambiguous (рукоятка?) — not guessed, returned as unmatched
    found, _, _, unmatched = NumberMatcher(store()).match("молоток 30 см", [HAMMER])
    assert found == [] and unmatched and unmatched[0]["value_text"] == "30 см"


def test_user_range_fits_ktru_range():
    found, errors, _, _ = NumberMatcher(store()).match("молоток массой 0.4-0.8 кг", [HAMMER])
    assert found and found[0]["value"] == "0.4–0.8" and errors == []


def test_yes_no_contradiction():
    _, errors, _ = ValueMatcher(store()).match("стул с подлокотниками без подлокотников", [CHAIR])
    assert [e["type"] for e in errors] == ["contradiction"]


def test_several_values_warning():
    _, errors, _ = ValueMatcher(store()).match("стул, обивка ткань сетка", [CHAIR])
    assert [(e["type"], e["severity"]) for e in errors] == [("multiple_values", "warning")]


def test_soft_narrowing_keeps_candidates_without_characteristic():
    st = store()
    cands = [{"code": CHAIR, "name": "Стул офисный", "score": 0.9}, {"code": HAMMER, "name": "Молоток", "score": 0.8}]
    chars = [{"name": "Наличие подлокотников", "value": "Да", "codes": [CHAIR], "source": "", "span": None}]
    remaining, _, _, _ = narrow(chars, cands, st)
    assert [c["code"] for c in remaining] == [CHAIR, HAMMER]  # the hammer lacks the characteristic — stays, ranked after


def test_contradiction_drops_candidate():
    st = store()
    cands = [{"code": CHAIR, "name": "Стул офисный", "score": 0.9}, {"code": ARMCHAIR, "name": "Кресло", "score": 0.8}]
    chars = [{"name": "Наличие подлокотников", "value": "Нет", "codes": [CHAIR], "source": "", "span": None}]
    remaining, _, _, _ = narrow(chars, cands, st)
    assert [c["code"] for c in remaining] == [CHAIR]  # the armchair has only «Да» — direct contradiction


def test_missing_required():
    st = store()
    names = missing_required([{"code": CHAIR}], [{"name": "Материал обивки"}], st)
    assert "Наличие колесных опор" in names and "Материал обивки" not in names


def test_missing_required_of_several_codes_is_what_all_of_them_require():
    st = store()
    both = missing_required([{"code": CHAIR}, {"code": ARMCHAIR}], [], st)
    assert both == ["Наличие колесных опор", "Наличие подлокотников", "Материал обивки"]
    # the chair and the stool require different things: whatever is still needed depends on the code
    assert missing_required([{"code": CHAIR}, {"code": STOOL}], [], st) == []
    assert missing_required([{"code": STOOL}], [], st) == ["Материал каркаса"]


def test_to_nominative():
    assert to_nominative(["противоскользящей", "рукояткой"]) == "противоскользящая рукоятка"


def test_milligrams_converted():
    found, _, _, _ = NumberMatcher(store()).match("молоток массой 500000 мг", [HAMMER])
    assert [(f["value"], f["user_value"]) for f in found] == [("0.4–0.8", "0.5 Килограмм")]


def test_conversion_has_no_float_noise():
    # 100000 мг * 1e-6 = 0.09999999999999999 in floats; with an exclusive bound it would fall into «< 0.1»
    from ml.units import contains
    assert not contains({"max": 0.1, "max_incl": False}, round(100000 * 1e-6, 9))


def test_unit_words_are_not_typos():
    from ml.correct import Corrector
    from ml.text import tokenize
    c = Corrector(store())
    text, corrections = c.correct("молоток 1500 мкг", tokenize("молоток 1500 мкг"))
    assert corrections == [] and text.endswith("мкг")
