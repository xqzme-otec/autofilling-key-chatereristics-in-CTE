from eval.run import _answer_for_case, resolve_dialog


CODE = "00.00.00.000-00000001"


class FakeStore:
    by_code = {
        CODE: {
            "name": "Table",
            "characteristics": [{"name": "Color", "unit": None, "values": [{"text": "Blue"}]}],
        }
    }


class DialogPipeline:
    store = FakeStore()

    def analyze(self, request):
        answers = request["answers"]
        if len(answers) == 1:
            return {"ktru": None, "next_step": {"type": "characteristic", "characteristic": "Color",
                                                 "options": ["Blue", "Red"]}}
        if len(answers) == 2:
            return {"ktru": None, "next_step": {"type": "code", "options": [CODE]}}
        return {"ktru": CODE, "next_step": None}


def test_oracle_dialog_reaches_exact_code_through_all_question_types():
    pipe = DialogPipeline()
    case = {"ktru": CODE, "name": "Table", "text": "blue table", "chars": {"Color": "Blue"}}
    first = {"ktru": None, "next_step": {"type": "product", "options": ["Table", "Chair"]}}

    result = resolve_dialog(pipe, case, first)

    assert result["result"]["ktru"] == CODE
    assert result["questions"] == 3
    assert result["completed"] is True
    assert result["used_code_choice"] is True


def test_oracle_rejects_question_without_labeled_option():
    pipe = DialogPipeline()
    case = {"ktru": CODE, "name": "Table", "text": "blue table", "chars": {}}
    question = {"type": "characteristic", "characteristic": "Color", "options": ["Red"]}

    assert _answer_for_case(pipe, case, question) is None
