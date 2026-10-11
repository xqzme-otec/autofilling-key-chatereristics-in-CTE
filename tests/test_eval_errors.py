from eval.errors import evaluate


class FakePipeline:
    def __init__(self, results):
        self.results = results

    def analyze(self, text):
        return self.results[text]


def test_error_report_counts_detection_correction_and_clean_false_positives():
    cases = [
        {"id": "hit", "text": "a", "expected_errors": ["unit_mismatch"],
         "expected_corrections": [{"from": "стл", "to": "стул"}]},
        {"id": "miss", "text": "b", "expected_errors": ["nonexistent_value"]},
        {"id": "clean", "text": "c", "expected_errors": []},
    ]
    pipe = FakePipeline({
        "a": {"errors": [{"type": "unit_mismatch"}],
              "corrections": [{"from": "стл", "to": "стул", "reason": "опечатка"}]},
        "b": {"errors": [], "corrections": []},
        "c": {"errors": [{"type": "out_of_range"}], "corrections": []},
    })

    report = evaluate(pipe, cases)

    assert report["errors_micro"] == {"tp": 1, "fp": 1, "fn": 1, "precision": 0.5, "recall": 0.5,
                                      "f1": 0.5}
    assert report["clean_false_positive_rate"] == 1.0
    assert report["corrections"]["exact_case_accuracy"] == 1.0
