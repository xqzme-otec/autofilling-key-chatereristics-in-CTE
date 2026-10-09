"""Plain-text preprocessing applied before the text is sent to the model service.

PLACEHOLDER: the real implementation will be provided later. Keep the signature:
the service layer depends on `preprocess(text) -> PreprocessedText` only.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class PreprocessedText:
    text: str
    # Optional tokens in the model service format: {text, lemma, span, known}.
    # None lets the model service tokenize the text itself.
    tokens: list[dict] | None = None


def preprocess(text: str) -> PreprocessedText:
    # TODO: replace with the real preprocessing; for now the text is passed through unchanged.
    return PreprocessedText(text=text)
