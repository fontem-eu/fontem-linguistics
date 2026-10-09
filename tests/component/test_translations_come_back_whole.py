"""A translation that comes back cut short is asked for again, through the API.

DeepSeek-V4-Flash in JSON mode can end a translation early and still answer
valid JSON. It closes the string at a quote inside the text (“financial
stability objectives”) and goes on to the next language, or translates the
first paragraph of a text of several and stops. Nothing failed, so the cut
text was stored as the translation: 46 of 518 lobbying goals with line
breaks, 2026-10-09. Asked as tagged lines, a text cut at a quote came back
whole; a text of several paragraphs did not always, so those go a line at a
time, as every title does.
"""
from __future__ import annotations

import json
import re

import httpx
from fastapi.testclient import TestClient

from tests.component.test_api_routes import _detect_app

EU = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu", "it",
      "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
QUOTED = ("The FSB said the LEI underpins “financial stability objectives” and offers many "
          "benefits to the private sector, from cheaper onboarding to clearer reporting.")
PARAGRAPHS = ("This registration covers the Group and its subsidiaries in Europe.\n\n"
              "Collectively, our remit is to:\n"
              "- Build the infrastructure of e-commerce\n"
              "- Make it easy to do business anywhere")


def whole(lang: str, text: str) -> str:
    return f"[{lang}] {text}"


class Provider:
    """Answers the way the model does: in JSON, the languages in ``cut`` end
    where ``cut_at`` says; as tagged lines, every language comes back whole;
    in either, a text of several lines comes back as its first line."""

    def __init__(self, cut=(), cut_at=lambda text: text[:text.index("“")]):
        self.cut, self.cut_at = set(cut), cut_at
        self.asked: list[tuple[str, list[str]]] = []      # (json|tagged, languages)
        self.texts: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        text = prompt[prompt.index("Text: ") + len("Text: "):]
        asked = re.findall(r"\b([a-z]{2}) \(", prompt.split("Target languages:")[1].split("\n")[0])
        self.texts.append(text)
        text = text.split("\n")[0]
        if "response_format" in body:
            self.asked.append(("json", asked))
            content = json.dumps({k: whole(k, self.cut_at(text) if k in self.cut else text)
                                  for k in asked})
        else:
            self.asked.append(("tagged", asked))
            content = "\n".join(f"<{k}>{whole(k, text)}</{k}>" for k in asked)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 100}})


def translate(provider: Provider, text: str, source: str = "en", targets=None) -> dict:
    targets = targets or [c for c in EU if c != source]
    r = TestClient(_detect_app(provider)).post("/translate", json={
        "text": text, "source_lang": source, "targets": targets, "backend": "nebius"})
    assert r.status_code == 200, r.text
    return r.json()


def test_a_translation_cut_at_a_quote_is_asked_for_again_in_its_languages_only():
    provider = Provider(cut=["da", "es", "fi"])
    body = translate(provider, QUOTED)
    assert body["translations"] == {k: whole(k, QUOTED) for k in EU if k != "en"}
    kinds = [kind for kind, _langs in provider.asked]
    assert kinds.count("tagged") == 1
    (again,) = [langs for kind, langs in provider.asked if kind == "tagged"]
    assert sorted(again) == ["da", "es", "fi"]


def test_a_text_of_several_paragraphs_comes_back_with_all_of_them_where_they_were():
    """The list keeps its markers, the blank line stays blank."""
    body = translate(Provider(), PARAGRAPHS, targets=["de", "fr"])
    assert body["translations"]["de"] == (
        "[de] This registration covers the Group and its subsidiaries in Europe.\n\n"
        "[de] Collectively, our remit is to:\n"
        "- [de] Build the infrastructure of e-commerce\n"
        "- [de] Make it easy to do business anywhere")
    assert body["translations"]["fr"].count("[fr]") == 4


def test_a_line_without_words_is_kept_as_written_and_not_paid_for():
    provider = Provider()
    text = "Our members build wind farms across Europe.\n***\n2025\nThey also store energy."
    body = translate(provider, text, targets=["de"])
    assert body["translations"]["de"] == (
        "[de] Our members build wind farms across Europe.\n***\n2025\n"
        "[de] They also store energy.")
    assert sorted(provider.texts) == ["Our members build wind farms across Europe.",
                                      "They also store energy."]


def test_a_line_cut_at_a_quote_is_asked_for_again_too():
    provider = Provider(cut=["fr"])
    body = translate(provider, f"{QUOTED}\n\n{QUOTED}", targets=["de", "fr"])
    assert body["translations"]["fr"] == f"[fr] {QUOTED}\n\n[fr] {QUOTED}"


def test_what_asking_again_cost_is_part_of_the_price():
    once = translate(Provider(), QUOTED, targets=["de", "fr"])["cost_usd"]
    twice = translate(Provider(cut=["fr"]), QUOTED, targets=["de", "fr"])["cost_usd"]
    assert twice == 2 * once


def test_a_whole_answer_is_asked_for_once():
    provider = Provider()
    translate(provider, QUOTED)
    assert [kind for kind, _langs in provider.asked] == ["json"]


def test_a_short_title_is_not_judged_by_its_length():
    """“Obras” to “Works” is a translation, not a cut: a few characters say
    nothing about whether anything is missing."""
    provider = Provider(cut=["en"], cut_at=lambda text: text[:2])
    body = translate(provider, "Obras públicas", source="es", targets=["en"])
    assert body["translations"] == {"en": "[en] Ob"}
    assert len(provider.asked) == 1
