"""Summaries and long texts through the HTTP API, against a scripted provider.

What a caller sees: a summary a tweet long in the text's own language and
its translations; a summary asked for twice paid for once; a lobbying goal
of 1,000 characters translated into all 23 languages although one answer
could not carry them all.
"""
from __future__ import annotations

import json
import re

import httpx
import pytest
from fastapi.testclient import TestClient

from tests.component.test_api_routes import _detect_app

EU = ["bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "ga", "hr", "hu", "it",
      "lt", "lv", "mt", "nl", "pl", "pt", "ro", "sk", "sl", "sv"]
GOALS = ("Die Interessen der deutschen Brauwirtschaft gegenüber den Institutionen der "
         "Europäischen Union vertreten, insbesondere in der Agrar-, Umwelt- und Steuerpolitik. ")


class Provider:
    """Answers each prompt the way the model does, and records them."""

    def __init__(self, summaries=None):
        self.summaries = list(summaries or ["Vertritt die deutsche Brauwirtschaft bei der EU."])
        self.prompts: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompt = body["messages"][0]["content"]
        self.prompts.append(prompt)
        if prompt.startswith("Summarise") or prompt.startswith("This summary is"):
            content = self.summaries.pop(0)
        else:                                            # a translation, JSON keyed by code
            keys = re.findall(r'"([a-z]{2})"', prompt.split("Target languages:")[0])
            content = json.dumps({k: f"[{k}] translated" for k in keys})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 300, "completion_tokens": 100}})

    def translation_calls(self):
        return [p for p in self.prompts if p.startswith("Translate")]


def test_a_summary_comes_in_the_texts_language_and_translated_into_the_rest():
    provider = Provider()
    client = TestClient(_detect_app(provider))
    r = client.post("/summarize", json={"text": GOALS * 6, "source_lang": "de", "targets": EU,
                                        "about": "what the organisation lobbies for"})
    assert r.status_code == 200
    body = r.json()
    assert body["lang"] == "de" and body["cached"] is False
    assert body["summaries"]["de"] == "Vertritt die deutsche Brauwirtschaft bei der EU."
    assert set(body["summaries"]) == set(EU) and body["summaries"]["fr"] == "[fr] translated"
    summarise = provider.prompts[0]
    assert "in German, in at most 280 characters" in summarise
    assert "what the organisation lobbies for" in summarise
    # The summary is what gets translated, not the full text.
    assert "Vertritt die deutsche Brauwirtschaft" in provider.translation_calls()[0]
    assert body["cost_usd"] > 0


def test_a_summary_asked_for_again_is_not_paid_for_again():
    """The second environment (shared, then prod) asks for the same text."""
    provider = Provider()
    client = TestClient(_detect_app(provider))
    ask = {"text": GOALS * 6, "source_lang": "de", "targets": ["en", "fr"]}
    first = client.post("/summarize", json=ask).json()
    asked = len(provider.prompts)
    second = client.post("/summarize", json=ask).json()
    assert second["summaries"] == first["summaries"] and second["cached"] is True
    assert second["cost_usd"] == 0 and len(provider.prompts) == asked


def test_a_summary_that_runs_long_is_shortened_then_cut_at_a_sentence():
    long = "Vertritt die deutsche Brauwirtschaft. " + "Sehr ausführlich, " * 30
    still_long = "Vertritt die deutsche Brauwirtschaft bei der EU. " + "Und mehr. " * 30
    provider = Provider([long, still_long])
    client = TestClient(_detect_app(provider))
    body = client.post("/summarize", json={"text": GOALS * 6, "source_lang": "de"}).json()
    assert body["summaries"] == {"de": still_long[:still_long.rindex(". ", 0, 280) + 1]}
    assert len(body["summaries"]["de"]) <= 280
    assert provider.prompts[1].startswith(f"This summary is {len(long.strip())} characters long")


def test_a_summary_of_a_text_in_no_known_language_is_written_in_english():
    provider = Provider(["Represents brewers."])
    body = TestClient(_detect_app(provider)).post(
        "/summarize", json={"text": "Bryggeriforeningen arbeider for norske bryggerier.",
                            "source_lang": "und", "targets": ["en", "de"]}).json()
    assert body["lang"] == "en" and body["summaries"]["en"] == "Represents brewers."
    assert "in English, in at most 280 characters" in provider.prompts[0]


def test_a_long_text_is_translated_into_every_language_across_several_calls():
    """A 1,000-character goal into 23 languages would not fit one answer."""
    provider = Provider()
    text = (GOALS * 7)[:1000]
    r = TestClient(_detect_app(provider)).post("/translate", json={
        "text": text, "source_lang": "de", "targets": [c for c in EU if c != "de"],
        "backend": "nebius"})
    assert r.status_code == 200
    translations = r.json()["translations"]
    assert set(translations) == set(EU) - {"de"}
    calls = provider.translation_calls()
    assert len(calls) >= 2
    asked = [re.findall(r'"([a-z]{2})"', c.split("Target languages:")[0]) for c in calls]
    assert sorted(code for group in asked for code in group) == sorted(set(EU) - {"de"})


def test_a_title_is_still_one_call():
    provider = Provider()
    TestClient(_detect_app(provider)).post("/translate", json={
        "text": "Roboty budowlane", "source_lang": "pl",
        "targets": [c for c in EU if c != "pl"], "backend": "nebius"})
    assert len(provider.translation_calls()) == 1


@pytest.mark.parametrize("backend", ["mistral", "nllb-local"])
def test_only_nebius_writes_summaries(backend):
    r = TestClient(_detect_app(Provider())).post(
        "/summarize", json={"text": "x", "source_lang": "en", "backend": backend})
    assert r.status_code == 400
