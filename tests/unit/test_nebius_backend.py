"""Nebius backend: the wire contract, the prompt, and what a call costs."""
from __future__ import annotations

import json

import httpx
import pytest

from src.backends.nebius import (
    NebiusBackend,
    NebiusBatchRefused,
    NebiusError,
    NebiusTransientError,
)
from src.backends.openai_chat import (
    NAMES_INSTRUCTION,
    PLACEHOLDER_INSTRUCTION,
    SCRIPT_INSTRUCTION,
    build_detect_prompt,
    build_translate_prompt,
    parse_detect_response,
    parse_translation_response,
)

pytestmark = pytest.mark.asyncio

TARGETS = ["mt", "ga", "et"]


def _build(handler, max_retries: int = 2) -> NebiusBackend:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://api.studio.nebius.com/v1",
        headers={"Authorization": "Bearer test-key-never-used"},
    )
    return NebiusBackend(
        api_url="https://api.studio.nebius.com/v1",
        api_key="test-key-never-used",
        chat_model="google/gemma-3-27b-it",
        timeout_s=5.0,
        max_retries=max_retries,
        price_input_per_mtok=0.13,
        price_output_per_mtok=0.40,
        client=client,
    )


def _completion(payload: dict, usage: dict | None = None) -> httpx.Response:
    return httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": usage or {"prompt_tokens": 138, "completion_tokens": 830},
    })


async def test_translate_returns_every_requested_target():
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        body = json.loads(request.content)
        assert body["model"] == "google/gemma-3-27b-it"
        assert body["response_format"] == {"type": "json_object"}
        assert body["temperature"] == 0.0
        return _completion({"mt": "Xogħol", "ga": "Obair", "et": "Töö"})

    got = await _build(handler).translate("Roboty budowlane", "pl", TARGETS)
    assert got == {"mt": "Xogħol", "ga": "Obair", "et": "Töö"}


async def test_the_prompt_names_the_language_not_just_the_code():
    """A model handed only `mt` has been known to answer in Malay. The
    prompt spells out Maltese, Irish and Estonian for exactly that reason."""
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["prompt"] = json.loads(request.content)["messages"][0]["content"]
        return _completion({t: "x" for t in TARGETS})

    await _build(handler).translate("Roboty budowlane", "pl", TARGETS)
    prompt = seen["prompt"]
    for name in ("Maltese", "Irish", "Estonian", "Polish"):
        assert name in prompt
    assert "Roboty budowlane" in prompt


async def test_a_missing_target_is_an_error_not_a_partial_result():
    """Writing a record as translated with languages quietly absent is worse
    than failing: nothing downstream would ever come back for them."""
    async def handler(_request: httpx.Request) -> httpx.Response:
        return _completion({"mt": "Xogħol", "ga": "Obair"})      # et missing

    with pytest.raises(NebiusError, match="missing/malformed"):
        await _build(handler).translate("Roboty budowlane", "pl", TARGETS)


async def test_cost_comes_from_the_providers_usage_block():
    """Measured 2026-09-23 on a real contract title: 138 + 830 tokens.
    At $0.13/$0.40 per Mtok that is $0.00035 — the number the spend cap
    settles on, rather than the pre-call estimate."""
    async def handler(_request: httpx.Request) -> httpx.Response:
        return _completion({t: "x" for t in TARGETS})

    backend = _build(handler)
    _got, cost = await backend.translate_with_cost("Roboty budowlane", "pl", TARGETS)
    assert cost == pytest.approx((138 * 0.13 + 830 * 0.40) / 1_000_000)
    # The estimate under-reserves, which is why finalize exists.
    assert backend.estimate_chat_usd(len("Roboty budowlane"), len(TARGETS)) < cost


async def test_5xx_is_retried_and_4xx_is_not():
    calls = {"n": 0}

    async def flaky(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="upstream busy")
        return _completion({t: "x" for t in TARGETS})

    assert await _build(flaky).translate("t", "pl", TARGETS)
    assert calls["n"] == 2

    calls["n"] = 0

    async def bad_model(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(404, text="model not found")

    with pytest.raises(NebiusError, match="404"):
        await _build(bad_model).translate("t", "pl", TARGETS)
    assert calls["n"] == 1, "a 404 is our mistake; retrying only burns the deadline"


async def test_exhausted_retries_surface_as_transient():
    async def always_503(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="still busy")

    with pytest.raises(NebiusTransientError):
        await _build(always_503, max_retries=1).translate("t", "pl", TARGETS)


async def test_malformed_json_is_rejected():
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "not json at all"}}],
            "usage": {},
        })

    with pytest.raises(NebiusError, match="malformed"):
        await _build(handler).translate("t", "pl", TARGETS)


async def test_an_undetermined_source_asks_the_model_to_look():
    """A Norwegian title labelled English is worse than no label: the model
    is told what to translate from, and the English target is never asked
    for because the runner thinks it already has it."""
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["prompt"] = json.loads(request.content)["messages"][0]["content"]
        return _completion({"en": "Wind turbine procurement", "mt": "x"})

    await _build(handler).translate("Anskaffelse av vindturbiner", "und", ["en", "mt"])
    assert "may equal the original text only for the language" in seen["prompt"]
    assert '"source_lang"' in seen["prompt"]
    assert "return the text unchanged" not in seen["prompt"]
    assert "from und" not in seen["prompt"]


async def test_the_plain_prompt_reads_as_before():
    """names=False is the fallback for an answer in the wrong script: the
    prompt as it was before the names guidance, which never mixed scripts."""
    assert build_translate_prompt("Travaux", "fr", ["de", "en"], names=False) == (
        "Translate the following text from French into the target languages. "
        "Preserve institutional terminology, do not paraphrase. Return strict "
        'JSON with keys: "de", "en". No prose, no explanation.\n'
        "Target languages: de (German), en (English).\nText: Travaux")


async def test_the_prompt_asks_to_translate_everything_but_names():
    prompt = build_translate_prompt("Ville de Dugny", "fr", ["de", "en"])
    assert NAMES_INSTRUCTION in prompt and SCRIPT_INSTRUCTION in prompt
    assert PLACEHOLDER_INSTRUCTION not in prompt and prompt.endswith("Text: Ville de Dugny")


async def test_brand_like_tokens_go_as_placeholders():
    prompt = build_translate_prompt("Projekt save2safe der CPAM", "de", ["en"])
    assert PLACEHOLDER_INSTRUCTION in prompt
    assert prompt.endswith("Text: Projekt {1} der {2}") and "save2safe" not in prompt


def _answer(translations: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(translations)}}], "usage": {}}


async def test_placeholders_are_put_back():
    got, _ = parse_translation_response(_answer({"en": "{2} project {1}"}), ["en"],
                                        NebiusError, "Projekt save2safe der CPAM")
    assert got == {"en": "CPAM project save2safe"}


@pytest.mark.parametrize("answer", [{"en": "Project"}, {"en": "Project {1} {1}"}])
async def test_a_placeholder_lost_or_doubled_is_asked_again_plainly(answer):
    with pytest.raises(NebiusError) as err:
        parse_translation_response(_answer(answer), ["en"], NebiusError, "Projekt save2safe")
    assert err.value.retry_plainly


async def test_the_plain_prompts_answer_is_taken_as_it_is():
    got, _ = parse_translation_response(_answer({"en": "Project save2safe"}), ["en"],
                                        NebiusError, "Projekt save2safe", names=False)
    assert got == {"en": "Project save2safe"}


async def test_a_latin_language_in_cyrillic_is_an_error_marked_as_such():
    with pytest.raises(NebiusError) as err:
        parse_translation_response(_answer({"cs": "Město Пеннес Мирабо", "bg": "Град Пен"}),
                                   ["cs", "bg"], NebiusError, "VILLE DES PENNES MIRABEAU")
    assert err.value.retry_plainly


@pytest.mark.parametrize("answer, source, expected", [
    ({"bg": "Град Дюни", "el": "Πόλη Ντυνί", "en": "City of Dugny"}, "Ville de Dugny", None),
    # The original has Cyrillic letters itself.
    ({"en": "Delivery for {1}"}, "Доставка за ГДГП", {"en": "Delivery for ГДГП"}),
])
async def test_cyrillic_and_greek_where_they_belong_are_fine(answer, source, expected):
    got, _ = parse_translation_response(_answer(answer), list(answer), NebiusError, source)
    assert got == (expected or answer)


async def test_an_answer_in_the_wrong_script_is_asked_again_plainly():
    prompts = []

    async def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][0]["content"]
        prompts.append(prompt)
        if len(prompts) == 1:
            return _completion({"cs": "Město Пеннес", "de": "Stadt Pennes"},
                               {"prompt_tokens": 100, "completion_tokens": 100})
        return _completion({"cs": "Město Pennes", "de": "Stadt Pennes"},
                           {"prompt_tokens": 100, "completion_tokens": 100})

    got, cost = await _build(handler).translate_with_cost("VILLE DES PENNES", "fr", ["cs", "de"])
    assert got == {"cs": "Město Pennes", "de": "Stadt Pennes"}
    assert NAMES_INSTRUCTION in prompts[0] and NAMES_INSTRUCTION not in prompts[1]
    assert cost == pytest.approx(2 * (100 * 0.13 + 100 * 0.40) / 1e6)    # both calls paid


async def test_a_text_of_nothing_but_names_is_its_own_translation():
    async def handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("no call for a text that is only a brand")
    got, cost = await _build(handler).translate_with_cost("save2safe", "en", ["de", "fr"])
    assert got == {"de": "save2safe", "fr": "save2safe"} and cost == 0.0


async def test_any_other_malformed_answer_is_not_asked_again():
    calls = []

    async def handler(_request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _completion({"cs": "Město"})                     # de missing

    with pytest.raises(NebiusError):
        await _build(handler).translate_with_cost("Ville", "fr", ["cs", "de"])
    assert len(calls) == 1


# ── language detection ──────────────────────────────────────────────


async def test_detect_asks_once_with_each_text_under_its_own_key():
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return _completion({"0": "pl", "1": "en"}, {"prompt_tokens": 400, "completion_tokens": 20})

    codes, cost = await _build(handler).detect_with_cost(["Roboty budowlane", "Road works"])
    assert codes == ["pl", "en"]
    assert cost == pytest.approx((400 * 0.13 + 20 * 0.40) / 1e6)
    prompt = seen["body"]["messages"][0]["content"]
    assert '"0": "Roboty budowlane"' in prompt and '"1": "Road works"' in prompt
    assert seen["body"]["response_format"] == {"type": "json_object"}


async def test_the_detect_prompt_names_the_eu_languages_as_the_expected_set():
    """Without them Gemma called Croatian and Slovene titles Serbian."""
    prompt = build_detect_prompt(["Nabava serverske infrastrukture"])
    for name in ("hr (Croatian)", "sl (Slovene)", "mt (Maltese)", "ga (Irish)"):
        assert name in prompt


async def test_an_unanswered_or_malformed_item_is_none_and_its_neighbours_stand():
    data = {"choices": [{"message": {"content": json.dumps(
        {"0": "FR", "2": "Swedish", "3": "und", "4": 7})}}]}
    codes, _usage = parse_detect_response(data, 5, NebiusError)
    assert codes == ["fr", None, None, "und", None]


async def test_an_unreadable_detection_is_an_error_for_the_call():
    data = {"choices": [{"message": {"content": "[\"fr\"]"}}]}
    with pytest.raises(NebiusError, match="not a JSON object"):
        parse_detect_response(data, 1, NebiusError)


# ── Batch inference ──────────────────────────────────────────────


class _BatchApi:
    """The files and batches endpoints, answering as Nebius does."""

    def __init__(self, create_status: int = 200) -> None:
        self.create_status = create_status
        self.uploads: list[bytes] = []
        self.created: list[dict] = []
        self.deleted: list[str] = []

    # One return per endpoint it plays, as a router would have.
    async def __call__(  # pylint: disable=too-many-return-statements
        self, request: httpx.Request,
    ) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "POST" and path.endswith("/files"):
            self.uploads.append(request.content)
            return httpx.Response(200, json={"id": "file-in", "purpose": "batch"})
        if method == "POST" and path.endswith("/batches"):
            if self.create_status != 200:
                return httpx.Response(self.create_status, json={
                    "detail": "Creating new batch job is temporarily unavailable"})
            self.created.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "batch-1", "status": "validating"})
        if method == "GET" and path.endswith("/batches/batch-1"):
            return httpx.Response(200, json={"id": "batch-1", "status": "completed",
                                             "output_file_id": "file-out"})
        if method == "GET" and path.endswith("/files/file-out/content"):
            return httpx.Response(200, text='{"custom_id": "a"}\n\n{"custom_id": "b"}\n')
        if method == "DELETE":
            self.deleted.append(path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={"deleted": True})
        return httpx.Response(404, json={"detail": "not found"})


async def test_a_batch_line_carries_the_same_request_as_a_realtime_call():
    """Same prompt, model, JSON mode and temperature: a title translated
    in a batch must read like one translated now."""
    nebius = _build(_BatchApi())
    line = nebius.batch_line("item-7", "Roboty budowlane", "pl", TARGETS)
    assert line == {"custom_id": "item-7", "method": "POST", "url": "/v1/chat/completions",
                    "body": nebius.translate_payload("Roboty budowlane", "pl", TARGETS)}
    assert line["body"]["messages"][0]["content"] == build_translate_prompt(
        "Roboty budowlane", "pl", TARGETS)


async def test_submit_batch_uploads_the_lines_then_starts_the_batch():
    api = _BatchApi()
    nebius = _build(api)
    lines = [nebius.batch_line(i, t, "pl", TARGETS) for i, t in (("a", "Łódź"), ("b", "Kraków"))]
    assert await nebius.submit_batch(lines, {"job_id": "j1"}) == ("batch-1", "file-in")
    uploaded = api.uploads[0].decode("utf-8")
    assert '"custom_id": "a"' in uploaded and "Łódź" in uploaded     # unescaped UTF-8
    assert b'name="purpose"' in api.uploads[0] and b"batch" in api.uploads[0]
    assert api.created == [{"input_file_id": "file-in", "endpoint": "/v1/chat/completions",
                            "completion_window": "24h", "metadata": {"job_id": "j1"}}]


async def test_a_refused_batch_is_its_own_error_and_leaves_no_file_behind():
    api = _BatchApi(create_status=403)
    nebius = _build(api)
    with pytest.raises(NebiusBatchRefused):
        await nebius.submit_batch([nebius.batch_line("a", "x", "pl", TARGETS)], {})
    assert api.deleted == ["file-in"]


async def test_a_provider_outage_on_batch_creation_is_transient():
    nebius = _build(_BatchApi(create_status=503))
    with pytest.raises(NebiusTransientError):
        await nebius.submit_batch([nebius.batch_line("a", "x", "pl", TARGETS)], {})


async def test_batch_and_result_file_are_read_back():
    nebius = _build(_BatchApi())
    batch = await nebius.get_batch("batch-1")
    assert batch["status"] == "completed"
    assert await nebius.file_lines(batch["output_file_id"]) == [
        {"custom_id": "a"}, {"custom_id": "b"}]


async def test_a_missing_batch_is_an_error_and_a_network_failure_transient():
    nebius = _build(_BatchApi())
    with pytest.raises(NebiusError):
        await nebius.get_batch("nope")

    async def down(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable")
    with pytest.raises(NebiusTransientError):
        await _build(down).get_batch("batch-1")


async def test_deleting_a_file_is_best_effort():
    async def fails(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")
    await _build(fails).delete_file("file-x")      # does not raise


async def test_the_batch_reservation_covers_what_a_title_actually_cost():
    """Measured 2026-09-23: 138 prompt + 830 completion tokens for one
    title into 23 languages. The reservation must not be below it."""
    nebius = _build(_BatchApi())
    actual = nebius.batch_chat_usd({"prompt_tokens": 138, "completion_tokens": 830})
    assert actual == pytest.approx((138 * 0.05 + 830 * 0.15) / 1_000_000)
    assert nebius.estimate_batch_usd(100, 23) >= actual


async def test_batch_prices_are_half_the_realtime_ones_by_default():
    nebius = _build(_BatchApi())
    usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    assert nebius.batch_chat_usd(usage) == pytest.approx(0.20)
