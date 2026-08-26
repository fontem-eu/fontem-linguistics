"""LocalSentenceTransformerBackend — the shared local-embedding base.

labse-local and minilm-local were duplicate implementations; they are now
one. That makes the load path worth testing directly rather than twice
over: it lazily pulls a several-hundred-MB model into RSS, and the
double-checked lock around it is what stops two concurrent /embed
requests loading it twice on a memory-capped pod.

torch and sentence_transformers are stubbed via sys.modules — the import
happens inside _ensure_loaded precisely so the package stays importable
without them.
"""
from __future__ import annotations

# pylint: disable=protected-access,missing-function-docstring,too-few-public-methods

import asyncio
import sys
import types
from unittest.mock import patch

import pytest

from src.backends.local_sentence_transformer import LocalSentenceTransformerBackend
from src.domain.models import BackendUnavailable

pytestmark = pytest.mark.asyncio


class _Backend(LocalSentenceTransformerBackend):
    backend_label = "test-local"


class _Arr(list):
    def tolist(self):
        return list(self)


class _Model:
    def __init__(self, path=None):
        self.path = path
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def encode(self, text, normalize_embeddings=False):
        assert normalize_embeddings is True
        if isinstance(text, list):
            return _Arr([[float(len(t))] for t in text])
        return _Arr([float(len(text))])


def _stub_libs(loaded, quantised=None):
    """Install fake torch + sentence_transformers for the duration."""
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = lambda path: loaded
    torch = types.ModuleType("torch")
    torch.nn = types.SimpleNamespace(Linear=object)
    torch.qint8 = "qint8"
    torch.quantization = types.SimpleNamespace(
        quantize_dynamic=lambda model, layers, dtype: quantised,
    )
    return patch.dict(sys.modules,
                      {"sentence_transformers": st, "torch": torch})


async def test_missing_model_path_names_the_backend(tmp_path):
    """The label is the only thing telling an operator which local backend
    failed, since both share this code path now."""
    b = _Backend(model_path=str(tmp_path / "nope"), encoder_id="e@1")
    with pytest.raises(BackendUnavailable) as exc:
        await b._ensure_loaded()
    assert str(exc.value).startswith("test-local: model_path")


async def test_absent_libraries_surface_as_backend_unavailable(tmp_path):
    """Import failure must not escape as ImportError — the API maps
    BackendUnavailable to a 503 with a useful body."""
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1")
    with patch.dict(sys.modules, {"sentence_transformers": None, "torch": None}):
        with pytest.raises(BackendUnavailable) as exc:
            await b._ensure_loaded()
    assert "test-local" in str(exc.value)


async def test_model_is_loaded_once_and_set_to_eval(tmp_path):
    model = _Model()
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1")
    with _stub_libs(model):
        await b._ensure_loaded()
        await b._ensure_loaded()
    assert b._model is model
    assert b._loaded is True
    assert model.eval_called is True


async def test_concurrent_callers_load_the_model_only_once(tmp_path):
    """Two requests arriving before the first load finishes must not both
    pull the weights into memory — that is an OOM on a capped pod."""
    loads = []

    class _SlowModel(_Model):
        def eval(self):
            loads.append(1)
            super().eval()

    model = _SlowModel()
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1")
    with _stub_libs(model):
        await asyncio.gather(*(b._ensure_loaded() for _ in range(5)))
    assert len(loads) == 1


async def test_quantise_replaces_the_model_when_enabled(tmp_path):
    original, quantised = _Model(), _Model()
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1", quantize=True)
    with _stub_libs(original, quantised=quantised):
        await b._ensure_loaded()
    assert b._model is quantised


async def test_quantise_is_skipped_when_disabled(tmp_path):
    original, quantised = _Model(), _Model()
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1", quantize=False)
    with _stub_libs(original, quantised=quantised):
        await b._ensure_loaded()
    assert b._model is original


async def test_embed_normalises_and_returns_a_plain_list(tmp_path):
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1")
    b._loaded = True
    b._model = _Model()
    vec = await b.embed("abcd")
    assert vec == [4.0]
    assert isinstance(vec, list)


async def test_embed_loads_on_first_use(tmp_path):
    """embed() must not assume _ensure_loaded ran — it is the entry point."""
    model = _Model()
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1")
    with _stub_libs(model):
        assert await b.embed("xy") == [2.0]
    assert b._loaded is True


async def test_embed_batch_preserves_input_order(tmp_path):
    """Callers zip the result back against their input; a reordered batch
    silently mislabels every vector."""
    b = _Backend(model_path=str(tmp_path), encoder_id="e@1")
    b._loaded = True
    b._model = _Model()
    assert await b.embed_batch(["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]


async def test_embed_batch_short_circuits_on_empty_input(tmp_path):
    """No model call at all — an empty batch must not force a load."""
    b = _Backend(model_path=str(tmp_path / "nope"), encoder_id="e@1")
    b._loaded = True
    assert await b.embed_batch([]) == []
