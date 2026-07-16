"""MinilmLocalBackend — same load-lock + BackendUnavailable behaviour as
LaBSE; the runtime import of sentence-transformers is skipped when the
dep is absent from the test environment."""
from __future__ import annotations

import pytest

from src.backends.minilm_local import MinilmLocalBackend
from src.domain.models import BackendUnavailable

pytestmark = pytest.mark.asyncio

async def test_missing_model_path_raises_backend_unavailable(tmp_path):
    """Cold-start with an absent model dir is the operator-visible failure
    mode when the mirror-pull InitContainer didn't run."""
    b = MinilmLocalBackend(
        model_path=str(tmp_path / "does-not-exist"),
        encoder_id="minilm@0.0.0-test",
    )
    with pytest.raises(BackendUnavailable) as exc:
        await b._ensure_loaded()  # pylint: disable=protected-access
    assert "minilm-local" in str(exc.value)
    assert "InitContainer" in str(exc.value)

async def test_embed_delegates_to_loaded_model(tmp_path, monkeypatch):
    """Once the model is 'loaded' (stubbed here), embed() runs
    encode(text, normalize_embeddings=True) in an executor and returns
    the vector as a list."""
    b = MinilmLocalBackend(
        model_path=str(tmp_path),
        encoder_id="minilm@0.0.0-test",
    )
    # Force the model-present branch: pretend load already happened.
    b._loaded = True  # pylint: disable=protected-access

    class _Stub:
        def encode(self, text, normalize_embeddings):
            assert normalize_embeddings is True
            # Return a numpy-array-like with tolist()
            return _Vec([0.1] * 384)

    class _Vec(list):
        def tolist(self):
            return list(self)

    b._model = _Stub()  # pylint: disable=protected-access
    vec = await b.embed("hello world")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert vec[0] == pytest.approx(0.1)

async def test_embed_batch_delegates_to_loaded_model(tmp_path):
    """embed_batch runs one encode(list) call and returns list-of-vectors,
    order preserved."""
    b = MinilmLocalBackend(
        model_path=str(tmp_path),
        encoder_id="minilm@0.0.0-test",
    )
    b._loaded = True  # pylint: disable=protected-access

    class _Batch:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            assert isinstance(texts, list)
            return _Arr([[float(i)] * 384 for i in range(len(texts))])

    class _Arr(list):
        def tolist(self):
            return list(self)

    b._model = _Batch()  # pylint: disable=protected-access
    vecs = await b.embed_batch(["a", "b", "c"])
    assert [v[0] for v in vecs] == [0.0, 1.0, 2.0]
    assert all(len(v) == 384 for v in vecs)


async def test_embed_batch_empty_returns_empty(tmp_path):
    """No-op fast path: an empty list short-circuits before touching the model."""
    b = MinilmLocalBackend(
        model_path=str(tmp_path),
        encoder_id="minilm@0.0.0-test",
    )
    # No model attached — should not be touched.
    b._loaded = True  # pylint: disable=protected-access
    assert await b.embed_batch([]) == []
