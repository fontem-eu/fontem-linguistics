"""LabseLocalBackend — mirror of the MinilmLocalBackend tests, covering
the added embed_batch path so the new batch-inference code has direct
unit coverage on both local backends."""
from __future__ import annotations

import pytest

from src.backends.labse_local import LabseLocalBackend
from src.domain.models import BackendUnavailable

pytestmark = pytest.mark.asyncio


async def test_missing_model_path_raises_backend_unavailable(tmp_path):
    b = LabseLocalBackend(
        model_path=str(tmp_path / "does-not-exist"),
        encoder_id="labse@0.0.0-test",
    )
    with pytest.raises(BackendUnavailable):
        await b._ensure_loaded()  # pylint: disable=protected-access


async def test_embed_batch_delegates_to_loaded_model(tmp_path):
    b = LabseLocalBackend(
        model_path=str(tmp_path),
        encoder_id="labse@0.0.0-test",
    )
    b._loaded = True  # pylint: disable=protected-access

    class _Batch:
        def encode(self, texts, normalize_embeddings):
            assert normalize_embeddings is True
            return _Arr([[float(i)] * 768 for i in range(len(texts))])

    class _Arr(list):
        def tolist(self):
            return list(self)

    b._model = _Batch()  # pylint: disable=protected-access
    vecs = await b.embed_batch(["a", "b"])
    assert [v[0] for v in vecs] == [0.0, 1.0]
    assert all(len(v) == 768 for v in vecs)


async def test_embed_batch_empty_returns_empty(tmp_path):
    b = LabseLocalBackend(
        model_path=str(tmp_path),
        encoder_id="labse@0.0.0-test",
    )
    b._loaded = True  # pylint: disable=protected-access
    assert await b.embed_batch([]) == []
