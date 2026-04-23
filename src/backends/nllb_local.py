"""NLLB-200 distilled-600M backend. CPU-only. Lazy-loaded.

ML dependencies (torch, transformers, sentencepiece) are imported lazily so
that unit tests and Mistral-only deployments don't require them.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from src.domain.models import BackendUnavailable


_NLLB_LANG = {
    "en": "eng_Latn", "fr": "fra_Latn", "de": "deu_Latn", "es": "spa_Latn",
    "it": "ita_Latn", "pl": "pol_Latn", "pt": "por_Latn", "nl": "nld_Latn",
    "ro": "ron_Latn", "cs": "ces_Latn", "hu": "hun_Latn", "sv": "swe_Latn",
    "da": "dan_Latn", "fi": "fin_Latn", "el": "ell_Grek", "bg": "bul_Cyrl",
    "hr": "hrv_Latn", "sk": "slk_Latn", "sl": "slv_Latn", "lt": "lit_Latn",
    "lv": "lvs_Latn", "et": "est_Latn", "ga": "gle_Latn", "mt": "mlt_Latn",
}


def to_nllb_code(iso: str) -> str:
    code = _NLLB_LANG.get(iso.lower())
    if code is None:
        raise BackendUnavailable(f"nllb-local: unsupported language {iso!r}")
    return code


@dataclass
class NllbLocalBackend:
    model_name: str
    local_path: str
    # Pinning revisions is supply-chain hygiene: without this, transformers
    # pulls whatever "main" points to on HF Hub, which can silently change.
    model_revision: str = "main"
    # Dynamic int8 quantisation on Linear layers — cuts resident memory
    # roughly 4x on the weights (~2.4 GB fp32 → ~0.7 GB int8 for NLLB-600M),
    # at a small throughput cost on CPU. We run CPU-only anyway; latency
    # isn't the binding constraint. Flip to False via env to debug quality
    # regressions against the fp32 baseline.
    quantize: bool = True
    _tokenizer: object | None = None
    _model: object | None = None
    _loaded: bool = False
    _load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        async with self._load_lock:
            if self._loaded:
                return
            try:
                import torch  # pylint: disable=import-outside-toplevel
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            except ImportError as exc:
                raise BackendUnavailable(
                    f"nllb-local: transformers/torch not installed ({exc})"
                ) from exc
            loop = asyncio.get_running_loop()
            tok = await loop.run_in_executor(
                None, lambda: AutoTokenizer.from_pretrained(
                    self.model_name, cache_dir=self.local_path,
                    revision=self.model_revision,
                ),
            )

            def _load_and_quantise():
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name, cache_dir=self.local_path,
                    revision=self.model_revision,
                )
                model.eval()
                if self.quantize:
                    # qint8 on nn.Linear is the only safe target on CPU —
                    # embeddings + layer-norms stay fp32 (not in the set).
                    # Torch does the cast at op-time (dynamic), so there's
                    # no calibration step to worry about.
                    model = torch.quantization.quantize_dynamic(
                        model, {torch.nn.Linear}, dtype=torch.qint8,
                    )
                return model

            model = await loop.run_in_executor(None, _load_and_quantise)
            self._tokenizer = tok
            self._model = model
            self._loaded = True

    async def translate(
        self, text: str, source_lang: str, targets: list[str]
    ) -> dict[str, str]:
        await self._ensure_loaded()
        src_code = to_nllb_code(source_lang)
        tok = self._tokenizer
        model = self._model

        def _run_one(tgt_iso: str) -> str:
            tgt_code = to_nllb_code(tgt_iso)
            tok.src_lang = src_code
            enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
            forced_bos = tok.convert_tokens_to_ids(tgt_code)
            out = model.generate(
                **enc, forced_bos_token_id=forced_bos, max_new_tokens=256
            )
            return tok.batch_decode(out, skip_special_tokens=True)[0]

        loop = asyncio.get_running_loop()
        results: dict[str, str] = {}
        # Run sequentially — NLLB-200 on CPU is the bottleneck; parallelism
        # inside one process just fights for the same cores. Horizontal
        # scaling is via replicas.
        for t in targets:
            results[t] = await loop.run_in_executor(None, _run_one, t)
        return results
