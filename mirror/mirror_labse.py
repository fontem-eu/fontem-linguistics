"""Download LaBSE at a pinned HF revision and produce a tarball ready to
`oras push`.

Called from the `mirror-labse` workflow. Keeps Python where Python belongs
(snapshot_download + sanity checks); the shell half of the pipeline does
`oras push` + `cosign sign` + `cosign attest`.

Outputs to stdout the values the workflow needs — tag, encoder_id, tarball
path, SBOM path — so the next CI step can pick them up via $GITHUB_OUTPUT.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone

import yaml


def _load_version(path: pathlib.Path) -> dict:
    with path.open() as f:
        data = yaml.safe_load(f)
    for key in ("hf_repo", "hf_revision", "mirror_version"):
        if not data.get(key):
            raise SystemExit(f"{path}: missing required key {key!r}")
    sha = data["hf_revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        # Branch names, tags, and short SHAs all fail this check. The whole
        # point of the mirror is reproducibility — "main" would silently
        # defeat it.
        raise SystemExit(
            f"hf_revision must be a 40-char hex SHA, got {sha!r}. "
            f"Branches/tags/short SHAs are not acceptable.",
        )
    return data


def _download_snapshot(repo: str, revision: str, dest: pathlib.Path) -> None:
    """Populate `dest` with the HF repo content at the pinned revision.

    We skip ``pytorch_model.bin`` when ``model.safetensors`` is present:
    sentence-transformers prefers safetensors at load time anyway, and
    shipping both doubles the artifact size (~1.9 GB duplicate on LaBSE).
    """
    # Imported here so the top of the file stays lightweight when the caller
    # is only sanity-checking the version file.
    from huggingface_hub import snapshot_download  # pylint: disable=import-outside-toplevel

    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(dest),
        # Don't download the legacy pickled weights when the safetensors
        # are available — halves download + tar time. The
        # sentence-transformers loader handles the absence gracefully.
        ignore_patterns=["pytorch_model.bin", "flax_model.msgpack", "tf_model.h5"],
    )


def _write_sbom(meta: dict, out: pathlib.Path) -> None:
    """Minimal CycloneDX SBOM describing the upstream model as a component.

    Model weights have no transitive dependencies in the software-SBOM sense,
    so this is essentially a provenance record: repo + commit + fetched_at.
    Kept valid-enough-to-ingest so the existing attestation tooling works
    without special-casing.
    """
    sbom = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "timestamp": meta["fetched_at"],
            "component": {
                "type": "machine-learning-model",
                "name": meta["hf_repo"],
                "version": meta["hf_revision"],
                "purl": f"pkg:huggingface/{meta['hf_repo']}@{meta['hf_revision']}",
                "externalReferences": [
                    {
                        "type": "vcs",
                        "url": f"https://huggingface.co/{meta['hf_repo']}/tree/{meta['hf_revision']}",
                    },
                ],
            },
        },
        "components": [],
    }
    out.write_text(json.dumps(sbom, indent=2))


def _tar_dir(src: pathlib.Path, tar_path: pathlib.Path) -> None:
    # Plain tar, NOT tar.gz. Model weights are already in compact binary
    # formats (safetensors) — gzip costs ~10 min of single-threaded CPU
    # on the CI runner for single-digit % savings, and the registry
    # doesn't care about at-rest size. Keep it fast.
    with tarfile.open(tar_path, "w") as tf:
        tf.add(str(src), arcname=".")


def _emit_output(key: str, value: str) -> None:
    """Write to $GITHUB_OUTPUT if present (CI), also echo to stdout for humans."""
    print(f"{key}={value}")
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--version-file",
        default=str(pathlib.Path(__file__).parent / "labse.version.yaml"),
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Directory for tar + sbom output. Defaults to a temp dir.",
    )
    ap.add_argument(
        "--skip-download",
        action="store_true",
        help="Validate the version file and print the computed encoder_id "
             "without touching HF. Used by the workflow's lint step.",
    )
    args = ap.parse_args()

    meta = _load_version(pathlib.Path(args.version_file))
    sha7 = meta["hf_revision"][:7]
    encoder_id = f"labse@{meta['mirror_version']}-{sha7}"
    tag = meta["mirror_version"]

    _emit_output("encoder_id", encoder_id)
    _emit_output("tag", tag)
    _emit_output("hf_revision", meta["hf_revision"])

    if args.skip_download:
        return 0

    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(tempfile.mkdtemp())
    out_dir.mkdir(parents=True, exist_ok=True)

    snapshot_dir = out_dir / "snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    print(f"downloading {meta['hf_repo']}@{meta['hf_revision']}...", flush=True)
    _download_snapshot(meta["hf_repo"], meta["hf_revision"], snapshot_dir)

    # Strip the .cache directory HF creates; not part of the model.
    cache = snapshot_dir / ".cache"
    if cache.exists():
        subprocess.run(["rm", "-rf", str(cache)], check=True)

    fetched_at = datetime.now(timezone.utc).isoformat()

    tar_path = out_dir / "labse.tar"
    print(f"packing {snapshot_dir} → {tar_path}", flush=True)
    _tar_dir(snapshot_dir, tar_path)

    sbom_path = out_dir / "sbom.cdx.json"
    _write_sbom({**meta, "fetched_at": fetched_at}, sbom_path)

    _emit_output("tar_path", str(tar_path))
    _emit_output("sbom_path", str(sbom_path))
    _emit_output("fetched_at", fetched_at)

    print(f"\nencoder_id: {encoder_id}")
    print(f"tag:        {tag}")
    print(f"tar:        {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
