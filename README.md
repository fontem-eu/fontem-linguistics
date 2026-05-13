# fontem-linguistics

Translation + embedding service. NLLB-200 distilled (translator) + LaBSE (sentence embeddings) running on cluster. Used by the consolidator for cross-language entity matching and by the API for translated authority names.

## Deploy

CI auto-deploys to the testing env on every merge to main. Promotion to staging / prod is **manual** — bump the version in `gitops/<env>/<service>.yaml` to land it in a given environment.

## Convention

See [/config/repos/CLAUDE.md](https://contribute.void42.internal/fontem/gitops) for workspace-wide rules (feature branches + CI gate, no direct push to main, full gate before declaring done, conventional commits).
