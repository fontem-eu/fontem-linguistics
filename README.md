> ### 🪞 This GitHub repository is a mirror
>
> Development happens on Fontem's own infrastructure; this mirror is
> updated automatically. **Issues and pull requests opened here are not
> monitored.**
>
> If you would like to contribute — code, data sources, review, or
> anything else — please get in touch at **team@fontem.eu** and we will
> set you up.

# fontem-linguistics

Translation + embedding service. NLLB-200 distilled (translator) + LaBSE (sentence embeddings) running on cluster. Used by the consolidator for cross-language entity matching and by the API for translated authority names.

## Deploy

CI auto-deploys to the testing env on every merge to main. Promotion to staging / prod is **manual** — bump the version in `gitops/<env>/<service>.yaml` to land it in a given environment.

## Convention

See [/config/repos/CLAUDE.md](https://contribute.void42.internal/fontem/gitops) for workspace-wide rules (feature branches + CI gate, no direct push to main, full gate before declaring done, conventional commits).

## License

Apache License 2.0 — see [LICENSE](LICENSE).
