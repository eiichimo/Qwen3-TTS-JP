# Playbook Adoption Record

## Source
- Repository: `engineering-playbook`
- URL: `https://github.com/eiichimo/engineering-playbook`
- Adopted commit SHA: `66d009cd13dae5e134101fe2a7f53b7703d7212e`

## Adopted Scope
- `docs/dev-workflow.md`
- `docs/github-workflow.md`
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/ISSUE_TEMPLATE/feature.md`
- `.github/ISSUE_TEMPLATE/bug.md`
- `.github/ISSUE_TEMPLATE/chore.md`
- `.github/workflows/tests.yml`
- `AGENTS.md`

## Repository-specific Adjustments
- `develop` 開発統合 / `main` リリース運用に合わせ、`docs/dev-workflow.md` / `docs/github-workflow.md` / `.github/PULL_REQUEST_TEMPLATE.md` を調整
- CI テストコマンドは Poetry 前提を外し、`python -m unittest` ベースへ調整
  - `tests/` が未作成の期間はスキップする
- 個人用 fork 運用に合わせ、fork 元への PR 非依存（`origin` 内運用）を明記

## Follow-ups
- GitHub 側で `main` / `develop` の branch protection と `Automatically delete head branches` の設定を確認する
