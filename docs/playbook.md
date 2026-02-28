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
- `main` 中心運用でも適用できるよう、`docs/dev-workflow.md` / `docs/github-workflow.md` / `.github/PULL_REQUEST_TEMPLATE.md` に暫定ルールを追記
- CI テストコマンドは Poetry 前提を外し、`python -m unittest` ベースへ調整
  - `tests/` が未作成の期間はスキップする

## Follow-ups
- `develop` / `release/*` 運用に切り替える場合は、同ドキュメント内の標準運用ルールを有効化する
- GitHub 側で `Automatically delete head branches` と branch protection の設定を確認する
