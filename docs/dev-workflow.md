# Development Workflow

## Current Repository Mode (2026-02-28)
- このリポジトリは現在 `main` 中心で運用している
- `develop` / `release/*` 運用は将来的に導入可能
- 本ドキュメントは「現行(main中心)」と「標準(develop+release)」の両方を扱う

## Quick Start
```bash
git switch main
git pull --ff-only origin main
git switch -c fix/my-task
```

## Sync Rules
- `git pull` の素実行は避ける
- `main` 更新は `git pull --ff-only origin main`
- `develop` 運用時は `git pull --ff-only origin develop`
- 作業ブランチ更新は `git fetch origin && git rebase origin/<base-branch>`

## PR Body
- `Summary`
- `Testing`

`gh pr create --body-file` を使う。

## Issue / PR Link Rules
- 通常PR本文には `Refs #xx` を記載する（`Fixes/Closes` は使わない）
- `main` に入るリリースPR本文に `Fixes #xx` / `Closes #xx` を記載して自動クローズする
- 未リリース可視化のため、Issue ラベルは `status: in develop` を運用する
- 詳細は `docs/github-workflow.md` を参照する

## CI Policy
- 基本は fast/full の2段構成
- docs-only PR の fast test スキップを許可
- 配布物（実行ファイル/パッケージ）を持つリポジトリでは Release 品質ゲートを追加する
  - publish/package が成功する
  - 必須アセットが同梱される
  - publish/package 成果物が起動直後に異常終了しない（スモーク起動）
  - 例（.NET）: `dotnet publish -c Release -o publish_test`

## Merge Strategy
- 現行運用（`main` 中心）では通常PRを `Squash and merge` する
- 標準運用（`develop` 導入後）では `develop` への通常PRを `Squash and merge` する
- 標準運用の `main` へのリリースPR（`develop -> main`）は `Create a merge commit` を使う
- 標準運用では `main` マージ前に `develop` へタグを付与する（例: `pre-release-YYYYMMDD`）
- Issue 自動クローズ（`Fixes/Closes #xx`）は `main` に入るPR本文へ記載する

## GitHub Branch Settings Checklist

GitHub のブランチ自動削除を有効にしつつ、`main`（必要に応じて `develop`）は削除禁止にするハイブリッド運用を採用する。

1. `Settings > General` の `Automatically delete head branches` を `ON` にする
2. `Settings > Branches` で `main` の保護ルールを作成する
3. `main` ルールで削除を許可しない（`Allow deletions` を無効）
4. `main` ルールで PR 必須（必要なら status checks も必須）を有効にする
5. `develop` を使う場合は同様に保護ルールを作成し、削除禁止と PR 必須を有効にする
6. `develop` 運用時はリリースPRを `develop -> main` ではなく `release/* -> main` で運用する
7. マージ後は `feature/*` と `release/*` のみ自動削除し、`main`（必要に応じて `develop`）は常時残す

## Branch Cleanup
- GitHub Repository Settings の `Automatically delete head branches` を有効化する
- マージ後はローカルを定期的に整理する
```bash
git fetch origin --prune
git branch -vv
```
- 追跡先が消えた不要ローカルブランチは `git branch -d <branch>` で削除する
