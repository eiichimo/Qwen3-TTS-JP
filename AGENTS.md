# AGENTS Guide

このファイルは、AI/自動化エージェント向けの運用ガイドです。

## 目的
- 変更の主導線を明確化する
- 事故を減らす
- ドキュメント/テスト/PR の一貫性を保つ

## 最初に確認するファイル
1. `README.md`
2. `docs/playbook.md`
3. `docs/dev-workflow.md`
4. `docs/github-workflow.md`
5. `.github/PULL_REQUEST_TEMPLATE.md`

## 開発方針
- 新規改修は既存の責務境界（`qwen_tts/`, `finetuning/`, `docs/`）を尊重する
- 互換レイヤは必要最小限の変更に留める
- 作業ブランチは `main`（または `develop` 運用時は `develop`）から作成する

## テスト方針
- 変更範囲に応じて最小のテストを実行する
- ドキュメントのみ変更はテスト省略可（PR の `Testing` に理由を記載）

## PR 規約
- 本文に `Summary` / `Testing` を記載する
- Issue 紐付けは `Refs #xx` を基本にする
- `Fixes/Closes #xx` は `main` に入るリリースPRで使う
