# GitHub Workflow Guide

このドキュメントは、以下の運用をチームで安定して回すためのテンプレートです。

- `main` = リリース（安定版）
- `develop` = 開発統合（導入時）
- `feature/*` = 個別開発
- 通常PRは原則 `Squash and merge`

## Current Repository Mode (2026-02-28)

- 現在は `main` 中心で運用している
- `develop` / `release/*` は必要時に導入する
- `develop` 未導入期間の feature PR は `feature/* -> main` で運用し、Issue 紐付けは `Refs #xx` を使う

## 1. 運用の前提

- `main` への直接コミットは禁止（PR 経由のみ）。
- 作業ブランチは現在 `main` から作成する（`develop` 導入後は `develop`）。
- Issue と PR を必ず紐付ける。
- 通常の feature PR では `Refs #<issue番号>` を使い、`Fixes/Closes` は使わない。
- Issue は「実装完了」と「リリース完了」を分けて管理する。

## 2. ブランチ戦略

| ブランチ | 役割 | 更新方法 |
|---|---|---|
| `main` | リリース用の安定ブランチ | 現在は feature PR を直接取り込む / 標準運用は `release/* -> main` |
| `develop` | 開発統合ブランチ | 導入時に `feature/* -> develop` を取り込む |
| `feature/*` | 機能/修正ごとの作業ブランチ | 現在は `main`、標準運用では `develop` から作成 |
| `release/*` | リリース準備ブランチ | `develop` 導入後に `develop` から作成し `main` へPR後に削除 |

### マージ方式

- `feature -> main`（現行）: `Squash and merge` を基本
- `feature -> develop`（標準）: `Squash and merge` を基本
- `release/* -> main`（リリース PR）: `Create a merge commit` 推奨（Squash でも可）

## 3. Issue と PR の紐付けルール

### feature PR（現行: `feature -> main` / 標準: `feature -> develop`）

- `Related Issues` に `Refs #123` を記載する。
- `Fixes #123` / `Closes #123` は書かない。

例:

```md
## Related Issues
- Refs #123
```

### リリース PR（`release/* -> main`）

- PR 本文に今回リリースで閉じる Issue を `Fixes #` / `Closes #` で列挙する。
- `main` 取り込み時に自動クローズさせる。

例:

```md
## Fixes
- Fixes #123
- Closes #130
```

### 例外（develop 取り込み時点で閉じる場合）

- リリース非依存のタスクは手動 Close してよい。
- Issue コメントで理由を残す。

## 4. ラベル設計（未リリース可視化）

必須:

- `status: backlog`
- `status: in progress`
- `status: in develop`（develop 取り込み済み / 未リリース）
- `status: released`（任意）

任意:

- `type: bug` / `type: feature` / `type: chore`
- `priority: high` / `priority: medium` / `priority: low`
- `area: ui` / `area: cli` / `area: core` / `area: docs`

状態遷移の目安:

`status: backlog` -> `status: in progress` -> `status: in develop` -> `status: released`

## 5. 運用フロー（1〜10）

1. Issue 作成（`status: backlog` を付与）
2. `feature/<issue番号>-<short-description>` を現在は `main` から作成（標準運用では `develop`）
3. 実装・コミット・push
4. feature PR 作成（現行は `feature -> main` / 標準は `feature -> develop`。`Refs #` を記載、`Fixes` は使わない）
5. レビュー / セルフレビュー
6. `Squash and merge` で取り込み（現行は `main`、標準は `develop`）
7. 標準運用時は Issue ラベルを `status: in develop` へ更新
8. 標準運用時は `develop` から `release/*` を作成し、`release/* -> main` のリリース PR を作成
9. `main` に入るリリース PR に `Fixes #` を列挙し、自動クローズ
10. 必要に応じてタグ付与・リリースノート更新・Issue 整理

## 6. リポジトリ設定の推奨

- Branch protection:
  - `main`: PR 必須、直接 push 禁止、削除禁止
  - `develop`: 導入時は PR 運用、削除禁止
- Merge options:
  - `Squash merge` を有効化
  - `Merge commit` を有効化（リリース PR 用）
- `Automatically delete head branches` を有効化
- fork 先で Discussions を無効化している場合、関連 workflow から `discussion` / `discussion_comment` event を外す
- 補足: 自動削除の対象は `feature/*` と `release/*`。`main`（必要に応じて `develop`）は保護ルールで残す
