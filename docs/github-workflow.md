# GitHub Workflow Guide

このドキュメントは、以下の運用をチームで安定して回すためのテンプレートです。

- `main` = リリース（安定版）
- `develop` = 開発統合
- `feature/*` = 個別開発
- 通常PRは原則 `Squash and merge`

## Current Repository Mode (2026-02-28)

- 現在は `develop` を開発統合、`main` をリリース用として運用している
- 通常の feature PR は `feature/* -> develop` で運用する
- リリース反映は `release/* -> main` で運用する
- 当面は個人用 fork 運用とし、fork 元への PR 作成は前提にしない
- PR の作成先は自分の `origin` リポジトリ内に限定する

## 1. 運用の前提

- `main` への直接コミットは禁止（PR 経由のみ）。
- `origin/main` への反映も必ず PR 経由とし、direct push は行わない。
- 作業ブランチは `develop` から作成する。
- Issue と PR を必ず紐付ける。
- 通常の feature PR では `Refs #<issue番号>` を使い、`Fixes/Closes` は使わない。
- Issue は「実装完了」と「リリース完了」を分けて管理する。
- `upstream` remote は任意の同期用途に限定し、変更の push は `origin` に対して行う。

## 2. ブランチ戦略

| ブランチ | 役割 | 更新方法 |
|---|---|---|
| `main` | リリース用の安定ブランチ | `release/* -> main` のPRのみ取り込む |
| `develop` | 開発統合ブランチ | `feature/* -> develop` を取り込む |
| `feature/*` | 機能/修正ごとの作業ブランチ | `develop` から作成 |
| `release/*` | リリース準備ブランチ | `develop` から作成し `main` へPR後に削除 |

### マージ方式

- `feature -> develop`: `Squash and merge` を基本
- `release/* -> main`（リリース PR）: `Create a merge commit` 推奨（Squash でも可）

## 3. Issue と PR の紐付けルール

### feature PR（`feature -> develop`）

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
2. `feature/<issue番号>-<short-description>` を `develop` から作成
3. 実装・コミット・push
4. feature PR 作成（`feature -> develop`。`Refs #` を記載、`Fixes` は使わない）
5. レビュー / セルフレビュー
6. `Squash and merge` で `develop` へ取り込み
7. Issue ラベルを `status: in develop` へ更新
8. `develop` から `release/*` を作成し、`release/* -> main` のリリース PR を作成
9. `main` に入るリリース PR に `Fixes #` を列挙し、自動クローズ
10. 必要に応じてタグ付与・リリースノート更新・Issue 整理

## 6. リポジトリ設定の推奨

- Branch protection:
  - `main`: PR 必須、直接 push 禁止、削除禁止
  - `develop`: PR 必須、直接 push 禁止、削除禁止
- Merge options:
  - `Squash merge` を有効化
  - `Merge commit` を有効化（リリース PR 用）
- `Automatically delete head branches` を有効化
- fork 先で Discussions を無効化している場合、関連 workflow から `discussion` / `discussion_comment` event を外す
- 補足: 自動削除の対象は `feature/*` と `release/*`。`main`（必要に応じて `develop`）は保護ルールで残す
