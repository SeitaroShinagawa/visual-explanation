# NExT-GPT 図解 — トピック概要と作成プロセスの記録

## このトピックについて

[NExT-GPT](https://github.com/NExT-GPT/NExT-GPT)(Wu+, ICML 2024 Oral)は、
ImageBind・Vicuna-7B・拡散モデル群(Stable Diffusion 2 / ZeroScope / AudioLDM)を
小さなプロジェクタで「接着」し、テキスト・画像・動画・音声を任意の組み合わせで
入出力できる Any-to-Any マルチモーダル LLM。本解説は、その**学習(3ステージ)と
生成(シグナルトークン→拡散モデル)のプロセス**を、実装コードの引用と
アニメーション図解で読み解く 4 ページ構成の日本語解説である。

| ページ | 内容 |
|---|---|
| `index.html` | 概要・全体アーキテクチャのアニメ図・章立て |
| `architecture.html` | 5 つの構成部品をテンソル形状を追いながら詳解 |
| `training.html` | 3 ステージ学習、損失の数式⇔コード対応 |
| `inference.html` | `generate()` の追跡(シグナルトークン検出→拡散生成) |

- 対象コード: 公式リポジトリのコミット [`60d618b`](https://github.com/NExT-GPT/NExT-GPT/tree/60d618b067ee4cb0d70e7075ae79852780b34fc2)(現行 `nextgpt/` コードベース。旧実装 `NExT-GPT-Lagacy/` は対象外)
- 設計方針(作成時の選択): 言語=日本語 / 深さ=コード+論文数式対応 / 図=CSS 自動ループ+ステップ操作

## 作成プロセス(他トピック作成時のテンプレート)

### 1. 調査

1. 対象リポジトリを**特定コミットで clone** し、以降の引用・行番号はすべてこのクローンから転記する
   (後述のパーマリンクにもこのコミットハッシュを使う)
2. エントリポイントから読む: 学習スクリプト(引数フラグに設計が現れる)→ モデル定義 → 損失計算 → 生成関数
3. 「解説の核」になる実装の癖をメモしておく。今回の例:
   - シグナルトークンはスペース区切りで並ぶため、コード各所に `n*2-1`(空白トークン込みの長さ)が現れる
   - `pretrain_dec.sh` は「音声用プロジェクタのみ tune=True」の状態でコミットされている(モダリティごとに差し替えて実行する想定)
   - 教師キャプション埋め込みは `preprocess_embeddings.py` で事前計算して `.npy` 保存

### 2. ページ作成

- 構成テンプレ: `概要(全体図+章立て) → 構成要素 → 学習 → 推論` の 4 ページ+`assets/style.css`
- 各ページ共通: sticky なパンくずナビ / 前後ページャ / ページ内 `<style>` に図のアニメ定義
- コード引用は `.codeblock`(ファイルパス+行番号+GitHub パーマリンク付きヘッダ)で統一
- **パーマリンク必須**: `blob/main/` ではなく `blob/<コミットハッシュ>/` を使う(行番号引用がずれないように)
- 数式は MathJax(SVG 版)、ハイライトは Prism を jsDelivr CDN から読み込み
- 図は**インライン SVG + CSS keyframes** の自動ループ。共通スクリプト
  `assets/diagram-controls.js` が `<figure class="diagram" data-cycle-ms="…" data-step-count="…">`
  を検出して「自動再生 ON/OFF」「1ステップ進む」ボタンを自動付与する
  (仕組み: `animation-play-state` の pause/resume + 1 ステップ分の時間だけ再生して自動停止)

### 3. 品質チェックリスト(検証は Playwright + 内蔵 Chromium)

- [ ] 全ページのフルページスクリーンショットでレイアウト・図・数式・ハイライトを目視確認
- [ ] `prefers-reduced-motion: reduce` エミュレーションで各図が**意味の通る静的フレーム**になるか
      (アニメを止めるだけでは基底 `opacity: 0` の要素が消える — 各ページの media query 内で静的状態を明示定義する)
- [ ] モバイル幅(390px)で `document.scrollWidth <= clientWidth` を全ページ確認
      (よくある原因: テーブル → `display:block; overflow-x:auto`、長いインラインコード → `overflow-wrap:anywhere`、
      SVG は `min-width` を与えて figure 内スクロールにする)
- [ ] リンク切れ・`id` 重複(見出しと figure の衝突に注意)がないか
- [ ] フォントスタックに `Noto Sans CJK JP` を含める(Linux 環境での日本語崩れ防止)
- [ ] 引用コードがクローンの実ファイルと一致するか、引用パスが対象コミットに存在するか(`git cat-file -e`)

### 4. 公開

- リポジトリ直下 `index.html` の解説一覧にカードを追加(リンクは `<topic>/index.html` と明示)
- GitHub Pages は main ブランチ root 配信(`.nojekyll` 設置済み)。設定は Settings → Pages → Deploy from a branch / `main` / `/ (root)`

## このトピックの作成経緯(PR [#1](https://github.com/SeitaroShinagawa/visual-explanation/pull/1))

1. 初版作成(4 ページ+トップページ、Playwright で検証)
2. Codex レビューによる修正: 日本語フォントフォールバック / reduced-motion の静的フレーム化 /
   モバイルでの SVG 可読性 / 数値の不整合(77×768→77×1024)
3. 機能追加: 明示的 `index.html` リンク、図のステップ操作コントロール(Codex 実装+動作検証)
4. コードリンクをコミット固定のパーマリンクに変更
