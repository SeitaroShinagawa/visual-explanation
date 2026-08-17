# CoDi-2 図解 — トピック概要と作成プロセスの記録

## このトピックについて

[CoDi-2](https://github.com/microsoft/i-Code/tree/main/CoDi-2)
(*In-Context, Interleaved, and Interactive Any-to-Any Generation*, Tang+, CVPR 2024)は、
ImageBind・LLaMA(Vicuna)+ LoRA・ImageBind 条件版 Stable Diffusion 2.1 unCLIP を
**2 つの小さな MLP** で接着したマルチモーダル LLM。本解説は、その **学習(2 種類の損失)と
生成(テキスト=離散トークン / 非テキスト=連続ベクトル)のプロセス**を、
実装コードの引用とアニメーション図解で読み解く 4 ページ構成の日本語解説である。

| ページ | 内容 |
|---|---|
| `index.html` | 概要・全体アーキテクチャのアニメ図・CoDi-1 / NExT-GPT との比較・章立て |
| `architecture.html` | 改造 ImageBind / 対称な 2 MLP / LoRA / unCLIP をテンソル形状を追って詳解 |
| `training.html` | ラベルの二重構造、CE 損失・コサイン回帰損失・拡散逆伝播の数式⇔コード対応 |
| `inference.html` | `generate()` の追跡(`<p>` 検出 → 隠れ状態 → c_adm → DDIM) |

- 対象コード: 公式リポジトリのコミット [`9b16730`](https://github.com/microsoft/i-Code/tree/9b16730b7e0254f02a833281335c91ee382552e6/CoDi-2)(MIT License)
- 設計方針(作成時の選択): 言語=日本語 / 深さ=コード+数式対応 / 図=CSS 自動ループ+ステップ操作
  / スタイルは NExT-GPT トピックの `assets/style.css` をそのまま複製(トピックを自己完結させる方針)

## このトピック固有の読みどころ(調査で見つかった実装の癖)

1. **`<p>` は語彙追加ではない** — 生成の合図に使われる `[529, 29886, 29958]` は
   LLaMA トークナイザで `▁<` + `p` + `>`、つまり既存語彙で書ける文字列 `<p>`。
   NExT-GPT のように `<image_00>` を語彙へ追加する方式とは対照的で、
   `embed_tokens` / `lm_head` は一切変更されない。
   (検証: `sentencepiece` で LLaMA の `tokenizer.model` を読み、`sp.encode('<p>')` が
   ちょうど `[529, 29886, 29958]` を返すことを確認した。停止トークン `2277` は `##`。)
2. **ImageBind が改造されている** — `imagebind_model.py` の VISION ヘッドで
   `SelectElement(index=0)` がコメントアウトされており、画像だけ `(L, 1024)` の
   パッチ列で返る。AUDIO 側は生きているので `(1, 1024)`。
   これが `perception_len` が画像だけ可変な理由。
3. **同じ位置に 2 種類のラベル** — `prompt_wrap` は特徴位置に回帰教師を書き込み、
   同じ位置の `target_ids` を `-100` にして CE から外す。
   「トークンを当てる/ベクトルを当てる」の住み分けがラベル生成だけで表現されている。
4. **損失は L2 ではなくコサイン** — `feature_loss = -cosine_sim ... * 0.1`。
   拡散側が受け取るのは方向情報が本質の画像埋め込み条件なので、スケールを合わせる必要がない。
5. **CFG が画像埋め込みだけに効く** — `c` と `uc` で `c_crossattn`(ネガティブプロンプト)は
   共通、違うのは `c_adm` が本物か `zeros_like` かだけ。差分を取るとテキスト側は相殺される。
6. **公開コードは推論グルーが一部欠けている** — `core/datasets/`(`FEATURE_ID` の定義元)が存在せず、
   `self.codi` は `__init__` で代入されない、`inference()` が呼ぶ `get_llm_outputs` は未定義、
   `generate_image` は `features` に先頭次元 1 を期待する、など。
   本解説では該当箇所を ⚠️ 付きの callout で明示した。

## 作成プロセス(他トピック作成時のテンプレート)

NExT-GPT トピックの手順([NExT-GPT/README.md](../NExT-GPT/README.md))を踏襲。差分のみ記す。

### 1. 調査

1. 対象リポジトリを**特定コミットで clone**(今回は `--sparse` で `CoDi-2/` のみ取得)し、
   以降の引用・行番号はすべてこのクローンから転記する
2. エントリポイントから読む。今回は学習スクリプトが存在しないため
   **`forward`(損失)→ `__init__`(構成)→ `generate`(推論)→ configs(生成器の正体)** の順に読んだ
3. 「マジックナンバー」は必ず裏を取る。今回は
   - トークン ID `[529, 29886, 29958]` → `sentencepiece` で実際にデコード
   - `adm_in_channels: 2048` → `1024(特徴)+ 1024(noise_level_emb)` の足し算であることをコードで確認
4. **ベンダリングされた依存ライブラリの差分を疑う**。ImageBind の改造はここで見つかった
   (`grep -n "SelectElement" imagebind/models/imagebind_model.py`)

### 2. ページ作成

- 構成テンプレは NExT-GPT と同一(4 ページ+`assets/style.css`+`assets/diagram-controls.js`)
- コード引用は `.codeblock`(パス+行番号+コミット固定パーマリンク付きヘッダ)で統一
- **SVG の中では MathJax が効かない**ので、図の中の数式はプレーンテキストで書く
  (`L_text` / `ε̂ = ε_u + 7.5·(ε_c − ε_u)` のように Unicode で表現する)
- コードブロック内の `<` は `&lt;` にエスケープする(`>` は不要だが `<` は必須)

### 3. 品質チェックリスト(検証は Playwright + 内蔵 Chromium)

- [ ] 全ページのフルページスクリーンショットでレイアウト・図・数式・ハイライトを目視確認
- [ ] `prefers-reduced-motion: reduce` エミュレーションで各図が**意味の通る静的フレーム**になるか
- [ ] モバイル幅(390px)で `document.scrollWidth <= clientWidth` を全ページ確認
- [ ] リンク切れ・`id` 重複がないか
- [ ] 引用コードがクローンの実ファイルと一致するか(行番号のずれを含めて機械照合)

### 4. 公開

- リポジトリ直下 `index.html` の解説一覧にカードを追加(リンクは `<topic>/index.html` と明示)
- ルート `README.md` の収録コンテンツ表に 1 行追加
