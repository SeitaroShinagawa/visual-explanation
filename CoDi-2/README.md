# CoDi-2 図解 — トピック概要と作成プロセスの記録

## このトピックについて

[CoDi-2](https://github.com/microsoft/i-Code/tree/main/CoDi-2)
(*In-Context, Interleaved, and Interactive Any-to-Any Generation*, Tang+, CVPR 2024)は、
ImageBind・Llama-2-7b-chat + LoRA・拡散モデル群(Stable Diffusion 2.1 unCLIP / AudioLDM2 / zeroscope v2)を
**2 つの小さな MLP** で接着したマルチモーダル LLM。本解説は、その **学習(3 項の損失)と
生成(テキスト=離散トークン / 非テキスト=連続ベクトル)のプロセス**を、
実装コードの引用とアニメーション図解で読み解く 4 ページ構成の日本語解説である。

| ページ | 内容 |
|---|---|
| `index.html` | 概要・全体アーキテクチャのアニメ図・CoDi-1 / NExT-GPT との比較・章立て |
| `architecture.html` | 改造 ImageBind / 対称な 2 MLP / LoRA / unCLIP をテンソル形状を追って詳解 |
| `training.html` | ラベルの二重構造、論文 §3.2 の損失式とコードの 1 対 1 対応、学習パイプラインとデータ構築 |
| `inference.html` | `generate()` の追跡(`<p>` 検出 → 隠れ状態 → c_adm → DDIM) |

- 対象コード: 公式リポジトリのコミット [`9b16730`](https://github.com/microsoft/i-Code/tree/9b16730b7e0254f02a833281335c91ee382552e6/CoDi-2)(MIT License)
- 参照した一次資料: 上記コード / 論文 [arXiv:2311.18775v1](https://arxiv.org/abs/2311.18775)(§3 Model Architecture、§4 データ構築、§5.1 Model Setups、§B 補遺)/
  [プロジェクトページ](https://codi-2.github.io/) のアーキテクチャ図
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
4. **論文の損失式とコードが 1 対 1 で対応する** — 論文 §3.2 の
   `L = α·MSE(c_MLLM, C_x(x)) + L_DM + L_t` は、コードの
   `feature_loss` / `diffusion_loss` / `outputs.loss` にそのまま対応し、`α` は `0.1`。
   ただし**論文は MSE と書くのに対し、コードの `'clip'` 主経路は負のコサイン類似度**
   (literal な MSE は `'vae'` 経路のみ)。この不一致はページ上に明示した。
5. **論文とコードで食い違う点が 2 つある** — ①LLM は論文が `Llama-2-7b-chat-hf` と明記する一方、
   コードの引数名は `vicuna_ckpt_path`(PandaGPT/NExT-GPT 系の名残)。
   ②音声・動画の生成器は論文では AudioLDM2 / zeroscope v2 576w だが、
   公開コードに残る `decode()` は CoDi-1 の VAE を呼ぶ形。どちらも両論併記した。
6. **CFG が画像埋め込みだけに効く** — `c` と `uc` で `c_crossattn`(ネガティブプロンプト)は
   共通、違うのは `c_adm` が本物か `zeros_like` かだけ。差分を取るとテキスト側は相殺される。
   コードから読み取ったこの構造は、論文 §B.1 の
   "we employ negative prompts as cross-attention conditions and utilize MLLM-generated
   features for embedding guidance" で裏付けられた。
7. **1 バッチ 1 モダリティなのは意図的** — `forward` の `if 'image' / elif 'text' / elif 'audio'` は、
   論文 §B.2 の「テキスト・音声・画像の生成フェーズを交互に切り替えて学習する」方針の実装。
8. **公開コードは推論グルーが一部欠けている** — `core/datasets/`(`FEATURE_ID` の定義元)が存在せず、
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
- **論文とコードが食い違う箇所は、どちらかに寄せず両方を書く**(本トピックでは LLM 名・音声/動画デコーダ・MSE vs コサインの 3 件)
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
