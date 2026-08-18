# BAGEL 図解 — トピック概要と作成プロセスの記録

## このトピックについて

[BAGEL](https://github.com/ByteDance-Seed/Bagel)
(*Emerging Properties in Unified Multimodal Pretraining*, Deng+, 2025 / arXiv:2505.14683)は、
ByteDance Seed による **7B activated / 14B total の統一マルチモーダル基盤モデル**。
本シリーズで先に扱った NExT-GPT / CoDi-2 が「凍結 LLM + 凍結拡散モデルを射影層で接着する」構図だったのに対し、
BAGEL は**外部の拡散モデルを一切持たず、LLM 本体が Rectified Flow のデノイザとして働く**。
本解説は、その **学習(CE と MSE の 2 本立て)と生成(自己回帰ループと ODE 積分ループの併存)** を、
実装コードの引用とアニメーション図解で読み解く 4 ページ構成の日本語解説である。

| ページ | 内容 |
|---|---|
| `index.html` | 概要・全体アーキテクチャのアニメ図・NExT-GPT / CoDi-2 との系譜比較・章立て |
| `architecture.html` | 二重視覚エンコーダ(ViT/VAE)/ MoT の 2 エキスパート / Generalized Causal Attention をテンソル形状を追って詳解 |
| `training.html` | 4 ステージ学習と論文 Table 3 のハイパーパラメータ、データ構成と採択比率、`sequence_plan` → パッキング、Rectified Flow の MSE と テキスト CE、CFG ドロップアウト、Diffusion Forcing、推論拡張データ |
| `inference.html` | `generate_text` と `generate_image` の 2 ループ、KV キャッシュ再利用、二重 CFG と CFG-Renorm、think モード |

- 対象コード: 公式リポジトリのコミット [`a2fa77d`](https://github.com/ByteDance-Seed/Bagel/tree/a2fa77dd8caeefc41e6607ae0ec17408d3f4ee9f)(Apache-2.0)
- 参照した一次資料: 上記コード全体(`modeling/bagel/`, `data/`, `train/`, `inferencer.py`, `app.py`, `TRAIN.md`, `README.md`)/
  論文 [arXiv:2505.14683v3](https://arxiv.org/abs/2505.14683)(2025-07-27)本文
- 設計方針(作成時の選択): 言語=日本語 / 深さ=コード+数式対応 / 図=CSS 自動ループ+ステップ操作
  / スタイルは CoDi-2 トピックの `assets/style.css` をそのまま複製(トピックを自己完結させる方針)

## 論文とコードで食い違う点(両論併記した箇所)

本シリーズの方針どおり、どちらかに寄せず両方をページ上に明記した。

| 項目 | 論文 | 公開コードの既定値 | 記載場所 |
|---|---|---|---|
| 損失の重み | CE : MSE = **0.25 : 1**(Table 3) | `ce_weight = mse_weight = 1.0` | `index.html` / `training.html#losses` |
| CFG ドロップアウト率 | text 0.1 / **ViT 0.5** / **clean VAE 0.1**(§2.3) | text 0.1 / ViT 0.3 / VAE 0.3 | `training.html#cfgdrop` |

いずれも公開コードが「学習レシピそのもの」ではなく雛形として提供されていることに由来すると思われる。
とくに CFG ドロップアウトは、論文設定では ViT を落とす確率が VAE の 5 倍で、
「意味は無視するが画素は保つ」状況を重点的に経験させる設計になっており、
コードの 0.3 / 0.3(対等)とは意味合いが異なる。

## 論文がコードの「謎挙動」を説明してくれた箇所

調査中にコードだけ読んで疑問符をつけていた 2 点は、論文 §2.3 を読むと**仕様どおり**だと判明した。
コードの `split` が、論文でいう **group** に対応する。

- **動画 split の attn モードが `'noise'` ではなく `'full'`** → “apply **full attention** within each group”。
  後続フレームは先行フレームの**ノイジーな表現を条件にする**(diffusion forcing)ので、隠してはいけない
- **同じ split 内のフレームが同じ timestep を共有する** → “**The noise level is the same inside each group**”

論文の “adds independent noise levels to different images” と併せて読むと、
**独立なのは group 間、共有されるのは group 内**という設計であることが分かる。
group サイズが「1 枚ずつ独立」と「全部同じ」の中間を調整するつまみになっている。
なお論文は “**randomly** group consecutive images” と書くが、
公開されている `_add_video` は 1 クリップを丸ごと 1 group にするため、
ランダムなグループ分割そのものはリリースされたデータコードには含まれていない。

## このトピック固有の読みどころ(調査で見つかった実装の癖)

1. **`llm2vae` はゼロ初期化されている** — `Bagel._init_weights` で weight/bias とも 0。
   学習開始時の速度場予測は恒等的に 0 で、「何もしない」状態から立ち上がる(DiT 系の作法)。
2. **画像 1 枚が占める RoPE 位置はたった 1** — `packed_position_ids` には
   `[curr_rope_id] * (num_img_tokens + 2)` が入る。画像内の位置関係は
   RoPE ではなく足し算される 2D sin-cos テーブル(`latent_pos_embed` / `vit_pos_embed`)が担当する。
   結果、4096 トークンの画像を挟んでもテキスト側の位置は 1 しか進まない。
3. **数値が噛み合っている** — ViT は 980px / patch 14 → 70 = `vit_max_num_patch_per_side`、
   VAE は 1024px / (8×2) = 64 = `max_latent_size`。前処理 `ImageTransform(980,224,14)` と
   `ImageTransform(1024,512,16)` の stride がそのまま「1 トークンの画素数」になっている。
4. **ノイズ有無を「時刻の値」で表す** — 条件画像の timestep は `float('-inf')`。
   モデル側の `torch.sigmoid(-inf) = 0` により混合率 0(=clean)になる。
   条件画像も生成対象も同じ 1 本の補間式を通る。
5. **`has_mse` と `mse_loss_indexes` は切り出し方が違う** — 予測は `mse_loss_indexes`、
   教師は `packed_timesteps > 0`。両者の一致はデータ側の不変条件
   (`loss==1` のときだけ `randn()`、それ以外は `-inf`)に依存している。自作データセットでは要注意。
6. **動画 split は `'noise'` ではなく `'full'` になり、split 内で timestep を共有する** —
   上記「論文が説明してくれた箇所」を参照。コードの `split` = 論文の `group`。
7. **`special_token_loss` は公開データコードで一度も 1 にならない** — パッカー側は
   `<|im_end|>` / `<|vision_end|>` にも CE を置ける作りだが、4 つのデータセット実装はすべて 0 固定。
   「いつ画像を出すか」をモデルに学ばせるフックが用意されているが未使用で、
   推論側も呼び出し側フラグ(`understanding_output`)で分岐する設計と整合している。
8. **CFG 文脈の作り分けが `deepcopy` のタイミングだけで表現されている** —
   `cfg_text_context` はテキストを足す**直前**のスナップショット、
   `cfg_img_context` には画像を**足さない**。差分を取ると各条件の寄与だけが残る。
   think モードで生成した思考は `gen_context` にだけ足されるので、思考内容も CFG の強調対象になる。
9. **CFG-Renorm は縮めることしかしない** — `clamp(min=cfg_renorm_min, max=1.0)`。
   `global` はテンソル全体で 1 つのノルムを取るため、コード中のコメントどおり実質バッチ 1 前提。
10. **`cfg_interval` は速度にも効く** — 区間外では scale が 1.0 になり、
    `if cfg_text_scale > 1.0` が偽になって追加の forward がスキップされる(3 回 → 1 回)。
11. **テキスト生成中は Gen Expert が一度も動かない** — `generate_text` は `mode="und"` 固定。
    「14B のうち 7B が活性化」という表現の実体。
12. **`vit_config.num_hidden_layers -= 1`** — `app.py` が SigLIP の最終層を 1 枚落として読み込む。
13. **CE:MSE = 0.25:1 には理由がある** — 論文 §4.2 のアブレーションで
    「学習率を上げると MSE は速く収束するが CE は悪化する」という綱引きが観測され、
    学習率を一本化したまま損失側の重みで折り合いをつけた結果。
14. **AdamW の `eps` が 1e-15** — 損失スパイクを抑えるため(論文 §4)。通常の 1e-8 より 7 桁小さい。
15. **PT/CT/SFT の学習率スケジューラが Constant** — 「学習を再開せずにデータを足せるように」という理由が
    論文に明記されている。実運用を意識した選択。
16. **CT で timestep shift を 1.0 → 4.0 に上げる** — 解像度を上げると
    適切なノイズレベル分布が変わるための調整。推論側の `timestep_shift`(既定 3.0)と同じ式。
17. **emerging の定義が「損失に現れない」ことを前提にしている** — だから論文は
    過去チェックポイントを掘り起こしてベンチマークを回し直すという方法を採っている。
    85% 到達トークン数は 理解 0.18T / 生成 0.68T / 編集 2.64T / 知的編集 3.61T。

## 作成プロセス(他トピック作成時のテンプレート)

NExT-GPT / CoDi-2 の手順([CoDi-2/README.md](../CoDi-2/README.md))を踏襲。差分のみ記す。

### 1. 調査

1. 対象リポジトリを**特定コミットで clone**し、以降の引用・行番号はすべてこのクローンから転記する
2. 今回は学習コードが公開されているので、**`Bagel.forward`(損失)→ `pack_sequence`(データ→列)→
   `Qwen2MoTDecoderLayer`(MoT の中身)→ `generate_image`(推論)** の順に読んだ。
   「損失がどのテンソルのどのインデックスに掛かるか」を先に押さえると、
   データ側の `sequence_plan` が何のために存在するのかが分かる
3. **マスク生成関数を最優先で読む**。統一モデルでは attention マスクが設計思想そのもの。
   今回は `create_sparse_mask` と `prepare_attention_mask_per_sample` の 2 実装を突き合わせ、
   後者(素朴な行列版)を図解の下敷きにした
4. **マスク図は手で描かず生成する** — 12×12 のマスク行列は Python で実際に構成し、
   その結果を SVG の `<rect>` 群として出力してページに埋めた(目視で描くと必ず間違える)
5. **コードだけで説明できない挙動には印をつけて保留し、論文で答え合わせする**。
   今回は作業環境の egress 制限で arXiv に到達できず、初稿はコード由来の記述と
   二次情報由来の記述を分けて書いた。その後に論文本文を入手して全面的に突き合わせたところ、
   保留していた 2 件(動画 split の attn モード / timestep 共有)は**仕様どおり**と判明し、
   逆に**新たに 2 件の論文⇔コード不一致**(損失重み・CFG ドロップアウト率)が見つかった。
   「分からないことを分からないまま書いておく」と、後から差分だけを直せる

### 2. ページ作成

- 構成テンプレは NExT-GPT / CoDi-2 と同一(4 ページ+`assets/style.css`+`assets/diagram-controls.js`)
- コード引用は `.codeblock`(パス+行番号+コミット固定パーマリンク付きヘッダ)で統一
- **引用の行番号は機械照合する** — 引用ブロック内の各行がクローンの実ファイルに存在するかを
  スクリプトで確認し、`...` による省略以外の差異を潰した
- **SVG の中では MathJax が効かない**ので、図の中の数式はプレーンテキストで書く
- コードブロック内の `<`・`>`・`&` はエスケープする(Python の比較演算子 `>=` や `->` が頻出するので特に注意)
- 本トピックは**シリーズ 3 作目**なので、前 2 作との比較を index と最終ページの両方に置き、
  「接着 → 統合」という縦の流れが読めるようにした

### 3. 品質チェックリスト(検証は Playwright + 内蔵 Chromium)

- [x] 全ページのフルページスクリーンショットでレイアウト・図・数式・ハイライトを目視確認
- [x] `prefers-reduced-motion: reduce` エミュレーションで各図が**意味の通る静的フレーム**になるか
- [x] モバイル幅(390px)で `document.scrollWidth <= clientWidth` を全ページ確認
- [x] リンク切れ・`id` 重複がないか
- [x] 引用コードがクローンの実ファイルと一致するか(行番号のずれを含めて機械照合)

### 4. 公開

- リポジトリ直下 `index.html` の解説一覧にカードを追加(リンクは `<topic>/index.html` と明示)
- ルート `README.md` の収録コンテンツ表に 1 行追加
