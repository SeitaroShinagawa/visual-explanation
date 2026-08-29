# Navigation World Models 図解 — トピック概要と作成プロセスの記録

## このトピックについて

[Navigation World Models](https://github.com/facebookresearch/nwm/)
(*Navigation World Models*, Bar+, CVPR 2025 Oral / arXiv:2412.03572)は、
FAIR at Meta・NYU・BAIR による **行動条件付きの映像生成モデル(世界モデル)**。
本シリーズで先に扱った NExT-GPT / CoDi-2 / BAGEL が「生成モデルをどう賢くするか」の系譜だったのに対し、
NWM は **生成モデルを行動決定の道具として使う** —— 軌道候補をモデルの中で試し打ちし、
ゴール画像に一番近い結果になった一本を選ぶ。
本解説は、その **学習(拡散の標準的な学習)と生成・計画(自己回帰ロールアウト + CEM)** を、
実装コードの引用とアニメーション図解で読み解く 5 ページ構成の日本語解説である。

| ページ | 内容 |
|---|---|
| `index.html` | 研究背景(既存方策の 2 つの限界)/ 関連研究の 4 系譜 / 問題設定(Eq. 1–2、行動と時間シフト)/ 全体アーキテクチャのアニメ図 / 本シリーズ他モデルとの対比 / 章立て |
| `architecture.html` | テンソル形状の流れ、CDiT ブロックの 3 段構成、11 分割 AdaLN、条件ベクトル ξ の作り方、位置埋め込み、DiT との計算量比較、モデルサイズ一覧、ゼロ初期化 |
| `training.html` | データセット 6 種、行動の 4 段階正規化、1 観測 4 ゴールによる反実仮想、時間シフトの正規化(1/128)、拡散損失(論文式 vs コードの ε 予測)、ハイパーパラメータ、「無いもの」の一覧 |
| `inference.html` | 2 つの生成モード、スライディングウィンドウ、エネルギー関数(Eq. 4–5)、CEM の実装(3 次元探索)、ヨーの扱い、制約付き計画、外部方策ランキング、推論高速化 |
| `evaluation.html` | 6 指標と 3 プロトコル、評価セットの作り方、アブレーション(Table 1)、CDiT vs DiT、FVD 比較、ナビゲーション性能(Table 2/7)、Ego4D の効果(Table 4/5)、TTA、限界と議論 |

- 対象コード: 公式リポジトリのコミット [`3f6cd8e`](https://github.com/facebookresearch/nwm/tree/3f6cd8e70d6f2d1e2b9684acff510710135f0f41)(2025-08-13, CC BY-NC 4.0)
- 参照した一次資料: 上記コード全体(`models.py`, `train.py`, `datasets.py`, `misc.py`, `diffusion/`,
  `isolated_nwm_infer.py`, `isolated_nwm_eval.py`, `planning_eval.py`, `interactive_model.ipynb`, `config/`, `README.md`)/
  論文 [arXiv:2412.03572v2](https://arxiv.org/abs/2412.03572)(2025-04-11)本文および Supplementary Material
- 設計方針(作成時の選択): 言語=日本語 / 深さ=コード+数式対応 / 図=CSS 自動ループ+ステップ操作
  / スタイルは BAGEL トピックの `assets/style.css` をそのまま複製(トピックを自己完結させる方針)

## 論文とコードで食い違う点(両論併記した箇所)

本シリーズの方針どおり、どちらかに寄せず両方をページ上に明記した。

| 項目 | 論文 | 公開コード | 記載場所 |
|---|---|---|---|
| 拡散の予測対象 | L_simple は **clean な状態 s(τ+1)** との二乗誤差として記述(§3.2) | `create_diffusion` の既定は `predict_xstart=False` → **ε 予測** | `training.html#loss` |
| CEM のハイパーパラメータ | N=120 サンプル、1 反復、M 回平均(Appendix 7) | `argparse` 既定は `num_samples=10, opt_steps=15, num_repeat_eval=1`。**README のコマンドは論文と一致**(120 / 1 / 3) | `inference.html#cem` |
| CEM 初期分布 | 「分散 Σ = diag(σ²)」、RECON なら σ²\_Δx = 0.02、σ²\_Δy = σ²\_φ = 0.1(§8.2) | `data_hyperparams_plan.yaml` の同じ数値を **標準偏差として**使用(`randn(...) * sigma + mu`) | `inference.html#cem` |
| ラベルなしデータ(Ego4D) | 「ξ の計算時に行動項を省く」(§3.2)/ Table 4–5 に実験結果 | `c = t + time_emb + y` に**分岐なし**。`data_config.yaml` にも Ego4D のエントリなし | `architecture.html#conditioning` / `training.html#data` |
| 外部方策のランキング(NWM + NoMaD) | Table 2 / Table 7 に結果 | `planning_eval.py` に **NoMaD を呼ぶ経路が無い**(痕跡はヘルプ文字列とコメントのみ) | `inference.html#ranking` |

## このトピック固有の読みどころ(調査で見つかった実装の癖)

1. **AdaLN が 11 分割** — DiT の 6 分割に対し、cross-attention の query 側 3 本と
   **key/value 側 2 本**が追加されている。文脈側も条件で変調されるので、
   「同じ文脈でも行動によって読み方が変わる」構造になっている。
   なお **1 ブロックのパラメータの 41%(XL で 14.6M)が AdaLN**。
2. **CDiT-L/2 ≈ 687M ≈ DiT-XL/2 の 675M** — 論文の
   「同じパラメータ数で 4 倍速い(CDiT-L vs DiT-XL)」という主張は、
   `models.py` の定義から数え上げると実際にほぼ同規模であることを確認できる。
   CDiT-XL/2 は約 1.01B で、論文の "1 billion parameters" と一致。
3. **`get_2d_sincos_pos_embed` は定義されているが使われていない** —
   `pos_embed` は `(context_size+1, num_patches, hidden)` の**学習可能パラメータ**。
   (時刻, 空間位置) の組ごとに独立したベクトルを持つため、**文脈長は学習後に変更できない**。
4. **マジックナンバー 128 が 2 か所にハードコード** —
   `datasets.py` の `rel_time = goal_offset/128.`(`# TODO: refactor` 付き)と
   `isolated_nwm_infer.py` の `rel_t = (1./128.) * num_timesteps`。
   ±64 フレーム(4 FPS で ±16 秒)を [-0.5, 0.5] に写すための定数。
5. **`model_forward_wrapper` の `num_timesteps` は拡散ステップ数ではなくフレーム数** — 引数名が紛らわしい。
6. **CFG のための条件ドロップアウトは実装されていない** —
   `train.py` L258 のコメント "This enables embedding dropout for classifier-free guidance" は
   DiT からの残骸で、`CDiT` にラベルドロップ機構は無い。
7. **最終ロールアウトでヨーに π が二重に掛かる** —
   CEM ループ内では `deltas[:, -1, -1] += sample[:, -1] * np.pi` でラジアン化しているが、
   `mu` はその**ラジアン値**の平均で更新され、最終ロールアウト(L283)で再度 `* np.pi` される。
   選抜時に評価した回頭角と、最後に実行される回頭角が π 倍ずれる。
   ATE / RPE は `actions_to_traj` が姿勢を単位クォータニオンで埋めるため**並進成分のみ**から計算され、
   Table 2 / Table 7 の数値には影響しない。影響するのは最終予測画像と `yaw_diff_norm`(Table 3 の δφ)。
8. **`opt.zero_grad()` のバグが 2025-07-29 に修正されたばかり** —
   コミット [`7782ade`](https://github.com/facebookresearch/nwm/commit/7782ade33153cebcde6c3cf3b870675190d66d94)
   以前は `if not bfloat_enable:` の内側にあり、**bfloat16 学習で勾配が累積し続ける**不具合があった。
9. **bfloat16 なのに `GradScaler` を使っている** — 害は無いが float16 前提の書き方の名残。
10. **推論はすべて EMA 重み** — `train.py` の検証、`isolated_nwm_infer.py`、`planning_eval.py`、
    デモノートブックのすべてが `ckp["ema"]` を読む。生の `model` 重みを使う経路は 1 つも無い。
11. **コード上のデータセット名 `sacson` = 論文の HuRoN**。
12. **`calculate_delta_yaw` は非正規化した Δx, Δy に `atan2` を適用する** —
    正規化空間は x と y でスケールが違う(min=[-2.5,-4], max=[5,4])ため、
    そのまま角度を取ると歪む。一方モデルに渡す並進量は正規化済みのまま。この使い分けは正しい。

## 図(アニメーション SVG)の一覧

| ページ | 図 | ステップ数 / 周期 |
|---|---|---|
| index | 行動の定義と時間シフトによる合成 | 4 / 12s |
| index | 全体データフロー(学習 → 生成 → 計画) | 7 / 14s |
| architecture | テンソル形状のパイプライン | 5 / 15s |
| architecture | CDiT ブロックの 3 段構成と AdaLN | 4 / 12s |
| architecture | DiT vs CDiT のアテンション行列サイズ | 2 / 10s |
| training | 行動の 4 段階正規化 | 4 / 16s |
| training | 1 サンプル → 4 ゴールペアの展開 | 4 / 12s |
| training | 拡散学習の 1 ステップ | 4 / 12s |
| inference | 2 つの生成モード | 2 / 10s |
| inference | 文脈のスライディングウィンドウ | 4 / 12s |
| inference | CEM の 5 段階 | 5 / 15s |
| evaluation | 3 つの評価プロトコル | 3 / 12s |
| evaluation | 評価セットの構築 | 3 / 12s |
| evaluation | FVD 比較(対 DIAMOND) | 4 / 16s |

すべて `figure.diagram[data-cycle-ms][data-step-count]` を持ち、
`assets/diagram-controls.js` が自動再生の ON/OFF と 1 ステップ送りのボタンを付与する。
`prefers-reduced-motion: reduce` の環境ではアニメーションを止め、全要素を可視状態にする。
