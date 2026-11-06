# Weight Space Learning 図解 — トピック概要と作成プロセスの記録

## このトピックについて

*A Survey of Weight Space Learning: Understanding, Representation, and Generation*
(Han, Wang, Zhao, Zhang, Li, Borth, Yu, Maron, Ye, Yin, Neri, 2026 / [arXiv:2603.10090](https://arxiv.org/abs/2603.10090))は、
学習済みニューラルネットの**重みそのものを学習の対象(データ)として扱う**研究領域 Weight Space Learning (WSL) を、
初めて統一的な分類体系で整理したサーベイ。本シリーズで初めて扱う**サーベイ論文**である。
本解説は、サーベイの 3 次元(理解 WSU・表現 WSR・生成 WSG)と、評価・応用・課題を、
サーベイの数式・本文の引用とアニメーション図解で読み解く 6 ページ構成の日本語解説である。

| ページ | 内容 |
|---|---|
| `index.html` | 研究背景(重み = 新しいデータモダリティ)/ 関連研究(視点の転換 4 段階、既存サーベイ、隣接分野)/ 問題設定(3 つの写像)/ 3 次元の全体像アニメ図 / 分類体系(Figure 3)の要約 / 章立て |
| `understanding.html` | 式 (1)(2) の不変性・同変性、商空間、置換対称性のアニメ図と**触れるデモ**、Table 1、線形モード連結性と Git Re-Basin、圧縮・最適化・データ拡張、§2.3 の議論 |
| `representation.html` | 式 (3)〜(12)、メタモデルの学習データ・構造、4 系統(対称性非考慮 / 考慮 / グラフ型 / プローブ型)、学習信号(教師あり・自己教師あり)、擬似コード、性能予測・検索・編集、§3.3 の議論 |
| `generation.html` | 式 (13)〜(22)、Table 2、ハイパーネットの二段階最適化と**パラメータ数計算機**、設計 4 軸、生成モデル 4 系統(AE / GAN / 自己回帰 / 拡散)、潜在拡散の流れと擬似コード、6 つのユースケース、§4.3 の議論 |
| `benchmarks.html` | model zoo の作り方、Table 4 の転記、MLP → CNN → RNN → Transformer、各タスクの評価方法、サーベイによる評価の批判 |
| `outlook.html` | 6 つの応用分野(Table 3)、近年の 5 つのトレンド(解説者の整理)、§7 の 3 つの未解決問題、考察と全体まとめ |

- 参照した一次資料: 提供された PDF(本文 39 ページ + 参考文献、全 51 ページ)を通読。
  付随リポジトリ [Zehong-Wang/Awesome-Weight-Space-Learning](https://github.com/Zehong-Wang/Awesome-Weight-Space-Learning) は論文リスト
- 設計方針: 言語=日本語 / 深さ=数式・図表対応 / 図=CSS 自動ループ+ステップ操作
  / スタイルは BeyondLM トピックの `assets/style.css` を複製し、末尾にインタラクティブ部品(`.widget`)を追加

## 「コードなし」トピックとしての方針

サーベイなので実装コードの行番号付き引用はできない。BeyondLM 回と同様に、次の 3 つを使う。

1. **サーベイの数式をそのまま**(式 1〜22)
2. **サーベイの表・本文の引用**(Table 1〜4、各節の Takeaways と本文)
3. **擬似コード**(メタモデル学習・ハイパーネット学習・潜在拡散)。必ず `⚠️ 擬似コード` と明記

サーベイ本文にない補足(自己教師あり hyper-representation の損失、性能予測の評価指標など)は
**「補足(サーベイ外)」**、解説者の解釈(トレンドの整理、未解決問題のつながり)は**「解説者の整理 / 考察」**と明記して区別した。

## 表記上の注意

- 式 (21) はサーベイ本文では \(\sqrt{\alpha_t}\) 表記だが、解説では DDPM の標準記法 \(\bar\alpha_t\) で書き、その旨をページに注記した
- 線形モード連結性の図(損失地形・補間曲線)は模式図であり、特定の実験結果のプロットではない(ページに注記)

## インタラクティブ部品

| ページ | 部品 | 内容 |
|---|---|---|
| understanding | 置換・スケーリングのデモ | 2-3-1 ReLU MLP の隠れニューロンを置換 / 正スケーリング / ノイズ付加し、重みベクトルの距離・出力・素朴なエンコーダ・DeepSets 型集約の変化を比較 |
| generation | パラメータ数計算機 | §4.1.1 の式 \(L N_{in} N_{out} f^2\) と \(Ld + hdN_{in} + hN_{out}f^2\) をスライダーで評価 |

## 図(アニメーション SVG)の一覧

| ページ | 図 | ステップ数 / 周期 |
|---|---|---|
| index | 重みを「産物」から「データ」へ | 3 / 12s |
| index | 視点の転換 4 段階 | 4 / 14s |
| index | WSU / WSR / WSG と応用 | 4 / 14s |
| understanding | 置換対称性 | 4 / 14s |
| understanding | 不変性と同変性 | 2 / 8s |
| understanding | 線形モード連結性と Git Re-Basin | 3 / 12s |
| understanding | WSU のユースケース | 3 / 12s |
| representation | メタモデルの流れ | 3 / 12s |
| representation | 表現手法の 4 系統 | 4 / 14s |
| representation | グラフ型表現とメッセージパッシング | 3 / 12s |
| representation | 埋め込み空間での予測・検索・編集 | 3 / 12s |
| generation | ハイパーネットの学習と生成 | 4 / 14s |
| generation | 生成モデル 4 系統 | 4 / 14s |
| generation | 潜在空間での重み拡散 | 4 / 14s |
| benchmarks | model zoo の作り方 | 4 / 14s |
| benchmarks | アーキテクチャ族の階段 | 4 / 14s |
| outlook | 6 つの応用分野 | 6 / 15s |
| outlook | 5 つのトレンド | 5 / 15s |
| outlook | 3 つの未解決問題 | 3 / 12s |

すべて `figure.diagram[data-cycle-ms][data-step-count]` を持ち、
`assets/diagram-controls.js` が自動再生の ON/OFF と 1 ステップ送りのボタンを付与する。
`prefers-reduced-motion: reduce` の環境ではアニメーションを止め、全要素を可視状態にする。
