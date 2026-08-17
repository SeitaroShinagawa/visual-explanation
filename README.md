# visual-explanation

アニメーション図解で読み解く GitHub Pages シリーズ。

## 収録コンテンツ

| トピック | パス | 内容 |
|---|---|---|
| [NExT-GPT](NExT-GPT/) | `NExT-GPT/` | Any-to-Any マルチモーダル LLM(ICML 2024)の学習・生成プロセスをコード引用+アニメーション SVG で解説(概要 / アーキテクチャ / 学習 / 生成の 4 ページ構成) |
| [CoDi-2](CoDi-2/) | `CoDi-2/` | In-Context・Interleaved・Interactive な any-to-any 生成を行う MLLM(CVPR 2024)。テキストは離散トークン、それ以外は連続ベクトルとして生成する仕組みを、論文の損失式と実装コードを突き合わせて同じ 4 ページ構成で解説 |

## GitHub Pages の公開設定

リポジトリの **Settings → Pages → Build and deployment** で
**Deploy from a branch / `main` / `/ (root)`** を選択すると、
`https://<ユーザー名>.github.io/visual-explanation/` で公開されます。

- トップページ (`index.html`) から各解説ページへ遷移する入れ子構造
- シンタックスハイライト(Prism)と数式(MathJax)は jsDelivr CDN から読み込み
- 図は SVG + CSS keyframes による自動ループアニメーション(`prefers-reduced-motion` 対応)
