# CoDi-2 図解

Microsoft Research の [CoDi-2](https://github.com/microsoft/i-Code/tree/main/CoDi-2) を、コードへのリンクと SVG/CSS アニメーションで解説する日本語サイトです。

| ページ | 内容 |
|---|---|
| `index.html` | NExT-GPT との違いと全体フロー |
| `architecture.html` | interleaved sequence、入力・出力 projector |
| `training.html` | language loss と feature alignment loss |
| `inference.html` | text の自己回帰生成と non-text の diffusion 生成 |

図は自動再生・一時停止・ステップ送りに対応し、`prefers-reduced-motion` では静止表示になります。研究コードの参照先は公式リポジトリの `main/CoDi-2` です。
