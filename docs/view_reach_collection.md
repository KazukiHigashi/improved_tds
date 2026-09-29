# Scissors 成功姿勢DBの再収集をMuJoCo Viewerで確認する

`scripts/view_reach_collection.py` は、2026-09-23のScissors 3–5学習結果にある
`best_success_policy.zip` と32姿勢のプールを読み、
`collect_reach_successes.py` の成功姿勢収集ループをViewerで再生する。
プレビューはDBを書き換えず、結果をメモリ内だけで管理する。

プロジェクト `improved_tds` を作業ディレクトリとしてPowerShellで実行:

```powershell
$env:PYTHONPATH=(Resolve-Path src).Path
..\fdms_reconfiguration\.venv\Scripts\python.exe scripts\view_reach_collection.py --variant 4 --episodes 30 --per-bin 1 --speed 1 --pause-on-success
```

Viewerは環境の既存 `render_mode="human"` を利用する。
`--speed 1` はシミュレーションの0.04秒/制御Stepに合わせた再生。
採用姿勢が出た時に `--pause-on-success` で一時停止し、端末でEnterを押すと次へ進む。
Ctrl+Cで終了。GUIは人が操作する必要があるためバックグラウンド起動しない。

全角度帯を少数ずつ見る例は `--per-bin 1`。
本収集と同じ各帯10件を試す場合は `--per-bin 10 --episodes 1000` などとする。
Scissors 5の端点付近は非常に成功しにくいため、十分な試行予算を指定する。
`--bin 9`で0.9–1.0 rad帯を重点表示し、`--pose 5`で初期姿勢を固定できる。
これらの指定時は収集器の標準の角度帯・初期姿勢選択を診断用に上書きする。

端末には試行番号、初期姿勢、目標角度、決定論的/確率的操作、各Stepの実角度、
角度誤差、現行把持判定の指数、5Step成功カウンタ、失敗状態を表示する。
`blade/handle/mirror`は刃／動くハンドル／反対側のハンドルに触れる指の数。
同じ指が複数部位に接触し得るため、この3値の和は把持指の数ではない。
現在の成功判定が刃接触を数える場合もあるので、両方を並べて観察する。
成功後にはDB採用または棄却理由、実関節姿勢と実指令のRMS差を表示する。

Viewerを開かずに経路だけ確認する場合は `--no-render --episodes 3`。
再学習や既存DBの変更は行わない。
