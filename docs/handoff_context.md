# 引き継ぎコンテキスト（TDS / tool-manip-gym 周辺）

このファイルは、作業を別のエージェント（Codex等）に委託するための文脈共有用プロンプトである。
そのまま冒頭に貼り付けて使うことを想定している。

---

## 最新更新（2026-08-14、Claude向け）

**この節は、下に残してある旧handoffの記述より優先する。** 特にGit状態、テスト件数、
spray初期姿勢DBの品質、学習結果はこの節が最新である。作業開始時は、まずこの節と
`docs/spray_training_failure_report_20260814.md`を読むこと。

### 現在の役割分担

- 別Windows 11マシン: `mjcf-pose-editor`を使った基準姿勢XMLのGUI編集だけを行う。
- この学習マシン: 編集済みXMLの取り込み、`tool-pose-pipeline`による摂動姿勢DB生成、
  `improved_tds`による学習・評価を行う。
- 姿勢DBや学習結果はGUIマシンへ置かず、この学習マシンで管理する。

### リポジトリと作業ツリーの最新状態

| project | HEAD / remote | 状態 |
| --- | --- | --- |
| `improved_tds` | `e5b3f78`, `main...origin/main` | **多数の未コミット変更・未追跡ファイルあり**。本handoff、pose pool wrapper、学習スクリプト、各レポートを含む |
| `tool-manip-gym` | `6f35f75`, `main...origin/main` | spray Lite pose XMLまではpush済み。`gym-spray/gym_spray/__init__.py`と`gym-spray/tests/test_spray_hands.py`の2ファイルが未コミット |
| `tool-pose-pipeline` | Git未初期化 | generator、QAコード、`grasps/*.npz`はローカル資産。消さないこと |
| `mjcf-pose-editor` | `f1b7d5e`, `main...origin/main` | clean。GitHubへpush済み |

remote:

- `improved_tds`: `git@github.com:KazukiHigashi/improved_tds.git`
- `tool-manip-gym`: `git@github.com:HigashimoriLab/tool_manip_gym.git`
- `mjcf-pose-editor`: `ssh://git@ssh.github.com:443/KazukiHigashi/mjcf_pose_editor.git`

既存の未コミット変更はすべてユーザ資産である。勝手に破棄、reset、checkout、commit、push
しないこと。まず各repoで`git status --short`と`git diff`を確認する。

### 前回handoff後に完了した主な変更

1. `mjcf-pose-editor`をGitHub公開し、別Windows 11マシン用の
   `setup_windows.cmd`、`launch_editor.cmd`、`WINDOWS_QUICKSTART_JA.md`を追加した。
   GUIマシンではリポジトリをcloneし、`setup_windows.cmd`、`launch_editor.cmd`の順に
   実行すればよい。編集対象のMJCFだけでなく、参照されるasset tree全体をコピーする。
2. `tool-manip-gym`のpushタスク・Lite hand対応を`efc9469`でpushし、spray1〜5の
   `shadow_hand_spray*_lite_pose.xml`を`6f35f75`でpushした。
3. spray Liteの authored XML確認用ID
   `ShadowHandLiteSpray{1-5}Pose-v0`を追加した。ただしこの登録とテストは現在
   `tool-manip-gym`の未コミット差分である。学習にはこのPose IDを使わず、canonicalな
   `ShadowHandLiteSpray{1-5}-v0`へpose poolを注入する。Pose IDで学習するとhand mount
   offsetを二重適用する危険がある。
4. `improved_tds`へ`InitialPosePoolWrapper`、`SprayGoalWrapper`、`PushGoalWrapper`を追加した。
   各episode reset時に対応する32姿勢DBから1姿勢をランダム選択し、その姿勢から開始する。
5. 学習CLIをSAC+HER既定にし、成功軌跡の保存、10万stepごとのcheckpoint、action飽和率、
   reward設定、pose pool pathをsummaryへ保存するようにした。
6. **学習によってTDS制御用の初期姿勢を獲得・最適化する経路は使用しない。** 現在の
   `train_sb3`は対応pose poolを必須とし、poolなしfallbackを持たない。学習対象は初期姿勢
   そのものではなく、DBから選ばれた姿勢を起点とするハンド制御方策である。
7. pushの旧Sparse Reward失敗を分析し、HER互換Dense Rewardへ変更した。落下・転倒時は
   他の正報酬をすべて上書きして`-250`とし、早期終了が長時間の未成功維持より有利に
   ならないようにした。

### 初期姿勢DB

全poolはLiteの13 action、32姿勢である。

```text
tool-pose-pipeline/grasps/push_lite_1.npz ... push_lite_5.npz
tool-pose-pipeline/grasps/spray_lite_spray1.npz ... spray_lite_spray5.npz
```

spray poolのpackaged学習環境上での50step人差し指駆動QA結果:

| variant | drive合格 | travel中央値 | 判定 |
| --- | ---: | ---: | --- |
| spray1 | 32/32 | 0.0358 rad | 最低基準は通るが余裕不足 |
| spray2 | 32/32 | 0.2700 rad | 良好 |
| spray3 | 0/32 | 0.00316 rad | 使用不可 |
| spray4 | 0/32 | 0.00142 rad | 使用不可 |
| spray5 | 0/32 | 0.00187 rad | 使用不可 |

原因は`tool-pose-pipeline/toolpose/generate_spray_lite.py`が
`RefineConfig(require_drive=False)`をhard-codeしていること。DB metadataでもspray2だけが
`require_drive=true`で、spray1・3・4・5は`false`だった。現DBのままspray3〜5を追加学習
してはいけない。`docs/spray_lite_training_setup.md`末尾の「spray2〜5は駆動しないが学習で
獲得する」という記載は調査前の古い判断であり、最新レポートで否定されている。

### push1〜5学習

旧run `runs/push_lite_sac_her_1m_20260810`ではSparse Rewardにより、push1が約7stepで
転倒させて負報酬の累積を避ける方策へ崩壊した。詳細は
`docs/push1_training_failure_report_20260811.md`を参照する。旧push2は264,727stepで
MuJoCo QACC warningがstderrへ出た際、PowerShellがNativeCommandErrorとして扱い中断した。

Dense Rewardへ変更後のrunは`runs/push_lite_dense_reward_1m_20260811`。全variantが
1,000,000 timestepを完了し、最終モデルと各10 checkpointsがある。

| variant | 最終rolling成功率 | 保存成功軌跡 | 学習時action飽和率 |
| --- | ---: | ---: | ---: |
| push1 | 100% | 3,637 | 26.5% |
| push2 | 94% | 3,899 | 23.3% |
| push3 | 99% | 3,469 | 26.3% |
| push4 | 91% | 3,618 | 26.9% |
| push5 | 99% | 3,094 | 29.6% |

これは学習ログのrolling値であり、固定目標×全姿勢の共通評価値ではない。今後モデル間を
比較するときは、spray調査と同じ固定目標グリッド評価を別途行うこと。

### spray1〜5学習と原因調査

runは`runs/spray_lite_1m_reward_v2_20260814`。全variantが1,000,000 timestepを完了し、
最終モデル、summary、training candidates、各10 checkpointsがある。全ログでNaN、Infinity、
Python例外、MuJoCo warningは検出されなかった。

| variant | 学習時rolling成功率 | 固定条件96episode安定成功 | 確率的50episode安定成功 |
| --- | ---: | ---: | ---: |
| spray1 | 23% | 33.3% | 24% |
| spray2 | 93% | 93.8% | 98% |
| spray3 | 0% | 0% | 0% |
| spray4 | 0% | 0% | 0% |
| spray5 | 0% | 0% | 0% |

固定条件評価は全32姿勢×目標0.04/0.12/0.20 radの96episode。spray1は0.04 radだけ
32/32成功し、0.12/0.20 radは0/32だった。spray1成功軌跡は目標0.020〜0.072 radに
限定され、0.08 rad以上は0件である。無操作でも約0.0404 radの受動ドリフトがあり、
低角目標とドリフトを利用した見かけ上の成功が多い。spray2だけが全目標域を操作できた。

spray3〜5の主因は上記DBの非駆動性であり、SAC/HER単独の問題ではない。spray4・5は
落下・転倒を避けてほぼ無操作を維持し、tracking基礎報酬を得る安全な局所解へ収束した。
現在のtrackingは`2 * (1 - error / 0.4)`なので、進捗0でも目標0.04/0.12/0.20 radに
対して1.8/1.4/1.0 reward/stepを得る。これは偽successではないが、失敗状態を過大評価する
reward shortcutである。

spray2ではtrigger qposが関節範囲`[0, 0.4]`外となるstepが1.77%あったが、範囲外stepの
平均報酬は範囲内より低く、グリッチによる報酬獲得は確認されなかった。詳細な数値、把持、
drop/tip、action飽和率、報酬内訳は次を参照する。

```text
improved_tds/docs/spray_training_failure_report_20260814.md
```

### 次に行う作業（優先順）

1. **P0: spray DB生成を修正する。** `generate_spray_lite.py`の
   `require_drive=False` hard-codeを廃止し、全accepted poseでpackaged runtime driveを
   必須化する。合格条件は最低でもtravel >= 0.02 rad、drive fraction >= 5%、
   index-to-trigger <= 50 mm、2指以上の把持可能性、非落下・非転倒。
2. **基準姿勢を再編集する。** spray2は現状を基準にできる。spray1はより大きな能動travelが
   必要。spray3〜5はGUI側で姿勢、hand mount、trigger geomとの位置関係を作り直す。
   gateを緩めて32件を埋めない。
3. **32姿勢poolを再生成・全数QAする。** zero-action保持とdrive試験をcanonical envへ注入して
   実測し、`require_drive=true`をNPZ artifactとテストで検証する。
4. **P1: spray報酬を修正する。** trackingを例えば
   `exp(-0.5 * (error / tolerance) ** 2)`の局所報酬へ変更し、tracking/progressを
   `0.25 + 0.75 * grasp_score`で部分ゲートする。2指接触継続とupright維持には小さな
   非負報酬を与え、drop/tipは引き続き`-250`で全項を上書きする。HER再計算可能性を保つ。
5. **短時間smoke runを先に行う。** 固定目標0.04/0.12/0.20 rad×全32姿勢で、角度到達、
   把持維持、drop/tip、pose別成功率、action飽和、報酬内訳を確認してから1Mへ進む。
6. trigger qpos範囲外率をtelemetryへ追加し、必要ならXML joint limit、solver、dampingを
   強化する。物理警告は1回出ただけで機械的に停止せず、非有限状態の伝播、連続発生、
   reset後の再発、学習成果物の破損を停止条件にする。

### 2026-08-14の回帰確認

共通venvで次の関連テストを実行し、合計378件が合格した。

- `improved_tds`: 65 passed（Gymnasiumの無限Box境界warning 14件）
- `tool-pose-pipeline`: 44 passed（テスト終了時に局所的なMuJoCo QACC warning 2件）
- `tool-manip-gym`: 157 passed
- `mjcf-pose-editor`: 112 passed

QACC warningはこのテストrunを失敗させず、NaNを成果物へ残していない。ただし、再現頻度を
上げる場合は物理QA項目として扱う。旧handoffの「345件合格」は過去時点の値である。

### Claudeが最初に確認するファイル

1. `improved_tds/docs/spray_training_failure_report_20260814.md`
2. `improved_tds/src/improved_tds/environments/pose_pool.py`
3. `tool-pose-pipeline/toolpose/generate_spray_lite.py`
4. `improved_tds/scripts/train_spray_lite_tasks.ps1`
5. `improved_tds/docs/spray_lite_training_setup.md`（末尾QAは最新レポートで読み替える）
6. `tool-manip-gym/gym-spray/gym_spray/__init__.py`の未コミット差分

このhandoff作成時点では、上記P0/P1改善は**まだ実装しておらず**、spray再学習も開始して
いない。原因調査済みの現DBで学習を延長しないこと。

## 0. あなたへの依頼の前提

あなたは、大阪大学の研究プロジェクト「TDS (Tool DoF Synergy)」のコードベースを引き継いで作業する。
以下はこれまでの経緯と現在の状態である。**推測で埋めず、不明点は必ずコードを読んで確認すること。**

重要な原則（これまでの作業で守ってきたもの）：

- **数値で検証する。** 「直したはず」で終わらせない。姿勢を変えたら零入力での保持量とツール自由度の駆動量を実測し、数値を示す。
- **周辺のコードのスタイルに合わせる。** コメントは「なぜ」を書き、「何を」は書かない。既存の密度・命名・イディオムに揃える。
- **失敗を隠さない。** テストが落ちたら出力とともに報告する。範囲外で壊れているものを見つけたら、直さずに報告して判断を仰ぐ。
- **研究資産に勝手に触らない。** 姿勢DB（`.npy`）、過去の実行記録（`outputs/`）、解析次元に関わる設定は、指示なく変更しない。

---

## 1. 環境

| 項目 | 値 |
| --- | --- |
| OS | Windows 11 / PowerShell（Bashツールも利用可） |
| ルート | `c:\Users\smasa\Documents\mujoco\projects` |
| Python | 3.12.10 |
| 主要パッケージ | mujoco 3.9.0 / gymnasium 1.2.3 |
| 仮想環境 | `fdms_reconfiguration/.venv`（**全プロジェクト共通でこれを使う**） |

インタプリタのパス（そのまま使う）：

```
c:/Users/smasa/Documents/mujoco/projects/fdms_reconfiguration/.venv/Scripts/python.exe
```

全テストの実行：

```bash
cd c:/Users/smasa/Documents/mujoco/projects
"c:/Users/smasa/Documents/mujoco/projects/fdms_reconfiguration/.venv/Scripts/python.exe" -m pytest \
  tool-manip-gym tool-pose-pipeline mjcf-pose-editor fdms_reconfiguration/tests -q
```

**旧handoff作成時は345件合格。2026-08-14の関連4project再検証は、上の最新更新節にある通り合計378件合格。**

MuJoCo警告は局所的に出る場合がある。探索的確認では出力を絞ってよいが、最終QAでは警告を記録し、非有限状態の伝播・連続発生・reset後の再発を確認する。警告1件だけを理由に機械的停止はしない。

---

## 2. プロジェクト構成

`projects/` 直下は git 管理されていない。各プロジェクトが独立したリポジトリ（または非リポジトリ）である。

| ディレクトリ | 役割 | git |
| --- | --- | --- |
| `tool-manip-gym/` | Gymnasium環境本体。4タスク×5変種×2ハンド | あり（HEAD `6f35f75`、未コミット2ファイル） |
| `tool-pose-pipeline/` | 初期姿勢プール生成（Stage A/C/D） | **なし（未初期化）** |
| `mjcf-pose-editor/` | 初期姿勢を手動作成するGUI | あり（HEAD `f1b7d5e`、clean、GitHubへpush済み） |
| `improved_tds/` | TDS実装、SAC+HER学習、論文・ドキュメント | あり（HEAD `e5b3f78`、未コミット変更あり） |
| `fdms_reconfiguration/` | 下流のRL/シナジー最適化。上記envを利用 | あり（今回の最新監査対象外。既存変更を保持） |
| `.deprecated/` | 旧版アーカイブ。**触らない** | - |

`tool-manip-gym`の最新コミットは`6f35f75`。push Lite対応は`efc9469`、spray Lite pose XMLは`6f35f75`までpush済みで、spray Pose環境登録とテストだけが未コミットである。

### タスクと環境

4タスク、各5変種、shadow / lite の両ハンド。

| タスク | パッケージ | 環境ID | ツール自由度 |
| --- | --- | --- | --- |
| はさみ | `gym-scissors` | `ShadowHand[Lite]Scissors{1-5}-v0` | `scissors_hinge_2:joint` |
| スライダ | `gym-slider` | `ShadowHand[Lite]Slider{1-5}-v0` | `controller_slider` |
| スプレー（トリガ式） | `gym-spray` | `ShadowHand[Lite]Spray{1-5}-v0` | `spray_trigger` |
| プッシュボトル | `gym-push` | `ShadowHand[Lite]Push{1-5}-v0` | `push_plunger_joint` |

旧IDも後方互換で残してある（`ExpScissor{n}-v0`, `ScissorsVariant{n}-v0`, `SliderVariant{n}-v0`, `SprayVariant{n}-v0`, `PushVariant{n}-v0`, `ShadowHandSprayBottle-v0`, `ScissorsManipulate-v0` 等）。

`gym-spray` には旧来の `spray_type="push"`（アルコールディスペンサ式）も残っている。これは `gym-push` とは**別物**なので混同しないこと。

---

## 3. 研究の狙い（これが判断基準になる）

TDS は「**ハンドの姿勢に対してPCAをかけたとき、第1主成分がツールの自由度に対応する**」ことを示そうとしている。

そのため初期姿勢には次が要求される。ユーザから繰り返し指摘された最重要事項である。

> **手首のグローバルな運動ではなく、指を活用してタスクを実行できる初期姿勢であること。**

具体的な合格条件：

1. **係合** — 指がツールと噛み合っている（はさみは指と親指が持ち手の穴を貫通、sprayは指がグリップを包み人差し指がトリガー前面、pushは4指が胴を包み親指がプランジャ直上）。
2. **非貫通** — ハンドとツールが深く食い込んでいない。
3. **安定** — 零入力でツールが動かない・倒れない。
4. **駆動性** — 手首を固定したまま、操作指だけでツール自由度を動かせる。

姿勢を変更したら、必ず 3 と 4 を数値で測ること（後述の検証スクリプトの型を流用する）。

---

## 4. ハンドモデル（ここが最も間違えやすい）

### Shadow Dexterous Hand（標準）

24関節 / 20自由度。手首2 + 人差指・中指・薬指 各4関節3自由度 + 小指5関節4自由度 + 親指5関節5自由度。

### Shadow Dexterous Hand **Lite**

**16関節 / 13自由度。** 公式仕様（[技術仕様書](https://shadowrobot.com/wp-content/uploads/2025/09/shadow_dexterous_hand_lite_technical_specification-2025.pdf) §2.3 Kinematic structure）に準拠。

| 部位 | 関節 | 関節数 | 自由度 |
| --- | --- | --- | --- |
| 人差指・中指・薬指 | J4, J3, J2, J1（J1とJ2は結合） | 12 | 9 |
| 親指 | TH5, TH4, TH2, TH1 | 4 | 4 |
| 合計 | | **16** | **13** |

標準ハンドとの差は **小指・親指のTH3・手首(WR1/WR2)** の3点。標準の仕様表にはTH3もWR1/WR2も載っているが、Lite側の表には行そのものが無い。

以前このリポジトリの `right_hand_lite.xml` は「標準ハンドから小指を削っただけ」の19関節/16アクチュエータで**誤っていた**。修正済み。

> **実装上の落とし穴（再発しやすい）**: 手首関節を削除すると `rh_palm` が `rh_forearm` に溶接され、MuJoCoの親子接触フィルタは溶接単位で判定するため、palmと各指の付け根の間のフィルタが失われる。放置すると初期状態から1mm貫通した自己接触が15点発生し、ツールが弾け飛ぶ（slider4で52m飛んだ）。`right_hand_lite.xml` に `<exclude>` を明示追加して回避している。**関節を消すときは必ず溶接の影響を確認すること。**

### 指のJ1/J2結合

各指の中節(J2)と末節(J1)は結合しており、**J2のみが行動として露出し、J1にはその指令がコピーされる**。

- モデル: J2とJ1それぞれに位置アクチュエータを置く（`ctrlrange` は各 0〜1.5708）
- env: `COUPLED_DISTAL_ACTUATORS` と `_mirror_coupled_ctrl()` が、制御を書き込むたびにJ2の指令をJ1へ複製
- J1は `HAND_ACTUATOR_NAMES` に入らないので行動次元は増えない

> **数え方に注意。** モデルの `nu`（shadow 24 / lite 16）と、行動次元（shadow 20 / lite 13）は**別物**。自由度を数えるときは `nu` ではなく `env.actuator_names` を見ること。この取り違えでテストを何度も落とした。

> **親指の THJ2/THJ1 は名前が似ているが独立した関節で、結合対象ではない。** 命名パターンで自動検出すると誤判定するので、結合ペアは明示リストで持っている。

**以前はJ2+J1を長さとする固定テンドン1本を1アクチュエータで駆動していた。** 自由空間では両関節がほぼ等分されるが、J2がツールに当たって止まると指令の残りをJ1が吸収する劣駆動的な動きだった。現在はJ2が接触で止まってもJ1は自分の指令へ追従する。これに伴い：

- アクチュエータ名 `rh_A_*J0`（和、範囲0〜3.1415）→ `rh_A_*J2`（J2の角度、範囲0〜1.5708）
- 既存の初期姿勢値はすべて **1/2** に変換済み
- **この変更前に生成した姿勢プール(.npz)は読み込み時にアクチュエータ名不一致でエラーになる。** 作り直しが必要

---

## 5. 初期姿勢の作成フロー

```
mjcf-pose-editor（人手で基準姿勢を作る）
        ↓ 編集済みMJCF
tool-pose-pipeline（摂動を加えて検証済み姿勢プールを作る）
        ↓ .npz
tool-manip-gym の env に PosePoolWrapper で注入して学習
```

最適化ベースの自動生成（Stage B / MAP-Elites / CMA-ES）も一度実装したが、**ユーザの判断で「基準姿勢は人手で作る」方針に変更された**。自動生成は使わない。

### mjcf-pose-editor

- `Scene(path, hand="shadow"|"lite")` — シーンの `<include file="right_hand.xml"/>` の1行だけを差し替えて、同じディレクトリの一時ファイル経由でコンパイルする。シーンファイルは両ハンド共通のまま保てる
- 差し替えXMLを一時ファイルにする理由は、MuJoCoが `<include>` と `meshdir` を読み込み元ファイルの位置基準で解決するため。`from_xml_string` にアセット辞書を渡す方式は、sprayのメッシュが `assets/spray1/` のようなサブディレクトリにあるため失敗する
- **保存されるMJCFは必ずハンド固有になる。** MjSpecが `<include>` を展開し、keyframeの `qpos` 幅も固定されるため。1ファイルで両ハンドを兼ねることは原理的に不可能
- 保存内容: ルートbodyの `pos`/`quat`、各関節の `springref`、`<keyframe name="initial_pose">`（`qpos` と `ctrl`）
- **`mj_resetData` では関節角が戻らない。** MuJoCoに関節の既定角を表す属性は `ref` しかなく、`ref` は「その角度を新しい零点にする」意味なので姿勢を動かさず原点をずらす。したがって関節角はキーフレームに入れてあり、読み込み側は `mj_resetDataKeyframe(model, data, 0)` を使う
- **結合ペアは1本のスライダーにまとめて表示される**（ラベル `FFJ2+J1`）。表示制御数は行動次元と一致（shadow 20 / lite 13）

### tool-pose-pipeline

- Stage A: 係合スペック（`toolpose/specs.py` の `TASK_SPECS`）
- Stage C: 物理検証（係合・非貫通・安定）
- Stage D: 摂動による姿勢プール生成

```bash
cd tool-pose-pipeline
python -m toolpose.generate --out artifacts --size 12          # 全ケース
python -m toolpose.generate --out artifacts --hand lite        # liteのみ
python -m toolpose.generate --out artifacts --task push --hand lite --variant 1 \
    --model ../tool-manip-gym/gym-push/gym_push/assets/shadow_hand/shadow_hand_push1_pose.xml
```

`--model` でエディタが出力したMJCFを渡せる。

タスクごとの評価指標：

| タスク | 係合条件 | 把持品質 | orbit記述子 | 貫通許容 |
| --- | --- | --- | --- | --- |
| scissors | 指と親指が持ち手の穴を貫通 | force closure | なし | 4mm |
| slider | 親指がノブ近傍 | **caging** | なし | 4mm |
| spray | 人差し指がトリガ近傍＋2指以上が胴周囲 | **caging** | あり(push型のみ) | 4mm |
| push | 親指がプランジャ近傍＋2指以上が胴周囲 | **caging** | あり | **10mm** |

sliderとsprayがcagingなのは、ユーザの明確な指示による（「スライダを手のひらに乗せて操作するので、厳密にフォースクロージャを満たしている必要はありません」）。
pushの貫通許容が10mmなのは、包み込む把持では指腹が胴に沈む量が指先接触より大きいため。既定4mmだと10ケース中6件が不合格になるが、係合も安定も正常だった。

---

## 6. lite専用の初期姿勢

shadow用の姿勢から不足関節を落としただけではliteで成立しないケースがある。該当したものは lite 専用姿勢を用意してある（`hand="lite"` のときだけ選択される）。

| 対象 | 症状 | 定義場所 |
| --- | --- | --- |
| spray（トリガ式1-5） | 小指が塞いでいた握りの下側が開き、spray4/5でボトルが抜けた | `TRIGGER_READY_POSE_LITE` |
| spray（旧push型） | 親指が1関節短くポンプを押し切れない | `SPRAY_TYPES["push"]["initial_joint_qpos_lite"]` |
| push | 親指がプランジャから約1cm遠く、push1/3が係合判定を外れた | `INITIAL_JOINT_QPOS_LITE` |

scissorsとsliderはshadowと同じ姿勢定義で問題ない（該当しない関節は自動的に読み飛ばされる）。

> **姿勢調整のコツ（実測で得たもの）**: 安定性だけを目的に最適化すると、握りが閉じてツール自由度が動かなくなる。**安定性と機能（トリガ駆動量・親指の到達距離・ポンプ押込量）を必ず同時に目的関数へ入れること。** 貫通も制約に入れないと指がツールに沈む。

---

## 7. 下流 `fdms_reconfiguration` との関係

同プロジェクトは `gym_spray` / `gym_scissors` を**直接import**している（`import gym_spray` すると `tool-manip-gym` 側が読まれる）。したがって env を変更すると壊れる。

`mapper: range_ratio` は「関節可動域比で連動比を固定する」方式で、姿勢DBの**値を使わない**（`fixed_joint_reference: neutral`）。姿勢DBファイルが存在しない場合は `zeros((1, num_joints))` で代替する分岐が `rl_train.py` にある。

本作業では以下を更新済み：

- `configs/scissor.yaml` / `scissor_4.yaml` / `scissor_5.yaml`: `num_joints` 14→13、`joint_names_path` → 新規 `scissor_action_names_lite13.json`、`posture_db_path` → 実在しない lite13 用パス（上記フォールバックを使う）
- `configs/spray_gym.yaml`: `joint_names_path` → 新規 `spray_action_names_coupled.json`、`initial_joint_qpos` の `rh_*J0: 1.57075` → `rh_*J2: 0.785375`
- 対応するテスト2ファイル

**触っていないもの（ユーザ指示による）**: `assets/posture_db/*.npy`、元の `*_action_names.json`、`outputs/` 配下の過去実行記録。

---

## 8. 既知の未解決問題

作業中に見つけたが、範囲外として手を付けていないもの。**勝手に直さず、必要なら先に確認すること。**

1. **spray3 の初期状態が shadow / lite の両方で不安定。** 最新のLite pool監査でも20stepボトル変位中央値14.53mm、drive合格0/32だった。モデル形状、hand mount、基準姿勢を含めて再検討が必要であり、現poolを学習へ再利用しない。
2. **push の shadow 姿勢がプランジャをほとんど押せていない。** 駆動量 0.0001〜0.001 に対し lite は 0.006〜0.015。shadow側の姿勢も見直す価値がある。
3. **scissors の零入力変位が全変種で大きい**（13〜44mm、shadow/lite同程度）。テーブル上で落ち着く分とみられるが未検証。
4. `tool-pose-pipeline`は**git未初期化**。`mjcf-pose-editor`はGitHubへpush済みでclean。
5. `.venv/mujoco_plugins_blocked_by_app_control/` に `elasticity.dll` / `sensor.dll` を退避してある（Windowsアプリ制御対策）。現在はスマートアプリコントロール設定済みなので、戻せる可能性がある。

---

## 9. 検証スクリプトの型

姿勢を変えたら必ずこれを回す。全20組合せでshadowとliteを比較する。

```python
import warnings; warnings.filterwarnings('ignore')
import numpy as np, mujoco
from gym_scissors.envs import ScissorsManipulateEnv
from gym_slider.envs import SliderManipulateEnv
from gym_spray.envs import ShadowHandSprayBottleEnv
from gym_push.envs import PushBottleEnv

DOF  = {'scissors':'scissors_hinge_2:joint','slider':'controller_slider',
        'spray':'spray_trigger','push':'push_plunger_joint'}
FREE = {'scissors':'scissors:joint','slider':'controller_free',
        'spray':'spraybottle_free','push':'push_object_free'}
DRV  = {'scissors':('rh_A_FFJ','rh_A_THJ'),'slider':('rh_A_THJ',),
        'spray':('rh_A_FFJ',),'push':('rh_A_THJ',)}
mk = {'scissors':lambda h,n: ScissorsManipulateEnv(hand=h,scissors_variant=n),
      'slider'  :lambda h,n: SliderManipulateEnv(hand=h,slider_variant=n),
      'spray'   :lambda h,n: ShadowHandSprayBottleEnv(hand=h,spray_type=f'spray{n}'),
      'push'    :lambda h,n: PushBottleEnv(hand=h,push_variant=n)}

def qa(m, nm):
    return m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm)]

def an(e):                       # env ごとに公開名が違う
    for a in ('actuator_names', '_actuator_names'):
        if hasattr(e, a):
            return list(getattr(e, a))

for task in mk:
    for n in range(1, 6):
        r = {}
        for h in ('shadow', 'lite'):
            e = mk[task](h, n); e.reset(seed=0); m, d = e.model, e.data
            fa, da = qa(m, FREE[task]), qa(m, DOF[task])
            p0 = d.qpos[fa:fa+3].copy()
            for _ in range(100):                       # 零入力での保持
                e.step(np.zeros(e.action_space.shape))
            hold = float(np.linalg.norm(d.qpos[fa:fa+3] - p0)) * 1000
            e.reset(seed=0)                            # 操作指だけで駆動
            a = np.zeros(e.action_space.shape[0])
            for k, nm in enumerate(an(e)):
                if any(nm.startswith(p) for p in DRV[task]):
                    a[k] = 1.0
            v = []
            for _ in range(200):
                e.step(a)
                v.append(float(d.qpos[da]))
            r[h] = (hold, max(v) - min(v)); e.close()
        print(f'{task}{n}: hold shadow {r["shadow"][0]:7.2f} lite {r["lite"][0]:7.2f} mm | '
              f'travel shadow {r["shadow"][1]:.4f} lite {r["lite"][1]:.4f}')
```

合格の目安: `hold` が shadow と同程度（liteが極端に大きくないこと）、`travel` が有意に非ゼロ。

---

## 10. 参照すべきドキュメント

- `improved_tds/docs/initial_pose_authoring.md` — 初期姿勢作成とプール構築の手順書（日本語）。**本件の最も詳しい説明。まず読むこと。**
- `improved_tds/` 配下の他ドキュメント — 論文ドラフトと関連資料

---

## 11. 依頼内容

具体的な次作業は、このファイル冒頭の「次に行う作業（優先順）」に従う。現時点の最優先はspray初期姿勢DBのruntime drive必須化と基準姿勢の再編集であり、現DBのまま追加学習しない。
