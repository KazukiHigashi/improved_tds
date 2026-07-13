# 閉ループTDSの実装計画とリポジトリ調査

## `tool-dof-synergy`の調査結果

参照したプロジェクトは`tool-dof-synergy`だけである。実行可能な現行移植版は外側の`tool_dof_synergy/`パッケージであり、内側の`tool_dof_synergy-main/`ツリーはTensorFlow/OpenAI Baselinesを使用した旧実装である。

- 環境: `ExpScissorEnv`が`ExpScissor1-v0`から`ExpScissor5-v0`を登録する。Gymnasiumの5要素`step` APIとHER互換のDict観測を使用する。
- MuJoCo: 5種類のハサミXMLには、14 actuatorのShadow Hand Lite、2つのハサミhinge関節、touch sensorが含まれる。現行移植版は公式`mujoco` APIを使用する。
- 観測・action: observationのshapeは`(67,)`、achieved goalとdesired goalのshapeは`(1,)`である。正規化actionのshapeは`(14,)`で、actuatorのctrl rangeへ写像される。
- RL: 現行経路はSB3 SAC/TD3/DDPGを使用し、SAC+HERを既定とする。HERは将来のgoalをサンプリングし、把持判定を含む報酬を再計算するため`copy_info_dict=True`を使用する。
- 既存シナジーデータ: `SynergyManager`は、成功したエピソード終盤のactionまたは姿勢サンプルを追加し、`[postures, scissors_angles]`を含むobject配列`synergy_dataset.npy`として保存する。複数成分のsklearn PCAを学習し、再構成誤差を報酬として用いるが、軌道loggerではない。
- 校正: 現行移植版には、独立して再利用できる`c=f(rho)`オブジェクトがない。旧visualization・実機スクリプトには、局所的なfitting処理とramp関数が含まれる。
- 実機: 旧ROSスクリプトはShadow Handへ関節軌道を送信し、serial readerがハサミ角度をradで返す。ただし、安定したトルク、電流、ツール状態センサ、非常停止のadapter APIは公開していない。
- テスト・設定: 参照元にはsmoke test、学習、評価スクリプトと、解決済みのJSON実行設定がある。一方、TDSの数式やコントローラ安全機構を検証するリポジトリテストはない。

## 採用した構成

新しい`improved_tds`パッケージでは、純粋な数値計算のcore moduleと、任意依存のシミュレーション・学習境界を分離する。

Gymnasium環境はTDSの比較を可能にする固定評価基盤であり、このリポジトリで新規taskや環境機能を追加しない。既存環境に対する互換性維持、明確なbug修正、再現性のための最小変更だけを許容する。報酬・観測・action・asset・物理モデルを含む環境の拡充は、専用環境リポジトリで管理する。

```text
data/          軌道schema、検証、レガシー変換器
synergy/       PCA、教師ありTDS、ファミリーTDS、校正
control/       フィードフォワード、フィードバック、アドミタンス、安定化、安全機構
tools/         ツールモデル、力推定器、実機・mock・replay interface
environments/  articulated tool API、レガシー互換ハサミ、trigger、button
learning/      成功した全軌道のcollector
evaluation/    追従指標、leave-one-instance-out評価
experiments/   CLI entry point。SB3は任意依存として維持
```

## 変更・新規ファイル

すべてのファイルは`improved-tds`内に新規作成し、参照元プロジェクトのファイルは変更していない。ライセンス付きのハサミXML・STL・texture資産と現行環境コードをコピーし、`legacy_scissors.py`として分離したうえで、articulated tool APIでラップした。正式なファイル一覧は`git status`で確認する。

## 互換性とリスクに関する判断

1. `ExpScissor1-v0`から`ExpScissor5-v0`、Dict key、`(67,)`のobservation、`(14,)`のactionを維持する。新しい`ImprovedScissor1-v0`から`ImprovedScissor5-v0`はaliasとして登録する。
2. レガシーobject配列は、`allow_pickle=True`を使用する明示的な変換器だけで読み込む。新schemaはpickleを必要とせず、ragged配列とNaNを拒否する。
3. PCAをbaselineとして維持する。すべての1次元推定器は物理関節空間の方向を正規化し、ツール状態との相関が正になるよう符号をそろえる。
4. Gymnasium、MuJoCo、SB3は任意依存の境界に置く。coreの推定・制御moduleは、これらがなくてもimportできる。SB3 SAC/TD3/DDPG+HERは任意のデータ生成baselineとし、学習中candidate DBと固定checkpoint由来のformal balanced DBを分離する。
5. ハサミのgeometry scaleや未知の実機値を暗黙に変更しない。実行時パラメータは、MuJoCoで安全に更新できる量に限定する。

## 未提供の実機依存情報

motor torque constant、gear ratio、電流・トルクの符号、センサのtimestamp・latency、ツール状態センサの校正、コントローラの通信方式、関節・力・電流制限、非常停止の意味論、実ツールgeometryは提供されていない。そのため、実機作動はprotocol、mock、offline replayの実装までに意図的に限定する。

## 実装・テスト順序

実装順序は、schema・interface、TDS・校正、安全機構・コントローラ、環境、収集・評価、実機統合とする。短時間の単体テストで数式とserializationを検証し、シミュレーションテストで登録済みの7環境を検証する。任意の学習処理と長時間実験は、別markerとして管理する。
