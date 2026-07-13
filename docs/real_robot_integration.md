# 実機統合の境界

`HardwareAdapter`は、`read_state`、`write_command`、非常停止、close処理を分離する。`MockHardwareAdapter`と`OfflineReplayAdapter`を使用すると、実機がなくても同じコントローラループを実行できる。`ForceEstimator`の実装は、関節トルクの直接取得、明示的なmotor current変換、PD追従誤差proxy、決定論的mockをサポートする。

スプレーボトルまたはリモートコントローラのadapterを追加する前に、次の情報を提示し、検証する必要がある。

1. 関節名・順序、radの規約、指令通信方式
2. timestamp・clock同期、最大センサlatency
3. ツール状態センサの単位、範囲、符号、欠測時の挙動
4. トルク・電流の符号、motor torque constant、gear ratio
5. 関節、速度、加速度、電流、トルク、接触力の制限
6. 非常停止・非常解放の意味論とhardware watchdog
7. ツールgeometryと、独立レビュー済みの初期把持

シミュレーションの既定値を使用して実機指令を有効にしてはならない。bias resetは既知の無負荷状態で実行し、古い観測は欠測時のfallbackへ移行させる。すべてのlogには、コントローラ診断情報とともにraw sensor値を保持する。

