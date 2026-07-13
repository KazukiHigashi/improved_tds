<!-- math: mathjax -->

# TDSの数式と仮定

関節姿勢をrad単位の\(q\)、平均姿勢を\(\bar q\)、単位作動方向を\(s_a\)、1次元座標を\(\rho\)とする。

$$
q = \bar q + s_a\rho.
$$

## 推定器

PCA-TDSは、中心化した成功サンプルの第1右特異ベクトルを使用する。説明分散比はbaseline指標として保持する。教師ありTDSは関節サンプルを標準化し、1成分PLSまたは関節・ツール状態間の共分散を使用する。学習したベクトルを物理関節空間へ戻して正規化する。ツール状態を使用できる場合は、\(\operatorname{corr}(\rho,c)\ge0\)となるよう符号を選択する。

インスタンスごとの方向\(s_j\)に対し、ファミリーTDSは符号を整列した後の

$$
\sum_j w_j s_j s_j^\top
$$

の第1固有ベクトルを使用する。除外した各インスタンスでは、平均姿勢と校正を個別に保持する。

## ツール状態の校正

\(c=f(\rho)\)は、単調なlinear、piecewise-linear、isotonic、PCHIP curveとして表現する。重複する座標は平均し、可逆でない平坦区間を除去する。逆変換時の外挿は既定で無効とする。open phaseとclose phaseには別々のcurveを使用でき、重複範囲における平均差をヒステリシスとして報告する。

## フィードバックと一般化力

ツール状態フィードバックは、次式で表す。

$$
\rho_{\mathrm{cmd}}
= f^{-1}(c_d)+K_pe+K_i\int e\,dt
+K_d\,\operatorname{filtered}\!\left(\frac{de}{dt}\right).
$$

積分動作は既定で無効とし、条件付きanti-windupを使用する。TDS方向の一般化反力推定値は\(\eta_\rho=s_a^\top\tau\)である。アドミタンスは半陰的Euler法で離散化する。

$$
M\ddot\rho+D\dot\rho
=K_c(c_d-c)+K_\eta(\eta_d-\eta_\rho).
$$

low-pass filter、位置・速度・加速度の飽和、力推定欠測時のfallbackを必須とする。1つの作動DoFでは、位置と反力を独立に厳密制御できない。そのため、`eta_d`は保証された力目標ではなく、コンプライアンスまたは安全上の選好として扱う。

## 安定化

指令は

$$
q_{\mathrm{cmd}}=\bar q+s_a\rho+b_g\phi
$$

とする。PCAのnull spaceをinternal force subspaceと同一とは仮定しない。手動またはPCA由来の方向には干渉指標が必要である。実験的推定器は、力に敏感な方向から、測定したツール状態gradientと作動gradientの成分を除去する。\(\phi\)、力、ツール状態偏差、関節指令を制限し、設定した制限への違反時には非常解放を行う。
