"""Build an auditable, Japanese paper-style report from frozen experiment JSON.

This script never reruns simulations or alters source results.  It uses only the
Python standard library; Plotly.js enhances charts when online, while inline
SVGs keep the report readable offline.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
from statistics import mean

from summarize_paper_one_dof import category_rows, summarize


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "analysis" / "paper15_paper_bundle_20260928"
FB = ROOT / "analysis" / "second_axis_feedback_effect_20261006"
ZERO = ROOT / "analysis" / "initial_pca2_nn_20261008"
FEW = ROOT / "analysis" / "few_shot_guarded_20261008"
TRAIN = ROOT / "runs" / "paper15_unified_3m_slew_20260927"
OUT = ROOT / "analysis" / "paper_ff_fb_generalization_20261008"
CATEGORIES = ("scissors", "push", "spray")
TASKS = tuple(f"{category}_lite_{i}" for category in CATEGORIES for i in range(1, 6))
FB_TASKS = ("scissors_lite_3", "scissors_lite_4", "push_lite_3", "spray_lite_3")
COLORS = ("#111111", "#0072B2", "#C33C3C", "#008B57", "#6C4E9A")
SYMBOLS = ("circle", "square", "diamond", "triangle-up", "x")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path, *, normalize_newlines: bool = False) -> str:
    content = path.read_bytes()
    if normalize_newlines:
        content = content.replace(b"\r\n", b"\n")
    return hashlib.sha256(content).hexdigest()


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def f2(value: float) -> str:
    return f"{value:.2f}"


def label(task: str) -> str:
    category, _, number = task.partition("_lite_")
    return {"scissors": "Sc", "push": "Pu", "spray": "Sp"}[category] + number


def table(headers: list[str], rows: list[list[object]], *, name: str = "") -> str:
    header = "".join(f"<th scope='col'>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f"<div class='table-wrap'><table aria-label='{esc(name)}'><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"


def load_data() -> dict:
    ff = summarize(BASE)  # Checks all 15 report hashes, DB sizes and timings.
    training_manifest = read(TRAIN / "manifest.json")
    if (training_manifest["training"]["cap_steps"] != 3_000_000
            or training_manifest["training"]["algorithm"].lower() != "sac"):
        raise ValueError("Selected training campaign differs")
    training = []
    for task in TASKS:
        summary = read(TRAIN / task / "model_training_summary.json")
        if summary["algorithm"].lower() != "sac":
            raise ValueError(f"Training algorithm differs: {task}")
        training.append({"task": task, "steps": int(summary["total_timesteps"]),
                         "early_stop": bool(summary["stopped_early"]),
                         "best_validation_success_percent":
                             100 * float(summary["validation"]["best"]["success_rate"])})
    fb = []
    for task in FB_TASKS:
        report = read(FB / task / "report.json")
        if report["trajectory_seconds"] != 5 or report["feedback_seconds"] != 2:
            raise ValueError(f"FB duration differs: {task}")
        if len(report["test_poses"]) != 8 or len(report["targets"]) != 9:
            raise ValueError(f"FB test design differs: {task}")
        for method in ("pca", "pls"):
            summaries = report["methods"][method]["summary"]
            if any(summaries[k]["trials"] != 72 for k in ("ff", "axis1_fb", "axis2_fb")):
                raise ValueError(f"FB trial count differs: {task}/{method}")
            fb.append({"task": task, "method": method, "summary": summaries,
                       "axis2_definition": report["methods"][method]["axis2_definition"]})

    zero = read(ZERO / "summary.json")
    if set(zero["categories"]) != set(CATEGORIES):
        raise ValueError("Zero-shot category set differs")
    # The original run records a machine-local absolute path. Preserve its
    # source fingerprint separately, but do not publish a personal home path.
    zero["source"] = str(BASE.relative_to(ROOT)).replace("\\", "/")

    few = []
    for task in TASKS:
        trial_path = FEW / f"{task}.json"
        local_path = FEW / f"{task}_local_pls.json"
        record, local = read(trial_path), read(local_path)
        if record["task"] != task or local["task"] != task:
            raise ValueError(f"Few-shot task mismatch: {task}")
        if (len(record["physical"]) != 15 or len(local["rows"]) != 15
                or record["test_poses"] != local["test_poses"]
                or record["calibration_pose"] != local["calibration_pose"]):
            raise ValueError(f"Few-shot comparison not paired: {task}")
        rows, local_rows = record["physical"], local["rows"]
        common_response = record["models"]["common"]["response"]
        selected_response = (record["models"][record["selected_candidate"]]["response"]
                             if record["selected_candidate"] is not None else None)
        few.append({
            "task": task,
            "baseline": 100 * mean(r["baseline_mae_fraction"] for r in rows),
            "fewshot": 100 * mean(r["adapted_mae_fraction"] for r in rows),
            "local": 100 * mean(r["mae_fraction"] for r in local_rows),
            "local_cal": 100 * mean(r["calibrated_mae_fraction"] for r in local_rows),
            "baseline_supported": sum(bool(r["baseline_supported"]) for r in rows),
            "fewshot_supported": sum(bool(r["adapted_supported"]) for r in rows),
            "local_supported": sum(bool(r["supported"]) for r in local_rows),
            "local_cal_supported": sum(bool(r["calibrated_supported"]) for r in local_rows),
            "probes": record["probes_attempted"],
            "selected": record["selected_candidate"],
            "common_probe_coverage": (100 * common_response["coverage_fraction"]
                                      if common_response is not None else None),
            "selected_probe_coverage": (100 * selected_response["coverage_fraction"]
                                        if selected_response is not None else None),
            "guard": record["common_coverage_stop"],
            "stopped_after_common": record["stopped_after_common"],
        })
    if any(row["guard"] != .75 for row in few):
        raise ValueError("Few-shot stop guard differs")

    return {"ff": ff, "ff_category": category_rows(ff), "training": training,
            "training_protocol": training_manifest["training"], "fb": fb,
            "zero": zero, "few": few,
            "source_hashes": {
                "ff_manifest_sha256": sha(BASE / "manifest.json"),
                "training_manifest_sha256": sha(TRAIN / "manifest.json"),
                "fb_manifest_sha256": sha(FB / "manifest.json"),
                "zero_summary_sha256": sha(ZERO / "summary.json"),
                "few_validator_sha256": sha(ROOT / "scripts" / "validate_guarded_few_shot.py",
                                             normalize_newlines=True),
            }}


def svg_plot(labels: list[str], series: list[dict], ylabel: str, maximum: float) -> str:
    """Deterministic vector fallback; no CDN or Python Plotly dependency."""
    width, height = 1060, 360
    left, right, top, bottom = 70, 15, 9, 66
    plot_w, plot_h = width - left - right, height - top - bottom
    n = len(labels)
    interval = plot_w / max(n, 1)
    points = [f"<svg xmlns='http://www.w3.org/2000/svg' role='img' aria-label='{esc(ylabel)}' viewBox='0 0 {width} {height}'><rect width='{width}' height='{height}' fill='white'/>"
              f"<path d='M{left},{top}V{top+plot_h}H{left+plot_w}' fill='none' stroke='#111' stroke-width='1'/>" ]
    for value in range(0, int(math.ceil(maximum / 10)) * 10 + 1, 10 if maximum <= 60 else 20):
        if value > maximum:
            continue
        y = top + plot_h * (1 - value / maximum)
        points.append(f"<path d='M{left},{y:.1f}h6' stroke='#111'/><text x='{left-8}' y='{y+4:.1f}' text-anchor='end' font-size='13' font-family='Times New Roman'>{value}</text>")
    points.append(f"<text x='15' y='{top+plot_h/2:.1f}' transform='rotate(-90 15 {top+plot_h/2:.1f})' text-anchor='middle' font-size='15' font-family='Times New Roman'>{esc(ylabel)}</text>")
    for i, item in enumerate(labels):
        x = left + interval * (i + .5)
        points.append(f"<path d='M{x:.1f},{top+plot_h}v-6' stroke='#111'/><text transform='translate({x:.1f},{top+plot_h+13}) rotate(-40)' text-anchor='end' font-size='13' font-family='Times New Roman'>{esc(item)}</text>")
    for j, item in enumerate(series):
        offset = (j - (len(series) - 1) / 2) * min(11, interval / (len(series) + 1))
        for i, value in enumerate(item["values"]):
            if value is None or not math.isfinite(value):
                continue
            x = left + interval * (i + .5) + offset
            y = top + plot_h * (1 - max(0, min(value, maximum)) / maximum)
            color = item["color"]
            if j % 3 == 0:
                points.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='4.4' fill='white' stroke='{color}' stroke-width='2'/>")
            elif j % 3 == 1:
                points.append(f"<rect x='{x-4:.1f}' y='{y-4:.1f}' width='8' height='8' fill='{color}'/>")
            else:
                points.append(f"<path d='M{x:.1f},{y-5:.1f}l5,5 -5,5 -5,-5Z' fill='{color}'/>")
    return "".join(points) + "</svg>"


def figure(fig_id: str, labels: list[str], values: list[tuple[str, list[float]]],
           ylabel: str, caption: str, maximum: float | None = None) -> tuple[str, dict]:
    finite_values = [v for _, vs in values for v in vs
                     if v is not None and math.isfinite(v)]
    upper = maximum or max(10., max(finite_values, default=0.) * 1.12)
    series = [{"name": name, "values": vals, "color": COLORS[i], "symbol": SYMBOLS[i]}
              for i, (name, vals) in enumerate(values)]
    legend = "".join(f"<span class='legend-item'><span class='swatch' style='--series:{item['color']}'></span>{esc(item['name'])}</span>" for item in series)
    fallback = svg_plot(labels, series, ylabel, upper)
    block = f"<figure><div class='legend'>{legend}</div><div class='graph' id='{esc(fig_id)}'>{fallback}</div><figcaption>{esc(caption)}</figcaption></figure>"
    return block, {"id": fig_id, "labels": labels, "series": series, "ylabel": ylabel,
                   "maximum": upper}


def equation(tex: str, number: int) -> str:
    return f"<div class='equation' role='math' aria-label='{esc(tex)}'>\\[{esc(tex)}\\]<span class='eqno'>({number})</span></div>"


def category_mean(rows: list[dict], category: str, key: str) -> float:
    return mean(row[key] for row in rows if row["task"].startswith(category))


def build_html(data: dict) -> str:
    sections: list[str] = []
    plots: list[dict] = []
    ff, ff_cat, training, fb, zero, few = (
        data[k] for k in ("ff", "ff_category", "training", "fb", "zero", "few"))
    sections.append("""<header><p class='kicker'>研究論文用・再集計ドラフト / 2026-10-08</p><h1>成功姿勢から得るツール操作シナジーの<br>PCA・PLS比較と汎化評価</h1><p class='subtitle'>1自由度フィードフォワード制御を主実験とし、閉ループ補正および未知ツールへの適応を探索的拡張として区別する。</p><div class='badge'>保存済み実験結果の再集計。新しい学習・シミュレーションは実施していない。</div></header>""")
    sections[-1] = sections[-1].replace(
        "保存済み実験結果の再集計。新しい学習・シミュレーションは実施していない。",
        "旧図表は保存済み実験の再集計。図解には別途実施した共通9目標再評価の実測例を使用。")
    sections.append("""<nav><a href='#abstract'>要旨</a><a href='#terms'>用語と読み方</a><a href='#methods'>手法</a><a href='#conditions'>評価条件</a><a href='#pose-indices'>姿勢ID</a><a href='#training'>学習条件</a><a href='#ff'>FF結果</a><a href='#fb'>FB結果</a><a href='#generalization'>汎化</a><a href='#discussion'>考察</a><a href='#audit'>整合性・倫理監査</a></nav>""")
    sections[-1] = sections[-1].replace("<a href='#conditions'>評価条件</a>",
        "<a href='#conditions'>評価条件</a><a href='#visual-guide'>手順・誤差の図解</a>")
    sections.append("""<section id='abstract'><h2>要旨</h2><p>Scissors、Push、Spray各5変種の成功最終姿勢を各200件、計3,000件集め、関節を一つの得点で動かすツール操作シナジーを抽出した。PCAは姿勢のばらつき、PLSは姿勢とツール状態の共変動から操作方向を求める。主実験は5秒の開ループ移動と2秒の保持で評価し、4タスクの閉ループ補正と15ツールの転用を探索的な追加実験として分けた。旧実験のMAEは評価目標の置き方が異なるため、図をまたいだ直接の優劣比較には用いない。共通9目標の新規再評価は別の報告に記す。</p></section>""")
    sections.append("""<section id='terms'><h2>最初に読む：用語と数値の意味</h2><p>この報告で「ツール」とは、はさみ・プッシュ機構・スプレー機構の各変種を指す。「初期姿勢」は操作前にハンドと対象物を配置するリセット状態、「成功姿勢」は強化学習で目標を達成した後に保存するハンドの最終姿勢である。両者は別のデータであり、成功姿勢200件は初期姿勢200件を意味しない。</p>""" + table(["用語", "この報告での意味", "数字を見るときの注意"], [
        ["TDS／操作シナジー", "Tool-DoF Synergyの略。多数のハンド関節を一つの得点で動かす方向。", "Scissors・Sprayは角度[rad]、Pushは押し込み変位[m]を操作する。"],
        ["RL・SAC・HER", "RLは強化学習、SACは今回の学習アルゴリズム、HERは使用していない目標再利用手法。", "RLの検証成功率はTDSの保持率ではない。"],
        ["DB・NN・PC1/PC2", "DBはデータベース、NNは最近傍法、PC1/PC2はPCAの第1/第2主成分。", "PLS残差のPC1をPLS第2潜在成分と呼ばない。"],
        ["FF（フィードフォワード）", "目標状態から計算した姿勢へ5秒で移動し、その後2秒保持する。", "移動中の状態誤差では指令を直さない。"],
        ["FB（フィードバック）", "FF移動後の2秒間、実測したツール状態の誤差で指令を補正する。", "第1軸FBと第2軸FBは補正に使う方向が違う。"],
        ["較正／逆写像", "得点を試して実際のツール状態を測り、『欲しい状態→必要な得点』を決める。", "較正用の試行と誤差評価用の試行を区別する。"],
        ["逆写像可能域", "較正で単調な応答を確認し、逆写像の使用を認めた目標状態の範囲。", "範囲外の誤差を含まない旧FFの条件付きMAEとは別に全域評価が必要。"],
        ["独立スイープ", "一つの得点ごとに初期姿勢へ戻して操作し、到達する状態を測る手続き。", "連続して往復するスイープとは異なる。"],
        ["プローブ", "未知ツールで得点を一つ試し、実際のツール状態を測る1回の試行。", "3回で済む場合と、候補5軸を試して15回かかる場合がある。"],
        ["ゼロショット／Few-shot", "前者は未知ツールの操作結果を測らずに制御、後者は少数のプローブ後に制御する。", "Few-shotの誤差をゼロショットと比較するときは実測の追加コストも示す。"],
        ["保持MAE", "所定の保持条件を満たした試行だけの平均絶対誤差。", "ScissorsはPalm球内、Push/Sprayは落下・転倒なし。保持率と対で読む。"],
        ["全有限終端MAE", "終端のツール状態が数値として得られた試行の平均絶対誤差。", "保持に失敗した有限試行も含む。未観測件数は別に示す。"],
        ["Coverage／到達幅", "RL目標範囲内に入った実測最大値と最小値の差を、目標幅で割る。", "100%が上限。較正幅と独立スイープ幅は別の測定量。"],
        ["LOTO／自ツールDB参照", "5変種から1つを未知とし、残り4変種で共通軸を作る。参照用PLSは未知変種のDBで作る。", "参照は性能の上限保証ではなく、ゼロショットでは使えない情報を持つ。"],
    ], name="初出用語の平易な定義") + "<p>例えば『MAE 20%』はツールが20%だけ動いたという意味ではない。目標との差の平均が、そのタスクのRL目標幅の20%に相当する。目標を無視して中央状態に固定するだけの数式上の誤差は旧5目標で24%、新9目標で27.78%だが、これは把持・操作できる物理対照ではない。</p></section>")
    method = ["<section id='methods'><h2>1. 手法</h2><h3>1.1 強化学習と成功姿勢データ</h3><p>Scissors、Push、Spray各5変種に対し、同一キャンペーンでSACを新規学習した。学習上限は各300万環境ステップであり、各タスクが300万ステップを実際に走ったという意味ではない。10万ステップごとに検証・チェックポイント保存を行い、検証成功率95%以上を2回連続で達成した場合に早期終了した。実行は同時4学習プロセス×各4環境、HERは用いず、目標指令範囲を0〜100%とした。行動変化率制限と行動変化ペナルティを用いる。共通報酬項は進捗・把持・成功ボーナス・失敗ペナルティ・制御変化/大きさであるが、追従項と接触補助項はカテゴリ別で同一係数ではない。従って報酬関数が15タスクで完全同一とは書かない。</p><p>各タスクの保存済みベストポリシーで成功した最終姿勢を目標角度10区間にわたって計200件収集し、対応するツール状態を付与した。PCA/PLSは同じ姿勢集合の全体平均で中心化する。PLSは1成分、<code>scale=False</code>で学習し、ツール状態との共分散が正となるよう軸の符号を反転する。第2成分を使用しない主実験では、独立操作指令は一つだけである。</p>"]
    method.append(equation(r"\widetilde{X}=X-\mathbf{1}\bar{\boldsymbol q}^{\mathsf T},\qquad \widetilde{\boldsymbol y}=\boldsymbol y-\bar y\mathbf{1}.", 1))
    method.append(equation(r"\boldsymbol w_{\mathrm{PCA}}=\underset{\|\boldsymbol w\|=1}{\arg\max}\;\|\widetilde X\boldsymbol w\|_2^2,\qquad \boldsymbol w_{\mathrm{PLS}}=\underset{\|\boldsymbol w\|=1}{\arg\max}\;\operatorname{Cov}(\widetilde X\boldsymbol w,\widetilde{\boldsymbol y})^2.", 2))
    method.append("<p>上式は操作方向の直観的表現である。実装ではPCAにSVD、PLSにscikit-learnの<code>PLSRegression(n_components=1, scale=False)</code>の<code>x_weights_</code>を用いる。姿勢単位の中心化は主実験には適用しない。</p><h3>1.2 1自由度FF制御と較正</h3>")
    method.append(equation(r"\boldsymbol q_{\mathrm{goal}}(\rho)=\operatorname{clip}_{[\boldsymbol q_{\min},\boldsymbol q_{\max}]}\!\left(\bar{\boldsymbol q}+\rho s\boldsymbol w\right),\qquad \boldsymbol q_{\mathrm{cmd}}(t)=\left(1-\frac{t}{5\,\mathrm{s}}\right)\boldsymbol q_0+\frac{t}{5\,\mathrm{s}}\boldsymbol q_{\mathrm{goal}},\;0\leq t\leq5\,\mathrm{s}.", 3))
    method.append("<p>式(3)のρは無次元の操作得点、sは軸を関節角の移動幅へ換算する尺度係数である。成功姿勢DB上の操作軸得点の最小・最大をそれぞれ1.5倍して17点を取る。較正用の8初期姿勢で各点を独立リセット・軌道制御し、採用できた応答から概ね単調な区間を選ぶ。等張回帰は測定点の順序を保つ単調な近似、ランプ関数は直線部と両端の一定部をもつ近似であり、同じ較正データで両者を作る。報告する旧主MAEは等張回帰の逆写像を、別の8初期姿勢と方式別の較正済み9目標で評価した値である。この8姿勢は較正では使っていないが、学習と同じ32姿勢プールから選んだ。ランプ関数の誤差は副比較として分ける。5秒移動後に2秒保持し、評価中はRL・オンラインのツール状態FBを使わない。</p>")
    method.append(equation(r"\hat f=\underset{f\,\mathrm{monotone}}{\arg\min}\sum_{j\in\mathcal C}\left(f(\rho_j)-\bar y_j\right)^2,\qquad \rho^*(y^*)=\hat f^{-1}(y^*)\quad(y^*\in\mathcal D_{\mathrm{inv}}).", 4))
    method.append(equation(r"\mathrm{MAE}_{\mathrm{norm}}=\frac{100}{N(y_{\max}-y_{\min})}\sum_{i=1}^{N}|y_i^{\mathrm{end}}-y_i^*|\;[\%],\qquad \mathrm{Coverage}=\frac{100\max\{0,\min(y_{\max},y_{\mathrm{reach,max}})-\max(y_{\min},y_{\mathrm{reach,min}})\}}{y_{\max}-y_{\min}}\;[\%].", 5))
    method.append("<p class='caution'>式(5)の到達幅は実装ではRL目標範囲内に切り詰める。独立スイープの到達幅と、較正から認めた逆写像可能幅は別指標である。旧MAEのNは条件を満たして終端が得られた試行数で、PCA/PLSの9目標も互いに異なりうる。72試行は8初期姿勢×9目標であり、72の独立初期姿勢ではない。</p><h3>1.3 探索的な閉ループ補正</h3><p>Scissors 3/4、Push 3、Spray 3に限り、共通の逆写像目標域・初期姿勢・乱数条件で、①FF＋保持、②第1軸FB、③第2軸FBを比較する。5秒のFF移動後、2秒間だけツール状態の誤差に基づく補正を加える。第2軸はPCAではPC2、PLSでは<strong>PLS第2潜在成分ではなく</strong>操作軸を除去した姿勢残差のPC1である。各評価目標そのもので、較正用8姿勢×3得点×2補正軸＝48試行の感度計測を行い、別の8姿勢へ適用する。未知目標へのFB汎化は評価していない。</p>")
    method.append(equation(r"e_k=y^*-y_k,\qquad g_{k+1}=\operatorname{clip}\!\left(g_k+K\frac{e_k}{\hat b}\Delta t,\;g_{\min}^{\mathrm{cal}},g_{\max}^{\mathrm{cal}}\right),\quad \hat b\approx\frac{\Delta y}{\Delta g}.", 6))
    method.append("<p>FB感度は、較正姿勢で無次元の補正座標gを±0.1だけ変えた応答から推定する。設計上限は±0.5で、今回の補正は計測した±0.1範囲内に制限した。この制限による飽和と停止は結果の一部であり、無効感度の試行も総数から除かない。評価姿勢は学習時と同じ32件のプールに含まれる。これは最適化済みの実機制御器ではなく、シミュレーション上の探索である。</p><h3>1.4 未知ツールへの汎化</h3><p>同一カテゴリの5変種から1つを未知ツールとして除き、残り4変種で軸を作るLeave-one-tool-out（LOTO）を行う。未知ツールの成功姿勢はゼロショット・Few-shot軸の構築に使わない。ゼロショットでは、評価初期姿勢ごとに2次元PCA空間の最近傍を探し、既知ツールから中心姿勢・操作軸・DB直線回帰の逆写像の組を選ぶ。Few-shotはまず共通軸を3回試し、実測した到達幅が75%未満なら既知4ツール由来の軸も各3回試すため最大15回である。採用軸の実測3点で逆写像を作り、<strong>較正に使っていない初期姿勢3件×5目標</strong>で評価する。未知ツール自身の成功姿勢DBで作ったPLSは情報量の異なる参照であり、性能の上限保証ではない。</p>")
    method.append(equation(r"\mathcal T_{-h}=\{1,\ldots,5\}\setminus\{h\},\qquad c^*=\underset{c\in\{\mathrm{common},\mathcal T_{-h}\}}{\arg\max}\;\widehat{\mathrm{Coverage}}_c,\qquad \hat\rho_h(\theta^*)=\hat f_{c^*,h}^{-1}(\theta^*).", 7))
    method.append("<p>式(7)は候補選択の概要であり、実装の無効サンプル・単調応答・タイブレーク規則はソースに従う。75%の停止閾値は先行相の結果を見て決めており、独立な未知ツール集合による事前登録済み確認ではない。</p><h3>1.5 手続きごとの成功・採用条件</h3>")
    method.append(table(["対象", "学習/成功DB", "FF較正・評価", "FB/Few-shotの扱い"], [
        ["Scissors", "Palm追従球半径0.05 m内にヒンジ中心。成功最終姿勢200件/タスク。", "Palm追従球半径0.075 m（1.5倍）内を採用。", "それぞれ保存報告の保持・終端有効性で判定。FFの条件を自動的に同一視しない。"],
        ["Push", "選定済みベストポリシーの成功最終姿勢200件/タスク。", "接触数ゲートを使わず、落下・転倒を除外。", "Few-shotプローブは終端支持・数値健全性を確認。"],
        ["Spray", "選定済みベストポリシーの成功最終姿勢200件/タスク。", "接触数ゲートを使わず、落下・転倒を除外。床接触は有効。", "同上。"],
    ], name="手続きごとの採用条件"))
    method.append("<p class='caution'>RLの成功率、成功姿勢DBへの採用率、TDS較正サンプルの採用率、TDS終端保持率は異なる母集団・定義を持つ。混同して成功率と呼ばない。</p></section>")
    sections.append("".join(method))

    sections.append("""<section id='conditions'><h2>評価条件の対応と、共通9目標再評価</h2><p>15ツールの各初期姿勢プールは<strong>32件</strong>である。1件は13個の駆動関節角とハンドの取付位置・向きからなる。姿勢プールはツールごとに異なる。学習時の各リセットはこの32件から乱数で選び、学習中の検証は<strong>同じ32件</strong>と11目標角度を順に使う。よって学習の検証成功率を『未知初期姿勢での成功率』とは呼べない。強化学習が作った成功最終姿勢200件/ツールも初期姿勢プールとは別物である。LOTOでは、未知とするツールの成功最終姿勢200件は共通軸・候補軸の学習に入れない。既知の4ツールから計800件を用いる。ただしタスク別PLS参照だけは未知ツールの200件を使うため、情報量が異なる。</p>""" + table(["保存済み実験", "較正・選択に使う初期姿勢", "誤差評価に使う初期姿勢", "目標角度の置き方", "試行数/ツール・方式"], [
        ["15タスクPCA/PLS FF", "32件から8件。17得点の応答を測る。", "別の8件。較正姿勢とは重複しない。", "方式ごとの逆写像可能角度域に等間隔で9点。", "8×9＝72"],
        ["4タスクFB", "8件。目標近傍の感度も測る。", "別の8件。FF/第1軸FB/第2軸FBで共有。", "比較条件で共有する逆写像目標域の9点。", "8×9＝72"],
        ["未知ツール・ゼロショット", "未知ツールでの物理較正は0件。既知4ツールのDBを使用。", "32件から乱数で選ぶ3件。", "RL目標角度幅の10、30、50、70、90%の5点。", "3×5＝15"],
        ["未知ツール・Few-shot後続相", "過去相の4件を除く残りから1件。共通軸3回、必要時は他4軸も各3回。", "同じ抽選で選ぶ別の3件。較正姿勢とは重複しない。", "RL目標角度幅の10、30、50、70、90%の5点。", "3×5＝15"],
    ], name="保存済み実験の初期姿勢と目標値の違い") + "<p>後続相で除く『過去相の4件』は、先行Few-shot相の較正1件とゼロショット相で用いた評価3件である。残り28件から後続相の較正1件・評価3件を選ぶ。一方、主実験FFの8＋8件と後続Few-shot相は、互いに重ならないよう設計された別分割ではない。従って、現在のFF図と未知ツール図のMAEは試行条件を揃えた直接比較ではない。</p><p class='caution'><strong>追加実験待ち：9目標の統一評価は未実施。</strong>件数だけを5点から9点に増やしても不十分である。主実験FFの9点は方式別の較正可能域内だが、未知ツールの5点はRL目標域内にある。公平な新評価では、各ツールのRL目標域に共通の9角度（0、12.5、25、37.5、50、62.5、75、87.5、100%）を固定し、同じ8評価初期姿勢・同じリセット種・5秒移動＋2秒保持で全方式を試す。角度が較正範囲外なら指令を端点に制限したことを記録し、その試行も全有限終端MAEに含める。保持MAEと保持率、数値的失敗件数、較正可能域幅も併記する。較正試行を評価に再利用しない。</p><p>この条件に厳密に揃えるには、凍結済みモデルと姿勢DBを再利用して<strong>FF主実験・FB比較・未知ツール評価を再シミュレーション</strong>する必要がある。新たな強化学習は不要である。既存の5点結果や方式別9点結果は履歴として残し、新9点の数値と混ぜない。評価姿勢は各ツール32件から事前に固定したFFの8評価IDを全条件で共有し、Few-shotの3点プローブにはその8件と異なる1件を使う。過去の実験で同じ姿勢群を調べているため、この再評価を『完全に未見のツール・姿勢での確認』とは呼ばない。追加実験はユーザーの指示待ちであり、この報告には新条件のMAEを載せない。</p></section>""")

    # Keep the historical protocol description but link to the completed,
    # separately reported matched-grid replay instead of calling it pending.
    sections[-1] = (sections[-1]
        .replace("評価条件の対応と、未実施の統一評価", "評価条件の対応と、共通9目標再評価")
        .replace("追加実験待ち：9目標の統一評価は未実施。", "共通9目標の統一再評価は完了した。")
        .replace("公平な新評価では", "新評価では")
        .replace("この条件に厳密に揃えるには、", "この条件では、")
        .replace("する必要がある。新たな強化学習は不要である。", "した。新たな強化学習は行っていない。")
        .replace("追加実験はユーザーの指示待ちであり、この報告には新条件のMAEを載せない。",
                 "新条件の数値は<a href='../matched_nine_goals_clipped_20261008/index.html'>共通9目標の報告</a>に分け、旧条件の数値と混ぜない。"))
    # Show the diagrams directly in this paper report, not only in its linked
    # matched-grid companion.  The accompanying prose separates old and new
    # target protocols so the frozen illustrative nine points are not passed
    # off as the old method-specific FF evaluation.
    from build_matched_nine_report import mae_illustration, protocol_flowchart
    matched = read(ROOT / "analysis/matched_nine_goals_clipped_20261008/summary.json")
    if (matched["protocol"] != "common_full_rl_nine_goal_v1"
            or len(matched["mae_illustration"]["rows"]) != 9):
        raise ValueError("matched-grid diagram source differs")
    sections.append(
        "<section id='visual-guide'><h2>図解：実験は何を行い、誤差をどう計算するか</h2>"
        "<p>下のフローチャートは、<strong>共通9目標で再評価した最新キャンペーン</strong>を表す。"
        "学習済み方策・成功最終姿勢を出発点に、PCA/PLSの1自由度FF、未知ツールへの転用、"
        "選定4タスクのFBへ分かれ、各ツールの同じ8評価初期姿勢×9目標へ合流する。"
        "較正と評価には別の初期姿勢IDを使う。ただし、各ツールの32姿勢プール自体は学習時にも使っており、"
        "未知の初期姿勢分布を調べた実験ではない。</p>"
        + protocol_flowchart()
        + "<p>このHTMLの後半に残す<strong>従来の結果</strong>は目標設定が異なる。"
        "従来の主FFは方式ごとの逆写像可能域内で9目標、従来の未知ツール実験はRL目標幅内で5目標を用いた。"
        "したがって旧図表のMAEをそのまま横並びに引き算しない。"
        "共通9目標での同条件比較値は"
        "<a href='../matched_nine_goals_clipped_20261008/index.html'>新しい再評価の数値表とグラフ</a>に示す。</p>"
        "<h3>目標と実測の『差』がMAEになる</h3>"
        "<p>下図は共通9目標の保存済み実測値を用いた計算例である。横軸の目標値と縦軸の実測値が一致すれば"
        "黒破線上に点が載る。点から黒破線までの縦方向の距離が、その試行の絶対誤差である。"
        "Pushでは角度[rad]の代わりに変位[m]を使い、分母は各ツールのRL目標幅に置き換える。</p>"
        + mae_illustration(matched["mae_illustration"])
        + "<p>各方式の主MAEは、完走して有限の終端値が得られた試行の正規化誤差を平均する。"
        "保持に失敗しても終端が有限なら含め、途中終了・数値異常は誤差0とせず欠測件数として別に示す。"
        "保持試行だけのMAE、保持率、ツール状態の到達幅は別指標である。"
        "この実測例の9点平均は1姿勢の説明値であり、全8姿勢の方式別MAEと同一視しない。</p></section>")
    pose_rows = []
    for task in TASKS:
        category = task.partition("_lite_")[0]
        ff_report = read(BASE / task / "report.json")
        zero_poses = sorted({int(row["pose"])
                             for row in zero["categories"][category]["physical"]
                             if row["task"] == task and row["arm"] == "common_axis"})
        few_report = read(FEW / f"{task}.json")
        ff_cal = list(map(int, ff_report["calibration_poses"]))
        ff_test = list(map(int, ff_report["test_poses"]))
        few_cal = int(few_report["calibration_pose"])
        few_test = list(map(int, few_report["test_poses"]))
        if (len(ff_cal) != 8 or len(ff_test) != 8 or len(zero_poses) != 3
                or len(few_test) != 3 or set(ff_cal) & set(ff_test)
                or few_cal in few_test or set(zero_poses) & ({few_cal} | set(few_test))):
            raise ValueError(f"Unexpected initial-pose split: {task}")
        pose_rows.append([task, "0–31 (32件)", ", ".join(map(str, ff_cal)),
                          ", ".join(map(str, ff_test)), ", ".join(map(str, zero_poses)),
                          str(few_cal), ", ".join(map(str, few_test))])
    fb_pose_rows = []
    for task in FB_TASKS:
        report = read(FB / task / "report.json")
        fb_pose_rows.append([task,
                             ", ".join(map(str, report["calibration_poses"])),
                             ", ".join(map(str, report["test_poses"]))])
    sections.append("<section id='pose-indices'><h2>各ツールの初期姿勢ID</h2><p>表のIDは各ツール専用プール内の0〜31番である。同じ番号でもツールが違えば同じ姿勢を指さない。ID表は分割の追跡用であり、以下の試行を新しく実施したことを意味しない。</p><details><summary>15ツールのFF・ゼロショット・Few-shot姿勢IDを開く</summary>" +
                    table(["ツール", "プール", "FF較正8", "FF評価8", "ゼロショット評価3", "Few-shot較正1", "Few-shot評価3"],
                          pose_rows, name="ツール別初期姿勢ID") +
                    "</details><details><summary>4タスクのFB較正・評価姿勢IDを開く</summary>" +
                    table(["ツール", "FB較正8", "FB評価8"], fb_pose_rows,
                          name="フィードバック実験の初期姿勢ID") + "</details></section>")

    training_table = table(["タスク", "実学習ステップ", "早期停止", "最良検証成功率 [%]"], [
        [r["task"], f"{r['steps']:,}", "あり" if r["early_stop"] else "なし",
         f2(r["best_validation_success_percent"])] for r in training],
        name="学習実施量と検証成功率")
    sections.append("<section id='training'><h2>学習実施量とDBの来歴</h2><p>Scissors 1〜5は30〜70万ステップ、Push 1〜5は20〜30万ステップで早期停止した。Spray 1〜3は上限300万ステップに達しており、その最良検証成功率は95%未満である。それでも選定済みベストポリシーから条件を満たす成功最終姿勢を各200件収集している。学習成功率と成功姿勢DBの構築可能性を混同しない。カテゴリ間で同じ上限を用いたが、実学習ステップも報酬詳細も一致しない。</p>" + training_table + "<p class='source'>出典: <a href='../../runs/paper15_unified_3m_slew_20260927/manifest.json'>学習キャンペーンmanifest</a>と各タスクのmodel_training_summary.json。最良検証成功率は選定チェックポイントの検証値であり、FF/TDSの成功率ではない。</p></section>")

    ff_map = {(r["task"], r["method"]): r for r in ff}
    labels = [label(t) for t in TASKS]
    ff_cov, fig = figure("fig-ff-cov", labels,
        [("PCA", [ff_map[t, "pca"]["independent_coverage_percent"] for t in TASKS]),
         ("PLS", [ff_map[t, "pls"]["independent_coverage_percent"] for t in TASKS])],
        "独立スイープCoverage [%]", "図1. 各タスクでの独立スイープ到達幅。分母は各RL目標幅。較正の逆写像可能幅ではない。", 105)
    plots.append(fig)
    ff_mae, fig = figure("fig-ff-mae", labels,
        [("PCA", [ff_map[t, "pca"]["normalized_held_mae_percent"] for t in TASKS]),
         ("PLS", [ff_map[t, "pls"]["normalized_held_mae_percent"] for t in TASKS])],
        "保持MAE / RL目標幅 [%]", "図2. 方式別逆写像可能域内の採用試行に限る条件付きMAE。PCA/PLSの目標集合が完全一致する比較ではない。")
    plots.append(fig)
    ff_summary = table(["カテゴリ", "方式", "逆写像可能幅 [%]", "独立Coverage [%]", "保持MAE [%]", "保持/全試行"], [
        [r["category"], r["method"].upper(), f2(r["inverse_domain_percent"]),
         f2(r["independent_coverage_percent"]), f2(r["normalized_held_mae_percent"]),
         f"{r['held_trials']}/{r['total_trials']}"] for r in ff_cat], name="15タスクFF集計")
    ff_detail = table(["タスク", "PCA Coverage [%]", "PLS Coverage [%]", "PCA MAE [%]", "PLS MAE [%]", "PCA保持", "PLS保持"], [
        [t, f2(ff_map[t, "pca"]["independent_coverage_percent"]),
         f2(ff_map[t, "pls"]["independent_coverage_percent"]),
         f2(ff_map[t, "pca"]["normalized_held_mae_percent"]),
         f2(ff_map[t, "pls"]["normalized_held_mae_percent"]),
         f"{ff_map[t, 'pca']['held_trials']}/72", f"{ff_map[t, 'pls']['held_trials']}/72"] for t in TASKS], name="タスク別FF全結果")
    sections.append(f"<section id='ff'><h2>2. 15タスクの1自由度FF結果</h2>{ff_summary}<p>PLSはPushで操作可能幅を大きく広げたが、Sprayでは平均保持MAEがPCAより大きい。従って一律の優位性は主張しない。 ScissorsではPLSの逆写像可能幅と独立Coverageが共に大きい。</p>{ff_cov}{ff_mae}<details><summary>15タスクの全数値表を開く</summary>{ff_detail}</details><p class='source'>出典: <a href='../paper15_paper_bundle_20260928/manifest.json'>選定済みmanifest</a>、各タスクのreport.json。<code>scripts/summarize_paper_one_dof.py</code>が15件の選定レポートSHA-256、成功姿勢200件、5秒＋2秒、各72試行を検証して再集計。</p></section>")

    fb_labels = [label(r["task"]) + "-" + r["method"].upper() for r in fb]
    fb_fig, fig = figure("fig-fb", fb_labels, [
        ("FF＋保持", [r["summary"]["ff"]["all_finite_mae_percent"] for r in fb]),
        ("第1軸FB", [r["summary"]["axis1_fb"]["all_finite_mae_percent"] for r in fb]),
        ("第2軸FB", [r["summary"]["axis2_fb"]["all_finite_mae_percent"] for r in fb])],
        "全有限終端MAE / RL目標幅 [%]", "図3. 同じ評価姿勢・目標を使った4タスクの探索的FB比較。第2軸無効試行もその条件の総平均に残す。")
    plots.append(fig)
    fb_table = table(["タスク/軸", "FF [%]", "第1軸FB [%]", "第2軸FB [%]", "第2軸較正成立", "第2軸停止", "FF転倒"], [
        [f"{r['task']} / {r['method'].upper()}",
         f2(r["summary"]["ff"]["all_finite_mae_percent"]),
         f2(r["summary"]["axis1_fb"]["all_finite_mae_percent"]),
         f2(r["summary"]["axis2_fb"]["all_finite_mae_percent"]),
         f"{r['summary']['axis2_fb']['enabled_trials']}/72",
         r["summary"]["axis2_fb"]["stopped_feedback"],
         r["summary"]["ff"]["tipped_trials"]] for r in fb], name="FB全条件")
    sections.append(f"<section id='fb'><h2>3. 第1軸・第2軸FBの探索的比較</h2><p>第1軸FBはこの4タスク・両抽出方式の旧評価で平均MAEを下げた。ただし差の大きさは条件ごとに違い、Scissors 4のPCAは3.37→3.00%である。第2軸FBはFFより低誤差の条件もあるが、第1軸FBを上回る一貫した結果は得られない。感度較正不成立、指令飽和、FB停止を無視した優劣判断はできない。</p>{fb_fig}{fb_table}<p class='caution'>軌道中に一度でも転倒が診断されたFF試行はScissors 3でPCA 23/72・PLS 24/72、Scissors 4でPCA 55/72・PLS 57/72だった。一方、終端のPalm球基準では有効と集計されうる。したがって有効率は軌道全体の安全な把持を保証しない。8評価姿勢×9目標の階層構造と4タスクへの限定により、図3の差を一般的な統計的有意差と解釈しない。</p><p class='source'>出典: <a href='../second_axis_feedback_effect_20261006/manifest.json'>FB実行manifest</a>、各タスクreport.json・保存ソース。各条件72試行、5秒FF＋2秒保持/FB、較正8姿勢・評価8姿勢。</p></section>")

    zcats = list(CATEGORIES)
    zkeys = [("共通軸", "common_axis"), ("初期姿勢13D-NN", "raw13_nn_package"),
             ("初期姿勢PCA2-NN", "pca2_nn_package"),
             ("PCA2選択軸のみ", "pca2_selected_axis_only"),
             ("自ツールDB参照", "task_specific_reference")]
    zero_fig, fig = figure("fig-zero", [c.title() for c in zcats], [
        (name, [100 * zero["categories"][c]["by_arm"][key]["mae_fraction"] for c in zcats])
        for name, key in zkeys], "MAE / RL目標幅 [%]",
        "図4. 旧条件のLOTO物理評価。自ツールDB参照は未知ツールの成功姿勢を使うため、ゼロショット条件でも性能上限でもない。")
    plots.append(fig)
    zero_table = table(["カテゴリ", "初期姿勢PCA2説明分散 [%]", "共通軸MAE [%]", "PCA2-NN MAE [%]", "PCA2軸のみMAE [%]", "自ツールDB参照MAE [%]"], [
        [c, f2(100 * zero["categories"][c]["offline"]["mean_cumulative_pca2_variance"]),
         f2(100 * zero["categories"][c]["by_arm"]["common_axis"]["mae_fraction"]),
         f2(100 * zero["categories"][c]["by_arm"]["pca2_nn_package"]["mae_fraction"]),
         f2(100 * zero["categories"][c]["by_arm"]["pca2_selected_axis_only"]["mae_fraction"]),
         f2(100 * zero["categories"][c]["by_arm"]["task_specific_reference"]["mae_fraction"])] for c in zcats], name="ゼロショット結果")
    few_fig, fig = figure("fig-few", labels, [
        ("共通軸FF", [r["baseline"] for r in few]),
        ("Few-shot適応", [r["fewshot"] for r in few]),
        ("タスク別PLS（DB）", [r["local"] for r in few]),
        ("タスク別PLS＋3点較正", [r["local_cal"] for r in few])],
        "全15評価試行MAE / RL目標幅 [%]",
        "図5. 較正に使っていない同じ32姿勢プール内の3件×目標5点、全試行MAE。Few-shotは未知ツールで3〜15回計測する。自ツールDB参照は情報量が異なる。")
    plots.append(fig)
    few_cov_fig, fig = figure("fig-few-cal-coverage", labels, [
        ("共通軸3点", [r["common_probe_coverage"] for r in few]),
        ("採用候補3点", [r["selected_probe_coverage"] for r in few])],
        "較正3点の実測ツール状態幅 / RL目標幅 [%]",
        "図6. 単一較正姿勢で測った3点の応答幅。較正不成立は点を描かず、幅0%と区別する。テスト姿勢での到達保証ではない。", 105)
    plots.append(fig)
    few_cat = table(["カテゴリ", "共通軸FF [%]", "Few-shot [%]", "タスク別PLS [%]", "タスク別PLS＋較正 [%]", "採用較正幅 [%]", "平均プローブ回数"], [
        [c, f2(category_mean(few, c, "baseline")), f2(category_mean(few, c, "fewshot")),
         f2(category_mean(few, c, "local")), f2(category_mean(few, c, "local_cal")),
         f2(category_mean(few, c, "selected_probe_coverage")),
         f2(category_mean(few, c, "probes"))] for c in zcats], name="Few-shotカテゴリ平均")
    few_detail = table(["ツール", "共通軸 [%]", "Few-shot [%]", "PLS-DB [%]", "PLS-DB＋較正 [%]", "プローブ", "較正幅 [%]", "採用候補", "Few-shot保持"], [
        [r["task"], f2(r["baseline"]), f2(r["fewshot"]), f2(r["local"]),
         f2(r["local_cal"]), r["probes"], f2(r["selected_probe_coverage"]),
         r["selected"], f"{r['fewshot_supported']}/15"] for r in few], name="Few-shot全ツール")
    worsened = [r["task"] for r in few if r["fewshot"] > r["baseline"]]
    if worsened != ["scissors_lite_1", "push_lite_3", "spray_lite_3"]:
        raise ValueError("Few-shot worsening list changed; update interpretation")
    sections.append(f"<section id='generalization'><h2>4. 未知ツールへの汎化（旧5目標条件）</h2><h3>4.1 初期姿勢のみのゼロショット選択</h3><p>初期姿勢の2次元PCAはScissorsとSprayの姿勢分散をよく説明するが、姿勢が近いことと操作軸が近いことは同じではない。PCA2-NNはカテゴリ平均で共通軸より高誤差だったが、5変種という小標本で差は確定せず、Sprayでは個別に改善する変種もある。軸だけを移す条件ではScissors・Sprayの平均が共通軸より低く、悪化が逆写像まで一緒に移すことに由来する可能性も残る。NNの参照元は同じ未知ツール内でも評価初期姿勢ごとに変わりうる。</p>{zero_fig}{zero_table}<h3>4.2 実測プローブによるFew-shot適応</h3><p>同じ評価姿勢で共通軸FFからFew-shotへ替えると、旧5目標でのカテゴリ平均誤差はScissors 19.62→10.11%、Spray 37.27→19.41%、Push 18.13→11.36%だった。しかし15ツール中3ツール（Scissors 1、Push 3、Spray 3）では悪化した。Scissorsの改善はScissors 3の大きな変化に強く依存し、これを除く4ツールでは約12.5→11.7%である。3回のプローブで停止したのは4/15ツールで、残り11/15は15回使った。未知ツールの成功姿勢DB200件を使う自ツールPLSに3点較正を加えた参照値は、Scissors 9.36%、Spray 15.71%、Push 11.20%であり、Few-shotと同じ情報量ではない。</p>{few_cat}{few_fig}{few_cov_fig}<details><summary>15ツールの全比較を開く</summary>{few_detail}</details><p class='caution'>較正不成立は図6で欠測として示し、真に測った幅0%と区別する。Spray 1の採用軸は15試行中12試行で逆写像の端点制限があり、MAE改善だけでは全域で操作できたとは言えない。75%停止規則は先行相を見た後に決めたため、後続相の別姿勢評価を独立な未知形状での確認とみなさない。Scissors 4のFew-shot保持は12/15だが、旧図5のMAEには有限な終端状態が得られた保持失敗3試行も含む。ゼロショット図4とFew-shot図5は評価姿勢の分割が異なり、数値を直接引き算しない。</p><p class='source'>出典: <a href='../initial_pca2_nn_20261008/summary.json'>ゼロショット集計</a>、<a href='../few_shot_guarded_20261008/scissors_lite_1.json'>Few-shot原始JSONの例</a>、15件の評価JSONと自ツールPLS対応JSON。</p></section>")

    sections.append("""<section id='discussion'><h2>5. 考察と結論</h2><p>保存済み成功姿勢から求めた1軸の開ループ操作でも、15タスクのツール状態に測定可能な変化は生じた。ただし到達幅は一様ではなく、Push 4のPCAはRL目標幅の約32%にとどまる。Push 1の旧保持MAEはPCA 26.3%、PLS 24.7%と高く、カテゴリ平均への影響が大きい。これらを伏せて15タスクすべての全域操作が成立したとは結論できない。Sprayの旧PCA/PLS保持MAE差は小さく、方式別の評価目標域も異なるため、PCAが優位とは言えない。</p><p>第1軸FBは選定した4タスクの探索的比較で誤差を下げたが、各目標で多数の感度試行を要し、飽和や停止もある。第2軸は一部条件でFFを改善するものの、第1軸を上回る安定した追加役割はまだ確立していない。これは第2軸一般の無効性ではなく、ここで定義した残差軸と較正手順についての結果である。</p><p>未知ツールでは、初期姿勢だけで移す軸と逆写像の組が共通軸より一貫して優れる証拠はない。少数の物理応答を測る適応は有望だが、最大15回の計測費用、較正不成立、端点制限を含めて評価する必要がある。論文の中心は1自由度TDSの操作可能性と限界とし、FB・Few-shotは条件を明示した探索的拡張として扱う。</p><p>次の独立確認には、未使用のツール形状と新しい初期姿勢プールを用い、Few-shotの75%規則を変更せずに試すことが必要である。実機では接触力、転倒、関節速度を別途監視し、シミュレーション上のPalm球終端判定だけを安全判定に使わない。</p></section>""")

    provenance = table(["固定データ", "SHA-256", "監査用途"], [
        [key, value, {"ff_manifest_sha256": "15件の選定レポート、DB・時間条件を検証", "training_manifest_sha256": "15タスクSAC学習キャンペーンを特定", "fb_manifest_sha256": "4タスクFBキャンペーンを特定", "zero_summary_sha256": "ゼロショット集計を特定", "few_validator_sha256": "Few-shot後続相の分割規則を特定。改行をLFに正規化したSHA-256"}[key]]
        for key, value in data["source_hashes"].items()], name="データ来歴")
    audit = ["<section id='audit'><h2>6. 実装整合性・学術的妥当性・研究倫理の監査</h2><p>下表はソース・保存結果から確認した事項であり、「不正が存在しない」ことを証明するものではない。外部独立監査の可否は末尾の監査状態に記す。</p>"]
    audit.append(table(["区分", "証拠", "論文上の対応"], [
        ["確認済み: FFの比較域差", "各report.json → configurations.pca_1d/pls_1d.summary.inverse_domain_fraction, controller_comparison.isotonic", "PLS/PCAの保持MAEは方法ごとに異なる逆写像可能域・目標集合内。図2を完全対応の優劣検定として扱わず、Coverageと分母を併記。"],
        ["確認済み: 手続き間の成功定義差", "paper_one_dof_baseline_20261006.md、report.json の palm_sphere_radius/evaluation_grasp_criterion", "RL/DB/TDSの保持・採用条件を表1で分離。Scissors TDS球0.075mをRL 0.05mと同一視しない。"],
        ["確認済み: FB第2軸の定義", "source_snapshot/improved_tds/synergy/two_axis.py、FB各report.json → axis2_definition", "PLS第2潜在成分と表記しない。PLS第1軸を除去した姿勢残差のPC1と明記。"],
        ["確認済み: FBの無効試行と転倒", "FB各report.json → methods.*.summary.axis2_fb.disabled_feedback / ff.tipped_trials", "感度較正不成立例を分母から消さない。転倒診断とPalm終端有効性を区別し、実機安全性を主張しない。"],
        ["確認済み: Few-shotの事後選択", "validate_guarded_few_shot.py 冒頭、COMMON_COVERAGE_STOP=.75", "75%規則は先行相後に設定。後続相は新規姿勢での検証だが、独立未知ツールでの確認ではない。"],
        ["探索的研究の位置付け", "過去の複数キャンペーンと選定済みpaper15_paper_bundle manifest", "条件変更の履歴を隠して事前登録済みの単一実験と表現しない。選定後の主結果と後続の探索実験を分離。"],
        ["確認済み: 自ツールDB参照", "*_local_pls.json → reference_type, source_trial_file", "未知ツール自身の成功DBを用いるためFew-shotと情報量が異なり、性能上限でもない。較正の有無も分ける。"],
        ["確認済み: 評価目標の不一致", "reach_tds_direct.py の np.linspace(*inverse_range,9)、simulate_initial_pose_conditioning.py の TARGET_FRACTIONS、各Few-shot JSON", "主FFの9点と未知ツールの5点は件数だけでなく角度域も異なる。共通9目標の再評価は別報告で提示し、旧条件間MAEの直接優劣は主張しない。"],
        ["統計的制約", "FF: 8姿勢×9目標、FB: 8姿勢×9目標、Few-shot: 3姿勢×5目標/ツール", "試行を独立サンプルとみなしたp値を示さない。少数ツール・多重比較・未登録閾値のため効果を探索的に記述。"],
        ["再現性の制約", "analysis/ と runs/ はGit追跡外。manifestハッシュは本報告に固定", "レポートだけでは全シミュレーション再実行不可。モデル、DB、MuJoCo、依存バージョンを別途アーカイブする。"],
        ["外部環境ソースの固定", "学習runとFFバンドルのsource_snapshotにはimproved_tdsを保存。tool-manip-gymの環境コードは同じ場所に見当たらない", "Push/SprayのRL成功条件・物理環境の完全再現には外部gymのGitコミットと依存バージョンも固定する。現ワークツリーの環境コードを過去runと無条件に同一視しない。"],
        ["未確認", "独立形状での追試・実機での安全/追従性", "限定ファイルのClaude監査は行ったが、実機性能や一般的な統計的有意性を証明しない。"],
    ], name="監査論点"))
    audit.append("<p>元論文はDDPG+HER、5 Scissors、成功姿勢2000件、12較正点、ランプ写像などを記述する歴史的原案である。今回の15タスク・200件/タスク・PCA/PLS・17点/8姿勢較正・等張回帰の結果を、元論文の2度などの値と混合しない。元論文PDFはユーザー提供資料であり、外部監査先には送信しない。</p>")
    audit.append(provenance)
    audit.append("<p class='caution'><strong>Claude外部監査:</strong> ユーザーの明示承認後、Opus 5.5 / High Effortで指定ファイルのみを読み取り専用で2回確認した。1回目はHTML本文が長い1行だったため全文を読めず、2回目に報告生成コードを点検した。評価目標とMAEの不一致、事後設定の成功基準・75%規則、自ツールDB参照を性能上限と誤称する点、較正不成立を0%と混同する点などを指摘され、この版で表現を修正した。これは不正が存在しないことの証明ではない。監査範囲と対応は <a href='../../docs/claude_external_audit_20261008.md'>監査記録</a>に明記する。元論文PDFは外部送信していない。</p></section>")
    sections.append("".join(audit))
    sections.append("""<section id='style'><h2>図の体裁・再現</h2><p>各図はPlotly.jsの白背景テンプレート（<code>plotly_white</code>相当）を明示し、Times New Roman、図内タイトルなし、黒・青・赤・緑の系列色、内向き主目盛・補助目盛を用いる。Plotlyの余白指定は上下左右2–4pxとし、軸名・目盛の切断だけを自動余白で防ぐ。SVGのベクターフォールバックはオフラインでも表示される。色だけに依存しないよう記号と凡例も併用する。単位とRL目標幅の正規化を軸名に明示した。</p><p>描画仕様の根拠: <a href='https://plotly.com/python/axes/'>Plotly公式・軸と補助目盛</a>、<a href='https://plotly.com/python/templates/'>Plotly公式・plotly_white</a>、<a href='https://plotly.com/python/setting-graph-size/'>Plotly公式・自動余白</a>、<a href='https://plotly.com/javascript/getting-started/'>Plotly公式・版を固定したCDN</a>。選択的報告を避ける基本方針は <a href='https://publicationethics.org/files/International%20standards_authors_for%20website_11_Nov_2011.pdf'>COPE著者向け基準</a>を参照した。</p></section>""")
    sections.append("<footer><p>生成スクリプト: <code>scripts/build_paper_ff_fb_generalization_report.py</code>。数値の機械可読な再集計は同ディレクトリの <a href='summary.json'>summary.json</a> を参照。</p></footer>")

    style = """<style>
*{box-sizing:border-box}body{margin:0;background:#f6f6f4;color:#111;font:16px/1.72 'Times New Roman','Yu Mincho',serif}header,section,nav,footer{max-width:1200px;margin:0 auto}header{padding:60px 30px 35px;background:#fff;border-bottom:3px solid #111}h1{font:700 clamp(29px,3vw,44px)/1.27 'Times New Roman','Yu Mincho',serif;margin:8px 0 22px}h2{font-size:28px;border-bottom:1px solid #111;padding-bottom:8px;margin-top:0}h3{font-size:21px;margin-top:28px}p{margin:12px 0}.kicker{letter-spacing:.13em;font-size:13px;text-transform:uppercase}.subtitle{font-size:19px;max-width:950px}.badge{display:inline-block;border:1px solid #111;padding:6px 10px;font-size:13px;margin-top:15px}nav{padding:10px 25px;background:#fff;position:sticky;top:0;z-index:3;border-bottom:1px solid #ccc;display:flex;gap:18px;flex-wrap:wrap}nav a{color:#111;text-decoration:none;font-size:14px}nav a:hover{text-decoration:underline}section{background:#fff;padding:34px 30px;margin:18px auto}figure{margin:25px 0 35px;padding:12px 0;border-top:1px solid #ddd;border-bottom:1px solid #ddd}.graph{width:100%;min-height:360px}.graph svg{display:block;width:100%;height:auto}figcaption{font-size:14px;line-height:1.5;margin-top:7px}.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:14px}.legend-item{white-space:nowrap}.swatch{display:inline-block;width:12px;height:12px;border:2px solid var(--series);margin-right:5px;vertical-align:middle}.table-wrap{overflow-x:auto;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #bbb;padding:7px 9px;text-align:left;white-space:nowrap}th{border-top:2px solid #111;background:#f5f5f5}tr:nth-child(even){background:#fafafa}.equation{font:19px/1.5 'Times New Roman',serif;text-align:center;position:relative;overflow-x:auto;padding:15px 50px;margin:18px 0;background:#fafafa}.eqno{position:absolute;right:10px;top:19px;font:15px 'Times New Roman',serif}.caution{border-left:4px solid #C33C3C;background:#fff8f8;padding:11px 14px}.source{font-size:14px;color:#444}details{border:1px solid #bbb;padding:9px 15px;margin:20px 0}summary{cursor:pointer;font-weight:bold}code{font:13px/1.5 Consolas,monospace;background:#f3f3f3;padding:2px 4px}a{color:#0057a6}footer{padding:25px 30px 50px;font-size:13px;color:#444}@media(max-width:700px){header,section{padding:20px 14px}nav{position:static}.graph{min-height:230px}}
@media print{body{background:#fff}nav{display:none}section{page-break-inside:avoid;margin:0;padding:18px 8px}figure{page-break-inside:avoid}.graph{min-height:0}details{display:block}details>*{display:block}}
</style>"""
    style += """<style>.procedure-figure,.mae-figure{border:1px solid #d5dbe1;border-radius:9px;padding:20px;background:#fcfdfd}
.flow-node{border:1.5px solid #667b8e;border-radius:8px;padding:13px 17px;background:white;display:flex;flex-direction:column;gap:5px}
.flow-node small{font-size:13px;color:#315979;letter-spacing:.04em}.flow-node strong{font-size:17px;line-height:1.45}.flow-node span{font-size:14px;line-height:1.65}
.flow-source,.flow-test,.flow-output{max-width:690px;margin:0 auto}.flow-source{border-color:#111}.flow-test{border-color:#0072B2;background:#f5faff}.flow-output{border-color:#008B57;background:#f4fcf7}
.flow-branches{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.flow-arrow{text-align:center;color:#445d70;font-size:24px;font-weight:bold;line-height:1.8}
.mae-svg{display:block;width:min(100%,740px);height:auto;margin:0 auto}.mae-grid{stroke:#e0e5e8;stroke-width:1}.mae-axis{stroke:#111;stroke-width:2}.mae-tick{font:15px 'Times New Roman',serif;fill:#111}.mae-label{font:18px 'Times New Roman','Yu Mincho',serif;fill:#111}.mae-legend{font:15px 'Times New Roman','Yu Mincho',serif}
.formula-block{border-left:4px solid #0072B2;background:#f5faff;padding:14px 22px;margin:20px 0;font:18px/1.9 'Times New Roman','Yu Mincho',serif;overflow-x:auto}.formula-block>div{white-space:nowrap}
@media(max-width:780px){.flow-branches{grid-template-columns:1fr}.formula-block{font-size:15px}.mae-figure,.procedure-figure{padding:10px}}
@media print{.flow-node,.procedure-figure,.mae-figure{break-inside:avoid}}</style>"""
    plot_json = json.dumps(plots, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    js = """<script src='https://cdn.plot.ly/plotly-4.1.1.min.js' charset='utf-8'></script>
<script src='https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js' async></script>
<script>
const paperPlots=__PLOTS__;
// Named white-background template, matching the Plotly Python plotly_white look.
const plotly_white={layout:{paper_bgcolor:'#ffffff',plot_bgcolor:'#ffffff',font:{family:'Times New Roman',size:15,color:'#111111'},colorway:['#111111','#0072B2','#C33C3C','#008B57']}};
if(window.Plotly){for(const p of paperPlots){
  const traces=p.series.map((s,j)=>({type:'scatter',mode:'markers',name:s.name,
    x:p.labels.map((_,i)=>i+(j-(p.series.length-1)/2)*0.13),y:s.values,
    marker:{size:9,color:s.color,symbol:s.symbol,line:{color:s.color,width:1.2}},
    hovertemplate:'%{y:.2f}%<extra>'+s.name+'</extra>'}));
  const axis={ticks:'inside',ticklen:6,tickwidth:1,tickcolor:'#111',
    minor:{ticks:'inside',ticklen:3,tickwidth:1,tickcolor:'#111',showgrid:false},
    showline:true,linecolor:'#111',linewidth:1,showgrid:false,zeroline:false,
    automargin:true,tickfont:{family:'Times New Roman',size:13}};
  const layout={template:plotly_white,margin:{l:4,r:3,t:2,b:4},showlegend:false,
    font:{family:'Times New Roman',size:15,color:'#111'},height:360,
    xaxis:{...axis,tickmode:'array',tickvals:p.labels.map((_,i)=>i),ticktext:p.labels,
      tickangle:-40,range:[-0.7,p.labels.length-0.3]},
    yaxis:{...axis,title:{text:p.ylabel,font:{family:'Times New Roman',size:15}},
      range:[0,p.maximum],rangemode:'tozero'}};
  Plotly.newPlot(p.id,traces,layout,{responsive:true,displayModeBar:false});
}}
</script>""".replace("__PLOTS__", plot_json)
    return "<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>ツール操作シナジー：PCA・PLS、FF・FB、未知ツール適応</title>" + style + "</head><body>" + "".join(sections) + js + "</body></html>"


def main() -> None:
    data = load_data()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "index.html").write_text(build_html(data), encoding="utf-8")
    print(OUT / "index.html")
    print(f"FF rows={len(data['ff'])}, FB rows={len(data['fb'])}, Few-shot tools={len(data['few'])}")


if __name__ == "__main__":
    main()
