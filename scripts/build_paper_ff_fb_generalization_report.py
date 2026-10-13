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
SWEEPS = ROOT / "analysis" / "tds_viability_admitted_20260929"
REPRESENTATIVE_DIR = "representative_frames_nonzero"
INITIAL_STATE_DIR = "initial_states"
TRAINING_REWARD_FILE = "training_reward_history.json"
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
            or training_manifest["training"]["algorithm"].lower() != "sac"
            or training_manifest["training"]["learning_rate"] != 5e-5):
        raise ValueError("Selected training campaign differs")
    training = []
    for task in TASKS:
        summary = read(TRAIN / task / "model_training_summary.json")
        if (summary["algorithm"].lower() != "sac"
                or summary["n_envs"] != 4 or summary["batch_size"] != 256
                or summary["buffer_size"] != 1_000_000
                or summary["learning_starts"] != 1_000
                or summary["gradient_steps"] != 2
                or summary["ent_coef"] != .02
                or summary["her_sampled_goals"] != 0
                or summary["target_min_fraction"] != 0.
                or summary["target_max_fraction"] != 1.):
            raise ValueError(f"Training hyperparameters differ: {task}")
        training.append({"task": task, "steps": int(summary["total_timesteps"]),
                         "early_stop": bool(summary["stopped_early"]),
                         "best_validation_success_percent":
                             100 * float(summary["validation"]["best"]["success_rate"])})
    training_reward = read(OUT / TRAINING_REWARD_FILE)
    if (training_reward["protocol"] != "paper15_validation_mean_step_reward_v1"
            or training_reward["manifest_sha256"] != sha(TRAIN / "manifest.json")
            or [item["task"] for item in training_reward["tasks"]] != list(TASKS)):
        raise ValueError("Training reward history does not match frozen campaign")
    for item, entry in zip(training_reward["tasks"], training):
        folder = TRAIN / item["task"]
        if (item["history_sha256"] != sha(folder / "validation_history.json")
                or item["summary_sha256"] != sha(folder / "model_training_summary.json")
                or item["training_steps"] != entry["steps"]
                or len(item["points"]) != item["training_steps"] // 100_000
                or any(point["validation_episodes"] != 352 for point in item["points"])):
            raise ValueError(f"Training reward history differs: {item['task']}")
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
            "training_reward": training_reward,
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


def training_history_svg(items: list[dict], *, success: bool) -> str:
    """Offline, Times-style line chart for five frozen validation histories."""
    width, height = 790, 355
    left, right, top, bottom = 70, 15, 12, 75
    plot_width, plot_height = width - left - right, height - top - bottom
    xmax = max(item["points"][-1]["train_steps"] for item in items) / 1_000_000
    values = [point["success_rate"] * 100 if success else point["mean_reward_per_sim_step"]
              for item in items for point in item["points"]]
    ymin, ymax = ((0, 100) if success else
                  (math.floor(min(values) - .1), math.ceil(max(values) + .1)))
    if ymax <= ymin:
        raise ValueError("invalid training reward chart range")
    ylabel = ("Validation success rate [%]" if success else
              "Mean validation reward [reward/step]")
    parts = [
        f"<svg class='training-chart' role='img' aria-label='{esc(ylabel)}' "
        f"viewBox='0 0 {width} {height}' xmlns='http://www.w3.org/2000/svg'>",
        f"<rect width='{width}' height='{height}' fill='#fff'/>",
        f"<path d='M{left},{top}V{top+plot_height}H{left+plot_width}' "
        "stroke='#111' stroke-width='1.2' fill='none'/>",
    ]
    yticks = ((0, 25, 50, 75, 100) if success else range(ymin, ymax + 1))
    for value in yticks:
        y = top + plot_height * (1 - (value - ymin) / (ymax - ymin))
        parts.append(f"<path d='M{left},{y:.2f}h7' stroke='#111'/>"
                     f"<text x='{left-9}' y='{y+4:.2f}' text-anchor='end' "
                     f"font-family='Times New Roman' font-size='14'>{value}</text>")
    if success:
        y = top + plot_height * .05
        parts.append(f"<path d='M{left},{y:.2f}H{left+plot_width}' stroke='#888' "
                     "stroke-dasharray='5 5' stroke-width='1'/>"
                     f"<text x='{left+plot_width-3}' y='{y-4:.2f}' text-anchor='end' "
                     "font-family='Times New Roman' font-size='12' fill='#555'>95%</text>")
    intervals = (round(xmax * 10) if xmax < 1 else 6)
    for index in range(intervals + 1):
        fraction = index / intervals
        x = left + plot_width * fraction
        value = xmax * fraction
        parts.append(f"<path d='M{x:.2f},{top+plot_height}v-7' stroke='#111'/>"
                     f"<text x='{x:.2f}' y='{top+plot_height+20}' text-anchor='middle' "
                     f"font-family='Times New Roman' font-size='14'>{value:.1f}</text>")
    parts.append(f"<text x='{left+plot_width/2:.2f}' y='{height-40}' text-anchor='middle' "
                 "font-family='Times New Roman' font-size='15'>Training steps [×10⁶]</text>"
                 f"<text transform='translate(17 {top+plot_height/2:.2f}) rotate(-90)' "
                 f"text-anchor='middle' font-family='Times New Roman' font-size='15'>{esc(ylabel)}</text>")
    for index, item in enumerate(items):
        color = COLORS[index]
        points = []
        for point in item["points"]:
            x = left + plot_width * (point["train_steps"] / 1_000_000) / xmax
            value = (100 * point["success_rate"] if success
                     else point["mean_reward_per_sim_step"])
            y = top + plot_height * (1 - (value - ymin) / (ymax - ymin))
            points.append((x, y, point["train_steps"], value))
        path = " ".join(f"{'M' if n == 0 else 'L'}{x:.2f},{y:.2f}"
                        for n, (x, y, _, _) in enumerate(points))
        parts.append(f"<path d='{path}' fill='none' stroke='{color}' stroke-width='2'/>" )
        for x, y, step, value in points:
            parts.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='3.2' fill='white' "
                         f"stroke='{color}' stroke-width='1.8'><title>"
                         f"{esc(item['task'])}: {step:,} steps, {value:.3f}"
                         "</title></circle>")
        legend_x = left + index * 138
        parts.append(f"<path d='M{legend_x},{height-16}h20' stroke='{color}' stroke-width='2'/>"
                     f"<text x='{legend_x+25}' y='{height-11}' "
                     f"font-family='Times New Roman' font-size='13'>{index+1}</text>")
    parts.append("</svg>")
    return "".join(parts)


def training_reward_section(data: dict, protocol: dict) -> str:
    if (protocol["cap_steps"] != 3_000_000
            or protocol["concurrent_learners"] != 4
            or protocol["environments_per_learner"] != 4
            or protocol["validation_every_steps"] != 100_000
            or protocol["checkpoint_every_steps"] != 100_000
            or protocol["early_stop"] != {"success_rate": .95, "consecutive": 2}
            or protocol["success_hold_steps"] != 5):
        raise ValueError("Training conditions differ from report")
    records = {item["task"]: item for item in data["tasks"]}
    conditions = table(
        ["項目", "凍結済み15タスク学習の設定"], [
            ["学習器", "SAC、HERなし。学習率5×10⁻⁵、固定エントロピー係数0.02、gradient steps 2、batch 256、replay buffer 100万、learning starts 1,000。"],
            ["並列実行", "最大4学習プロセス、それぞれ4環境。各タスクは独立モデル・独立乱数種。"],
            ["目標と初期姿勢", "目標は各タスクのRL範囲0～100%から抽出。タスク固有の32初期姿勢プールを使用。"],
            ["Step上限・保存", "各タスク300万Step上限、10万StepごとにValidationとCheckpointを実施。最良成功率モデルを別保存。"],
            ["Early Stop", "10万StepごとのValidationで成功率95%以上を2回連続達成。成功判定は5Step連続。"],
            ["行動変化の制限", "Scissorsは要求行動と適用行動の差を1Step当たり0.05に制限。Push/Sprayは正規化行動の変化を1秒当たり1.25に制限。いずれも要求・適用差にペナルティ。"],
        ], name="学習の共通条件とカテゴリ別の行動制限")
    reward_terms = table(
        ["項", "共通の係数・役割", "タスクカテゴリによる違い"], [
            ["角度・変位の追跡", "粗い誤差と細かい誤差のガウス型報酬。合計最大2、把持ゲート0.25+0.75gを乗じる。", "Scissorsは粗・細各1で粗い誤差幅0.35 rad。Push/Sprayは追跡係数2を半分ずつ割り当て、粗い誤差幅を目標幅の35%と許容値の大きい方にする。"],
            ["進捗", "直前Stepからの絶対誤差の減少を目標幅で割り、−1～1へ制限。係数1、把持ゲートを乗じる。", "対象はScissors角度、Push変位、Sprayトリガー角度。"],
            ["把持", "把持スコアgに係数1。", "Scissorsはヒンジ中心がPalm追従球内にあるか。Push/Sprayは必要な2指接触の充足度。"],
            ["操作補助", "最大係数0.25。", "Scissorsは姿勢維持情報、Pushは親指とプランジャーの接触、Sprayは人差し指とトリガーの近接。Scissorsの追加接触補助係数は0。"],
            ["成功・失敗", "成功中は各Stepに+5。失敗時の基本報酬は−25へ置換。", "ScissorsはPalm球内＋角度誤差30 mrad。Push/Sprayは目標許容域＋2指接触＋落下・転倒なし。Scissorsの接触数・落下・転倒はRL成功の必須条件ではない。"],
            ["制御・制動", "行動変化コスト×−0.1、行動量コスト×−0.002、要求・適用差の二乗平均×−1、目標近傍の過速度を最大−0.5。", "行動制限の実装は上記の通りScissorsとPush/Sprayで異なる。"],
        ], name="学習時に使用した報酬項の対応")
    rows = []
    for task in TASKS:
        item = records[task]
        low, high = item["target_range"]
        goal = (f"{low*1000:g}–{high*1000:g} mm" if task.startswith("push")
                else f"{low:g}–{high:g} rad")
        first, last = item["points"][0], item["points"][-1]
        rows.append([task, goal, item["seed"], f"{item['training_steps']:,}",
                     "あり" if item["stopped_early"] else "なし",
                     f"{first['mean_reward_per_sim_step']:.2f} → "
                     f"{last['mean_reward_per_sim_step']:.2f}",
                     f"{100*item['best_success_rate']:.1f}% "
                     f"({item['best_success_step']//1000:,}k Step)"])
    task_table = table(["タスク", "RL目標範囲", "Seed", "実学習Step", "早期停止",
                        "Validation平均報酬：最初→最後 [reward/Step]",
                        "最良Validation成功率（Step）"], rows,
                       name="15タスクの学習条件と報酬・成功率")
    parts = [
        "<section id='training-detail'><h2>学習条件・報酬設計・学習曲線</h2>",
        "<p>本節は、論文で使用した凍結済みの15タスク学習キャンペーンだけを対象とする。"
        "タスクごとに異なるシードで1モデルを学習し、別の反復Seedによる平均・信頼区間ではない。"
        "学習上限を揃えていてもEarly Stopで実学習Stepは異なる。</p>",
        conditions,
        "<h3>報酬の定義</h3>",
        "<p>追跡・進捗・把持の係数と成功・失敗の尺度をカテゴリ間でできる限り揃え、"
        "操作接点だけをカテゴリ固有の補助項とした。把持スコアgは0～1であり、"
        "追跡と進捗に掛けるゲートは0.25+0.75gである。成功ボーナスは一度だけではなく、"
        "成功条件を満たす各Stepで付く。失敗時は基本報酬を−25に置き換えるため、"
        "下式の通常Stepの単純和を失敗Stepへそのまま適用しない。</p>",
        "<div class='formula-block'>\\[r_t=r_{\\mathrm{track}}+r_{\\mathrm{progress}}"
        "+r_{\\mathrm{grasp}}+r_{\\mathrm{interaction}}"
        "+5\\,\\mathbf{1}_{\\mathrm{success}}-0.1c_{\\Delta u}"
        "-0.002c_u-c_{\\mathrm{gap}}+r_{\\mathrm{brake}}.\\]</div>",
        "<div class='formula-block'>\\[G_t=0.25+0.75g_t,\\quad "
        "r_{\\mathrm{track}}=G_t\\left[\\exp\\!\\left(-\\frac{e_t^2}{2\\sigma_c^2}\\right)"
        "+\\exp\\!\\left(-\\frac{e_t^2}{2\\tau^2}\\right)\\right],\\quad "
        "r_{\\mathrm{progress}}=G_t\\,\\operatorname{clip}\\!\\left("
        "\\frac{e_{t-1}-e_t}{L},-1,1\\right).\\]</div>",
        "<p>ここでeはツール状態の絶対誤差、LはRL目標幅、τはタスクの許容誤差、"
        "σ<sub>c</sub>は粗い誤差幅である。式は通常Stepの共通骨格を示し、"
        "各カテゴリの失敗時置換と補助項の定義は下表で区別する。</p>",
        reward_terms,
        "<h3>タスク別の実学習量と到達成績</h3>", task_table,
        "<h3>10万Stepごとの固定Validation推移</h3>",
        "<p>以下は<strong>学習中ロールアウトの報酬ではない</strong>。各Checkpointの決定論的方策を"
        "同じ32初期姿勢×11目標＝352試行で検証した記録である。各試行の記録値は"
        "『その試行の各Step報酬の平均』なので、各試行Step数で重み付けして全352試行の"
        "平均1Step報酬を再計算した。横軸は学習Stepであり実時間ではない。"
        "報酬グラフはカテゴリ別に縦軸を拡大しているため、カテゴリをまたぐ線の高さで"
        "性能を比較しない。右の成功率は同じValidationの別指標で、95%破線に2回連続で"
        "達したときEarly Stopする。最良モデルは報酬最大ではなく検証成績で保存する。</p>",
        "<div class='formula-block'>\\[\\overline r(s)=\\frac{\\sum_{j=1}^{352}"
        "n_j\\,\\overline r_j(s)}{\\sum_{j=1}^{352}n_j},\\qquad"
        "\\overline r_j(s)=\\frac{1}{n_j}\\sum_{t=1}^{n_j}r_{j,t}(s).\\]</div>",
    ]
    for category, title in (("scissors", "Scissors"), ("push", "Push"), ("spray", "Spray")):
        items = [records[f"{category}_lite_{i}"] for i in range(1, 6)]
        parts.append(f"<h4>{title} 1～5</h4><p>凡例の1～5は{title}の変種番号。"
                     "線の終端はEarly Stopまたは300万Step上限を表す。</p>"
                     "<div class='training-chart-pair'><figure>"
                     + training_history_svg(items, success=False)
                     + "<figcaption>平均1Step報酬（固定Validation）</figcaption></figure><figure>"
                     + training_history_svg(items, success=True)
                     + "<figcaption>成功率（同じ固定Validation）</figcaption></figure></div>")
    parts.append("<p class='caution'>報酬値が上昇しても、未知初期姿勢・未知ツールやTDSの"
                 "1自由度制御が成功したことにはならない。各Validationは学習時と同じ32姿勢プールを"
                 "使用し、比較は各タスク1 Seedである。Scissorsの追跡成功条件とPush/Sprayの"
                 "接触成功条件は完全には同一でない。</p>"
                 f"<p class='source'>出典: <a href='{TRAINING_REWARD_FILE}'>"
                 "Validation報酬時系列と元ログSHA-256</a>、"
                 "<a href='../../runs/paper15_unified_3m_slew_20260927/manifest.json'>"
                 "学習キャンペーンmanifest</a>、各タスクのmodel_training_summary.json、"
                 "凍結した環境・検証ソース。報酬系列の再集計コードは"
                 "scripts/export_training_reward_history.py。</p></section>")
    return "".join(parts)


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


def protocol_comparison(ff: list[dict], zero: dict, matched: dict) -> str:
    """Explain the non-interchangeable inverse maps and evaluation populations."""
    if matched["protocol"] != "common_full_rl_nine_goal_v1":
        raise ValueError("matched evaluation protocol changed")
    protocols = table(
        ["実験・方式", "軸と未知ツール情報", "逆写像を作る観測", "物理較正／評価初期姿勢", "評価目標と誤差集計"],
        [
            ["2章：PCA／PLS 1自由度FF", "自ツール成功最終姿勢200件。PCAまたはPLS第1軸", "17スコア×8姿勢の物理応答から単調区間を選び、等張回帰", "較正8姿勢／別の評価8姿勢", "方式別逆写像可能域内の9目標×8姿勢。保持MAE"],
            ["3章：FF・第1／第2軸FB", "自ツール軸と2章の物理較正済みFFを固定", "各目標で8較正姿勢×3オフセット×2軸の感度計測を追加", "感度8姿勢／別の評価8姿勢", "共通逆写像域の9目標×8姿勢。保持・停止・飽和を併記"],
            ["4.1節：共通軸／PCA2-NN", "未知ツールを除く4ツール計800姿勢。NNは既知ツールの軸を選択", "既知DBの得点―ツール状態の直線回帰。未知ツールで物理較正なし", "較正0姿勢／評価3姿勢", "RL範囲の10, 30, 50, 70, 90%×3姿勢。全試行MAE"],
            ["4.1節：自ツールDB・線形逆写像参照", "未知ツール自身の成功最終姿勢200件とPLS第1軸", "同じ200件の得点―ツール状態の直線回帰。物理較正なし", "較正0姿勢／評価3姿勢", "上と同じ5目標×3姿勢。全試行MAE。ゼロショットではない"],
            ["4.2節：Few-shot", "既知4ツールの共通軸と候補軸。未知ツールDBは不使用", "未知ツールの1姿勢で候補ごと3点の物理応答を直線近似。計3～15プローブ", "較正1姿勢／別の評価3姿勢", "RL範囲の5目標×3姿勢。全有限終端MAE"],
            ["4.2節：自ツールDB＋3点較正参照", "未知ツール自身の成功最終姿勢200件とPLS第1軸", "未知ツール1姿勢の3点物理応答を追加", "較正1姿勢／別の評価3姿勢", "Few-shotと同じ5目標×3姿勢。情報量はFew-shotより多い"],
            ["共通9目標：固定FFと未知ツール方式", "RL・成功DB・既存FF較正を固定。新しい学習はしない", "FFは既存17×8較正を再使用。未知ツール方式の3点応答は別途実測してフィット", "FFの評価8姿勢で全方式を揃える。プローブは別の較正姿勢", "RL全範囲の0, 12.5, …, 100%×8姿勢。完走・有限終端MAE"],
        ], name="章ごとの軸、較正、評価母集団の違い")
    old_and_matched = table(
        ["カテゴリ", "旧2章：物理較正PLS・保持MAE [%]", "旧4.1節：自ツールDB直線・全試行MAE [%]", "共通9目標：物理較正PLS・完走有限MAE [%]", "共通9目標：自ツールDB直線・完走有限MAE [%]"],
        [[category.capitalize(),
          f2(mean(row["normalized_held_mae_percent"] for row in ff
                  if row["category"] == category and row["method"] == "pls")),
          f2(100 * zero["categories"][category]["by_arm"]["task_specific_reference"]["mae_fraction"]),
          f2(mean(matched["ff"][f"{category}_lite_{i}"]["pls"]["complete_finite_mae_percent"]
                  for i in range(1, 6))),
          f2(mean(matched["unknown"][f"{category}_lite_{i}"]["arms"]["local_pls_oracle"]["complete_finite_mae_percent"]
                  for i in range(1, 6)))] for category in CATEGORIES],
        name="旧条件と共通9目標条件のカテゴリ別誤差。旧条件の二列は直接比較しない")
    parts = [
        "<section id='protocol-comparison'><h2>実験間で何が同じで、何が違うか</h2>",
        "<p>同じ「PLS第1軸」でも、成功姿勢DBから得た関節方向、目標角度から指令を求める逆写像、実際にハンドを動かして測った較正応答は別物である。特に2章と4.1節の自ツール参照は<strong>軸と中心姿勢が同一</strong>である一方、逆写像の構築法が違う。したがって4.1節の誤差を「自ツールのPLS軸自体が悪い」と解釈しない。</p>",
        protocols,
        "<h3>実験別の実行順序</h3>",
        "<p><strong>2章・1自由度FF：</strong>タスクごとの成功姿勢200件からPCAとPLSを別々に作る。各軸で拡張得点範囲の17点を、較正用8初期姿勢から1点ずつ独立リセットして5秒軌道・2秒保持で試す。採用応答の単調区間を等張回帰し、逆写像可能域の9目標を別の8姿勢から評価する。各方式の目標域が異なるため、旧図のPCA対PLSも同一目標での厳密なペア比較ではない。</p>",
        "<p><strong>3章・FB：</strong>Scissors 3・4、Push 3、Spray 3のみを対象とし、2章のFF指令を固定する。各目標・各軸で8較正姿勢×3オフセットの物理応答から感度を見積もる。第1軸または第2軸のどちらか一方を使って、5秒のFF移動後、実測ツール状態の偏差を2秒間補正する。第2軸はPCAのPC2、PLSでは第1軸を除いた姿勢残差のPC1であり、PLS第2潜在成分ではない。感度計測だけで各目標48試行／方法に相当し、1自由度FFと同じ較正予算ではない。</p>",
        "<p><strong>4.1節・ゼロショット：</strong>5ツールのうち1ツールを除外し、既知4ツールの成功姿勢計800件から共通軸を作る。別案では各既知ツールの軸を、初期姿勢PCAの2次元空間で最近傍となるツールから選ぶ。これらは未知ツールの成功姿勢・物理応答を軸選択に使わない。自ツールDB参照だけは除外ツール自身の200件から軸と直線逆写像を作る<strong>情報量の異なる参照</strong>である。3初期姿勢×5目標を測る。</p>",
        "<p><strong>4.2節・Few-shot：</strong>まず共通軸を未知ツールの1較正姿勢で3回試す。候補スコアは既知DB回帰の応答15%、50%、85%相当から決めるが、採用判断には実測値を用いる。3点が有限・終端支持などの条件を満たし、概ね単調かつ十分な応答幅を持つ場合だけ直線応答を作る。共通軸の実測CoverageがRL目標幅の75%以上なら3回で止め、未満なら既知4ツールの軸も各3回試して最大15回とする。採用候補は較正時の実測Coverageを第一基準として決め、候補が一つも成立しなければ共通軸のDB直線指令へ戻す。較正と異なる3姿勢×5目標で評価し、フォールバック試行も分母から除かない。75%閾値は先行相後に設定した探索的規則であり、新しい未知形状での独立検証ではない。</p>",
        "<p><strong>共通9目標の再評価：</strong>RL、成功姿勢DBと主FFの等張回帰を凍結し、各ツールの主FF評価用8姿勢を全方式で共有する。Few-shotの3点応答だけは、評価姿勢と重ならない較正姿勢で新たに測る。9目標はRL全範囲の両端を含むため、逆写像外への指令は端点に制限して記録する。ここでは全方式の評価集合を揃えたが、未知ツールDBの使用可否と物理較正回数は揃えていない。</p>",
        "<h3>成功姿勢DBで作る軸と、二つの逆写像</h3>",
        "<p>各ツールの成功最終姿勢を関節角ベクトル qᵢ、同時に記録したツール状態を yᵢ とする。平均姿勢を q̄、長さ1の操作軸を w とすると、姿勢DB内の得点は次式である。</p>",
        equation(r"\bar{\boldsymbol q}=\frac{1}{200}\sum_{i=1}^{200}\boldsymbol q_i,\qquad z_i=(\boldsymbol q_i-\bar{\boldsymbol q})^{\mathsf T}\boldsymbol w,\qquad \|\boldsymbol w\|_2=1.", 8),
        "<p>4.1節の自ツールDB参照は、<strong>RLが到達した成功最終姿勢200件</strong>について、得点と正規化ツール状態の直線関係を最小二乗で求める。ツール状態は実装ではDB中の最大目標値で割る。今回の15タスクでは目標下限が0であり、これはRL目標幅による正規化と一致する。この回帰には、再構成姿勢へ5秒移動した物理応答は入らない。</p>",
        equation(r"(\hat a,\hat b)=\underset{a,b}{\arg\min}\sum_{i=1}^{200}(\tau_i-a-bz_i)^2,\qquad \tau_i=\frac{y_i}{y_{\max}^{\mathrm{RL}}},\qquad z_{\mathrm{DB}}(\tau^*)=\operatorname{clip}_{[1.5z_{\min},\,1.5z_{\max}]}\!\left(\frac{\tau^*-\hat a}{\hat b}\right).", 9),
        equation(r"\boldsymbol q_{\mathrm{DB}}(\tau^*)=\operatorname{clip}_{\mathrm{joint}}\!\left(\bar{\boldsymbol q}+z_{\mathrm{DB}}(\tau^*)\boldsymbol w\right).", 10),
        "<p>一方、2章の主FFは、得点範囲を1.5倍に広げて17点を試し、各点を8較正初期姿勢から5秒で再構成姿勢へ移動・2秒保持して<strong>実際のツール応答</strong>を測る。採用できた応答から概ね単調な区間を選び、等張回帰の単調な平均応答を逆に解く。ρ は無次元制御入力、s は関節角スケールであり、DB得点 z と同じ単位・数値ではない。</p>",
        equation(r"\boldsymbol q_{\mathrm{FF}}(\rho)=\operatorname{clip}_{\mathrm{joint}}\!\left(\bar{\boldsymbol q}+s\rho\boldsymbol w\right),\quad \bar y_j=\operatorname{mean}_{p\in\mathcal C_j}y_{\mathrm{sim}}(p,\rho_j),\quad \hat f=\operatorname{IsotonicFit}\{(\rho_j,\bar y_j)\}_{j\in\mathcal M}.", 11),
        equation(r"\rho^*(y^*)=\hat f^{-1}\!\left(\operatorname{clip}_{\mathcal D_{\mathrm{inv}}}(y^*)\right),\qquad z_{\mathrm{FF}}(y^*)=s\rho^*(y^*),\qquad \boldsymbol q_{\mathrm{goal}}=\boldsymbol q_{\mathrm{FF}}\!\left(\rho^*(y^*)\right).", 12),
        "<p>ここで Cⱼ はそのスコア点で応答を採用できた較正初期姿勢、M は選択した単調区間、D<sub>inv</sub> は逆写像可能なツール状態域を表す。<strong>軸 w と平均 q̄ が等しくても、z<sub>DB</sub>(y*) と z<sub>FF</sub>(y*) が一致するとは限らない。</strong>DB回帰はRLポリシーが到達した姿勢間の相関であり、再構成姿勢を開ループで指令したときの因果的な物理応答そのものではない。</p>",
        "<h3>評価目標・分母・較正コストの違い</h3>",
        "<p>旧2章の9目標は方式ごとの逆写像可能域に置き、保持が成立した試行だけを主MAEの分母に入れた。旧4.1節はRL目標範囲の内側5点を使い、自ツールDB参照を含む各方式の15試行について誤差を集計した。よって旧2章と旧4.1節の数値は、同じ軸でも<strong>比較可能な対照実験ではない</strong>。</p>",
        equation(r"\operatorname{MAE}_{A}[\%]=\frac{100}{|A|\,(y_{\max}^{\mathrm{RL}}-y_{\min}^{\mathrm{RL}})}\sum_{i\in A}|y_i^{\mathrm{end}}-y_i^*|,\quad A_{\mathrm{held}}=\{i:\mathrm{complete}_i\land\mathrm{finite}_i\land\mathrm{held}_i\},\quad A_{\mathrm{finite}}=\{i:\mathrm{complete}_i\land\mathrm{finite}_i\}.", 13),
        "<p>共通9目標の再評価では、各ツールのRL目標範囲を12.5%刻みで9点に固定し、各方式を<strong>同じ8評価初期姿勢・同じリセット種</strong>で実行した。目標が逆写像可能域の外なら端点へ制限し、その試行も完走・有限であればMAEに含む。保持失敗も有限終端なら含め、数値異常や未完走は別に件数を示す。5秒移動＋2秒保持は共通である。ただし軸に与えた情報と較正試行数まで同一になるわけではない。</p>",
        old_and_matched,
        "<p>表の左側二列は旧条件の差を確認するための参考値で、横差から方式の優劣を結論しない。右側二列は同じ9目標・8姿勢・終端MAEで比較できる。<strong>物理較正済みPLSの誤差が、DB直線逆写像参照より小さい</strong>という傾向は共通条件でも残る。これは同じ軸に対し、物理応答で逆写像を作る効果と整合するが、較正予算が大きく違うため、軸選択アルゴリズム単体の優劣は示さない。</p>",
        "<p>例えば共通9目標のScissors 2・初期姿勢ID 16・目標0.5 radでは、物理較正済みPLSの終端値が0.4998 rad、自ツールDB直線参照が0.1863 radだった。両方とも保持成立、逆写像の端点制限なしである。この差は把持失敗だけでは説明できない。一方で前者には関節角のクリップがあり、単一例から差の原因を物理較正だけに帰することもできない。</p>",
        "<p class='source'>実装照合: <code>reach_tds_direct.py</code>（17点・物理較正・等張回帰）、<code>simulate_initial_pose_conditioning.py</code> と <code>simulate_initial_pca_nn.py</code>（DB直線回帰・旧5目標）、<code>few_shot_axis_adaptation.py</code>（3点応答）、<code>validate_guarded_few_shot.py</code>（75%停止規則）、<code>replay_matched_nine_ff.py</code> と <code>replay_matched_nine_generalization.py</code>（共通9目標）。集計元は <a href='../matched_nine_goals_clipped_20261008/summary.json'>共通9目標summary.json</a>。</p>",
        "</section>",
    ]
    return "".join(parts)


def matched_nine_section(matched: dict, plots: list[dict]) -> str:
    """Put the completed equal-goal replay in the paper, not just a link."""
    if matched["protocol"] != "common_full_rl_nine_goal_v1":
        raise ValueError("unexpected matched-nine protocol")
    arms = ("common_zero", "pca2_nn_zero", "common_three_probe",
            "selected_three_probe", "local_pls_oracle", "local_pls_three_probe_oracle")
    arm_names = ("共通軸・計測なし", "初期姿勢NN・計測なし", "共通軸＋3点計測",
                 "候補選択＋3～15点計測", "自ツールDB・直線", "自ツールDB＋3点計測")
    labels = [label(task) for task in TASKS]
    ff_plot, spec = figure("matched-ff", labels, [
        (method.upper(), [matched["ff"][task][method]["complete_finite_mae_percent"]
                          for task in TASKS]) for method in ("pca", "pls")],
        "完走有限終端MAE / RL目標幅 [%]",
        "共通9目標・図1。同じ8評価姿勢×9目標のPCA／PLS比較。較正写像は各方式の保存済み値。")
    plots.append(spec)
    unknown_plot, spec = figure("matched-unknown", labels, [
        (name, [matched["unknown"][task]["arms"][arm]["complete_finite_mae_percent"]
                for task in TASKS]) for name, arm in zip(
                    ("共通軸", "初期姿勢NN", "選択軸＋計測", "自ツールDB・直線"),
                    ("common_zero", "pca2_nn_zero", "selected_three_probe", "local_pls_oracle"))],
        "完走有限終端MAE / RL目標幅 [%]",
        "共通9目標・図2。未知ツール転用と自ツールDB参照。後者は未知ツール自身の成功姿勢200件を使うため、情報条件が異なる。")
    plots.append(spec)
    fb_labels = [f"{label(task)}-{method.upper()}" for task in FB_TASKS
                 for method in ("pca", "pls")]
    fb_plot, spec = figure("matched-fb", fb_labels, [
        (name, [matched["fb"][task][method][condition]["complete_finite_mae_percent"]
                for task in FB_TASKS for method in ("pca", "pls")])
        for name, condition in (("FF", "ff"), ("第1軸FB", "axis1_fb"),
                                ("第2軸FB", "axis2_fb"))],
        "完走有限終端MAE / RL目標幅 [%]",
        "共通9目標・図3。4タスクの同一目標・姿勢におけるFFとFB。FBには各目標の感度較正が追加される。")
    plots.append(spec)
    coverage_plot, spec = figure("matched-coverage", labels, [
        ("PLS FF到達幅", [matched["ff"][task]["pls"]["held_reached_span_percent"]
                         for task in TASKS]),
        ("未知・選択軸到達幅", [matched["unknown"][task]["arms"]
            ["selected_three_probe"]["held_reached_span_percent"] for task in TASKS])],
        "保持された実測到達幅 / RL目標幅 [%]",
        "共通9目標・図4。到達幅は8姿勢をまとめた保持終点の最大−最小であり、全目標での成功保証ではない。", 105)
    plots.append(spec)
    category_table = table(
        ["カテゴリ", "PCA FF MAE [%]", "PLS FF MAE [%]", "未知・共通軸 MAE [%]",
         "未知・選択＋計測 MAE [%]", "自ツールDB直線 MAE [%]"],
        [[category.capitalize(), *[f2(mean(
            (matched["ff"][f"{category}_lite_{i}"][method]
             if stage == "ff" else matched["unknown"][f"{category}_lite_{i}"]
                 ["arms"][method])["complete_finite_mae_percent"]
            for i in range(1, 6))) for stage, method in (
                ("ff", "pca"), ("ff", "pls"), ("unknown", "common_zero"),
                ("unknown", "selected_three_probe"), ("unknown", "local_pls_oracle"))]]
         for category in CATEGORIES], name="共通9目標のカテゴリ内5ツール等重み平均")
    ff_table = table(
        ["ツール", "PCA MAE [%]", "PLS MAE [%]", "PCA保持/72", "PLS保持/72",
         "PCA端点制限/72", "PLS端点制限/72", "PCA関節制限/72", "PLS関節制限/72"],
        [[task,
          f2(matched["ff"][task]["pca"]["complete_finite_mae_percent"]),
          f2(matched["ff"][task]["pls"]["complete_finite_mae_percent"]),
          *[f'{matched["ff"][task][method]["held"]}/72' for method in ("pca", "pls")],
          *[f'{matched["ff"][task][method]["inverse_clipped"]}/72' for method in ("pca", "pls")],
          *[f'{matched["ff"][task][method]["joint_clipped"]}/72' for method in ("pca", "pls")]]
         for task in TASKS], name="共通9目標FFの全15ツール結果")
    unknown_table = table(
        ["ツール", *[f"{name} MAE [%]" for name in arm_names], "計測回数",
         "選択＋計測の保持/72", "選択＋計測の端点制限/72"],
        [[task, *[f2(matched["unknown"][task]["arms"][arm]
                     ["complete_finite_mae_percent"]) for arm in arms],
          matched["unknown"][task]["probes_attempted"],
          f'{matched["unknown"][task]["arms"]["selected_three_probe"]["held"]}/72',
          f'{matched["unknown"][task]["arms"]["selected_three_probe"]["inverse_clipped"]}/72']
         for task in TASKS], name="共通9目標の未知ツール全6方式と全15ツール")
    fb_table = table(
        ["ツール・軸抽出", "FF MAE [%]", "第1軸FB MAE [%]", "第2軸FB MAE [%]",
         "FF保持/72", "第1軸保持/72", "第2軸保持/72", "第2軸FB成立/72"],
        [[f"{task} / {method.upper()}",
          *[f2(matched["fb"][task][method][condition]["complete_finite_mae_percent"])
            for condition in ("ff", "axis1_fb", "axis2_fb")],
          *[f'{matched["fb"][task][method][condition]["held"]}/72'
            for condition in ("ff", "axis1_fb", "axis2_fb")],
          f'{matched["fb"][task][method]["axis2_fb"]["feedback_enabled"]}/72']
         for task in FB_TASKS for method in ("pca", "pls")],
        name="共通9目標のFB全4タスク結果")
    paired_table = table(
        ["ツール", "PLS−PCA FF差 [%]", "対応完走/72",
         "未知・選択＋計測−共通軸差 [%]", "対応完走/72"],
        [[task,
          f2(matched["paired"]["ff"][task]["second_minus_first_mae_percent"]),
          matched["paired"]["ff"][task]["paired_complete_finite"],
          f2(matched["paired"]["unknown"][task]["second_minus_first_mae_percent"]),
          matched["paired"]["unknown"][task]["paired_complete_finite"]]
         for task in TASKS], name="同一姿勢・同一目標で対応する試行のMAE差")
    category_readout = "、".join(
        f"{category.capitalize()}：PLS FF {mean(matched['ff'][f'{category}_lite_{i}']['pls']['complete_finite_mae_percent'] for i in range(1, 6)):.2f}%、"
        f"未知・共通軸 {mean(matched['unknown'][f'{category}_lite_{i}']['arms']['common_zero']['complete_finite_mae_percent'] for i in range(1, 6)):.2f}%、"
        f"未知・選択＋計測 {mean(matched['unknown'][f'{category}_lite_{i}']['arms']['selected_three_probe']['complete_finite_mae_percent'] for i in range(1, 6)):.2f}%"
        for category in CATEGORIES)
    missing_ff = sum(72 - matched["ff"][task][method]["complete_finite"]
                     for task in TASKS for method in ("pca", "pls"))
    missing_unknown = sum(72 - matched["unknown"][task]["arms"][arm]["complete_finite"]
                          for task in TASKS for arm in arms)
    missing_fb = sum(72 - matched["fb"][task][method][condition]["complete_finite"]
                     for task in FB_TASKS for method in ("pca", "pls")
                     for condition in ("ff", "axis1_fb", "axis2_fb"))
    return (
        "<section id='matched-nine'><h2>共通9目標・同条件再評価の結果</h2>"
        "<p>旧実験の数値とは別に、保存済みモデルと成功姿勢DBを使い、RL目標範囲の0～100%を12.5%刻みの9点、同じ8評価初期姿勢で再評価した。各方式は72試行／ツール。5秒の線形軌道と2秒の保持、リセット種、完走・有限終端MAEの定義を揃えた。逆写像外の目標を端点に制限しても試行は残し、保持失敗も有限終端ならMAEに含める。</p>"
        "<p class='caution'>揃えたのは評価目標・姿勢・時間・誤差分母であり、較正予算ではない。FFは保存済み17点×8姿勢の物理較正、未知ツール方式は0または3～15プローブ、FBは各目標の感度較正を追加する。自ツールDB参照は未知ツール自身の成功姿勢200件を使い、ゼロショット手法の性能上限ではない。75%停止則は探索後に固定した。</p>"
        "<h3>カテゴリ別の主要数値</h3>" + category_table
        + f"<p>カテゴリ内5ツールの等重み平均は、{category_readout}であった。"
          "Few-shotは平均誤差を下げたが、同じ較正予算のゼロショット比較ではなく、"
          "Sprayの未知・選択＋計測でも約25%の誤差が残る。"
          f"完走・有限終端の欠測はFFで{missing_ff}件、未知ツール方式で{missing_unknown}件、"
          f"FBで{missing_fb}件。FB内のFFはFF表と重複するため欠測数を合算しない。</p>"
        + "<h3>PCA／PLSの1自由度FF</h3>" + ff_plot
        + "<details><summary>全15ツールの誤差・保持・制限件数</summary>" + ff_table + "</details>"
        + "<h3>未知ツールへの転用・Few-shot</h3>" + unknown_plot
        + "<p>共通軸＋3点と選択軸＋計測を区別し、未知ツールDBを使う二条件は情報量の異なる参照として扱う。端点制限率と保持率をMAEと併記する。</p>"
        + "<details><summary>全15ツール・6方式の数値</summary>" + unknown_table + "</details>"
        + "<h3>第1軸／第2軸FB</h3>" + fb_plot
        + "<details><summary>全4タスク×PCA／PLSの数値</summary>" + fb_table + "</details>"
        + "<h3>到達幅と対応試行の差</h3>" + coverage_plot
        + "<p>下表の差は同一ツール・姿勢・目標で両方式が完走した試行のみで計算し、負値が後者の低誤差を意味する。試行を独立なサンプルとみなした有意差検定は行わない。</p>"
        + "<details><summary>全15ツールの対応試行差</summary>" + paired_table + "</details>"
        + "<p class='source'>数値の出典: <a href='../matched_nine_goals_clipped_20261008/summary.json'>共通9目標の集計JSON</a>。"
        "追加の欠測・転倒診断・較正幅・制限層別は<a href='../matched_nine_goals_clipped_20261008/index.html'>詳細報告</a>を参照。"
        "本節はリンクだけでなく主要図表をこの論文HTML内に再掲した。</p></section>")


def initial_state_section() -> str:
    """Show the same reset with and without non-hand visuals for all tools."""
    manifest_path = OUT / INITIAL_STATE_DIR / "manifest.json"
    manifest = read(manifest_path)
    records = {item["task"]: item for item in manifest["tasks"]}
    if (manifest["protocol"] != "frozen_ff_test_initial_state_palm_facing_v1"
            or not manifest["no_simulation_step"] or set(records) != set(TASKS)
            or len(manifest["tasks"]) != len(TASKS)):
        raise ValueError("Initial-state image manifest is incomplete")
    parts = [
        "<section id='initial-states'><h2>15タスクの初期状態とハンド初期姿勢</h2>",
        "<p>各タスクの32種類の初期姿勢から、1自由度FF実験で評価に使用した"
        "<strong>最初の姿勢IDを1例</strong>として示す。画像は目標をRL範囲の50%に設定して"
        "リセットした直後（Step 0）であり、学習済み方策やTDSの操作前である。"
        "同じリセット状態を左はハンド＋対象物、右はハンドのみで描画した。右図では"
        "対象物・床などの<strong>描画だけ</strong>を隠しており、物理状態や関節姿勢は変更していない。"
        "カメラはPalm面に正対する方向から撮影した。1例で32姿勢すべてのばらつきを"
        "示すものではない。姿勢IDはタスクごとのプール内番号であり、別タスクの同じIDは"
        "同一の姿勢を意味しない。</p>",
    ]
    for category, title in (("scissors", "Scissors"), ("push", "Push"), ("spray", "Spray")):
        parts.append(f"<details><summary>{title} 1～5：初期状態・初期ハンド姿勢を開く</summary>"
                     "<div class='initial-task-grid'>")
        for variant in range(1, 6):
            task = f"{category}_lite_{variant}"
            item = records[task]
            source = BASE / task / "report.json"
            report = read(source)
            if (item["pose_id"] != int(report["test_poses"][0])
                    or item["pool_size"] != 32
                    or item["report_sha256"] != sha(source)
                    or item["camera_dot_palm_normal"] > -.999
                    or len(item["initial_actuated_q"]) != 13):
                raise ValueError(f"Initial-state source or camera differs: {task}")
            parts.append(f"<article class='initial-task'><h3>{title} {variant} "
                         f"<small>初期姿勢ID {item['pose_id']}</small></h3>"
                         "<div class='initial-pair'>")
            for key, caption in (("state_image", "初期状態：ハンドと対象物"),
                                 ("hand_image", "初期ハンド姿勢：対象物・床を非表示")):
                relative = str(item[key]).replace("\\", "/")
                path = (OUT / INITIAL_STATE_DIR / relative).resolve()
                if (not path.is_relative_to((OUT / INITIAL_STATE_DIR).resolve())
                        or not path.is_file()):
                    raise FileNotFoundError(path)
                href = f"{INITIAL_STATE_DIR}/{relative}"
                alt = f"{title} {variant}、初期姿勢ID {item['pose_id']}、{caption}"
                parts.append(f"<figure><a href='{esc(href)}' target='_blank' rel='noopener'>"
                             f"<img src='{esc(href)}' loading='lazy' width='512' height='384' "
                             f"alt='{esc(alt)}'></a><figcaption>{esc(caption)}</figcaption>"
                             "</figure>")
            parts.append("</div></article>")
        parts.append("</div></details>")
    parts.append(f"<p class='source'>姿勢ID、リセット種、カメラ方向、画像の来歴："
                 f"<a href='{INITIAL_STATE_DIR}/manifest.json'>初期状態画像のmanifest</a>。"
                 "これらの描画は既存の学習・誤差評価結果を変更しない。</p></section>")
    return "".join(parts)


def synergy_sweep_section() -> str:
    """Show the archived, source-verified PCA/PLS hand-only score sweeps."""
    scores = (-1.0, -0.5, 0.0, 0.5, 1.0)
    methods = (
        ("PCA", "pca_principal_sweeps_hand_only", "pca_2d",
         (("pc1", "第1主成分：ツール操作軸"),
          ("pc2", "第2主成分：姿勢分散の次の方向"))),
        ("PLS", "pls_axis_sweeps_hand_only", "pls_2d",
         (("pls1", "第1軸：ツール操作軸"),
          ("residual_pca", "第2軸相当：PLS第1軸を除いた姿勢残差のPC1"))),
    )
    parts = [
        "<section id='synergy-sweeps'><h2>PCA・PLSシナジースイープ：ハンド姿勢の変化</h2>",
        "<p>全15タスクについて、PCAとPLSの操作軸、および各方式の第2軸相当を"
        "−1、−0.5、0、0.5、1の5点で示す。得点は関節限界で正規化した無次元の"
        "<strong>物理スコア</strong>であり、PCAの生の主成分得点ではない。表示対象の軸だけを動かし、"
        "もう一方の軸の入力は0に固定した。PCA・PLSの第1軸は、主実験の1自由度FFで使う軸と同一である。</p>",
        "<p>各画像は元のタスク環境で5秒移動・2秒保持した後の実測ハンド姿勢を、"
        "ハンド単体モデルに写して表示したもの。Palm法線と撮影視線の内積は−1である。"
        "ツールは画像に写っていないため、元試行の終端ツール状態を画像下に併記する。"
        "画像には採用不可の試行も含まれ、画像だけから把持成功や制御性能を判定しない。"
        "スコア0は共通の平均姿勢なので、同一タスクのPCA・PLS画像が一致するのは正常である。"
        "PLSの第2軸相当は<strong>PLS第2潜在成分ではなく</strong>、第1軸を除いた姿勢残差の第1主成分である。"
        "各タスクの画像は初期状態で閉じている。</p>",
    ]
    for task in TASKS:
        category, _, number = task.partition("_lite_")
        task_name = f"{category.capitalize()} {number}"
        source_report = BASE / task / "report.json"
        report = read(source_report)
        parts.append(f"<details><summary>{esc(task_name)}（{esc(label(task))}）：PCA・PLS、各2軸×5点</summary>")
        for method, directory, fit_name, axes in methods:
            manifest_path = SWEEPS / directory / task / "manifest.json"
            manifest = read(manifest_path)
            if (manifest["task"] != task or manifest["source_fit"] != fit_name
                    or manifest["source_report_sha256"] != sha(source_report)
                    or manifest["scores"] != list(scores)
                    or manifest["trajectory_seconds"] != 5.0
                    or manifest["hold_seconds"] != 2.0
                    or manifest["camera"]["optical_forward_dot_palm_normal"] > -0.999):
                raise ValueError(f"sweep source or camera differs: {task}/{method}")
            base_axis = report["configurations"][f"{method.lower()}_1d"]["fit"]["operation_axis"]
            sweep_axis = report["configurations"][fit_name]["fit"]["operation_axis"]
            if base_axis != sweep_axis:
                raise ValueError(f"sweep operation axis differs from paper FF: {task}/{method}")
            images = {(item["axis"], item["score"]): item for item in manifest["images"]}
            expected = {(axis, score) for axis, _ in axes for score in scores}
            if len(manifest["images"]) != 10 or set(images) != expected:
                raise ValueError(f"sweep images incomplete or duplicated: {task}/{method}")
            for axis, description in axes:
                parts.append(f"<h3>{esc(method)} ― {esc(description)}</h3><div class='sweep-grid'>")
                for score in scores:
                    item = images[(axis, score)]
                    image = item["image"].replace("\\", "/")
                    path = (SWEEPS / image).resolve()
                    if not path.is_relative_to(SWEEPS.resolve()) or not path.is_file():
                        raise FileNotFoundError(path)
                    src = f"../{SWEEPS.name}/{image}"
                    factor, unit = (1000.0, "mm") if category == "push" else (1.0, "rad")
                    tool = item["achieved_tool"] * factor
                    caption = f"スコア {score:+.1f} ／ 元試行の終端ツール状態 {tool:.3f} {unit}"
                    alt = f"{task_name}、{method}、{description}、{caption}。Palm正面から見たハンド単体姿勢"
                    parts.append(f"<figure class='sweep-frame'><a href='{esc(src)}' target='_blank' rel='noopener'>"
                                 f"<img src='{esc(src)}' loading='lazy' width='400' height='400' alt='{esc(alt)}'></a>"
                                 f"<figcaption>{esc(caption)}</figcaption></figure>")
                parts.append("</div>")
            manifest_href = f"../{SWEEPS.name}/{directory}/{task}/manifest.json"
            parts.append(f"<p class='source'>{esc(method)}画像の出典・条件："
                         f"<a href='{esc(manifest_href)}'>マニフェスト</a>。</p>")
        parts.append("</details>")
    parts.append("</section>")
    return "".join(parts)


def representative_section() -> str:
    manifest_path = OUT / REPRESENTATIVE_DIR / "manifest.json"
    manifest = read(manifest_path)
    groups = (tuple(f"ff_{method}" for method in ("pca", "pls"))
              + tuple(f"unknown_{arm}" for arm in (
                  "common_zero", "pca2_nn_zero", "common_three_probe",
                  "selected_three_probe", "local_pls_oracle", "local_pls_three_probe_oracle"))
              + tuple(f"fb_{method}_{axis}" for method in ("pca", "pls")
                      for axis in ("axis1_fb", "axis2_fb")))
    labels = {
        "ff_pca": "PCA・1自由度FF", "ff_pls": "PLS・1自由度FF",
        "unknown_common_zero": "未知ツール：共通軸・計測なし",
        "unknown_pca2_nn_zero": "未知ツール：初期姿勢NN・計測なし",
        "unknown_common_three_probe": "未知ツール：共通軸＋3点計測",
        "unknown_selected_three_probe": "未知ツール：選択軸＋3～15点計測",
        "unknown_local_pls_oracle": "自ツールDB・直線逆写像参照",
        "unknown_local_pls_three_probe_oracle": "自ツールDB＋3点計測参照",
        "fb_pca_axis1_fb": "PCA・第1軸FB", "fb_pca_axis2_fb": "PCA・第2軸FB",
        "fb_pls_axis1_fb": "PLS・第1軸FB", "fb_pls_axis2_fb": "PLS・第2軸FB",
    }
    records = {(case["group"], case["rank"]): case for case in manifest["cases"]}
    if (manifest["protocol"] != "matched_nine_ranked_representative_frames_nonzero_v1"
             or set(records) != {(group, rank) for group in groups
                                    for rank in ("best", "median", "worst")}):
        raise ValueError("representative screenshot manifest is incomplete")
    parts = [
        "<section id='representatives'><h2>目標0を除いた代表試行の軌道画像：最良・中央値・最悪</h2>",
        "<p>各方式について、対象となる全ツールの<strong>目標値0を除いた完走・有限終端</strong>試行を1試行ごとの正規化絶対誤差で昇順に並べ、最小値・中央順位・最大値を機械的に選んだ。目標0では動かさずとも小さな誤差になりうるため、代表画像の選定からのみ除外した。数値結果のMAE集計対象は変更していない。中央値は試行数が偶数のとき下側中央とする。成功例だけを選んだ図ではなく、保持失敗も候補に残す。画像は保存済みの指令・姿勢・目標・リセット種で再生し、終端値と保持判定が元データと一致することを検証した。</p>",
        "<p>Palm面のローカル−y法線側から撮影し、MuJoCoレンダラが配置した実際のカメラ位置から求めた視線とPalm法線の内積を各画像で−1と検証した。開始時と25 Stepごとの画像を示す。各メニューは初期状態で閉じている。角度はrad、Pushの変位はmmで表示する。単一試行の極値はツール平均や代表的な性能を意味しない。</p>",
    ]
    for group in groups:
        parts.append(f"<h3>{esc(labels[group])}</h3>")
        for rank, rank_label in (("best", "最良"), ("median", "中央値"),
                                 ("worst", "最悪")):
            case = records[(group, rank)]
            if abs(case["target"]) <= 1e-12:
                raise ValueError(f"zero target selected as representative: {group}/{rank}")
            if not case["frames"] or case["frames"][0]["step"] != 0:
                raise ValueError(f"missing initial frame: {group}/{rank}")
            steps = [frame["step"] for frame in case["frames"]]
            if any(step % 25 for step in steps) or steps != sorted(set(steps)):
                raise ValueError(f"frame interval differs: {group}/{rank}")
            if any(frame.get("camera_dot_palm_normal", 1.) > -.999
                   for frame in case["frames"]):
                raise ValueError(f"Palm-facing camera not verified: {group}/{rank}")
            scale, unit = (1000., "mm") if case["task"].startswith("push") else (1., "rad")
            summary = (f"{rank_label}：{case['task']}／初期姿勢ID {case['pose']}／"
                       f"目標 {case['target']*scale:.3f} {unit}／終端 {case['achieved']*scale:.3f} {unit}／"
                       f"誤差 {case['error_percent']:.2f}%／保持 {'成立' if case['held'] else '不成立'}")
            parts.append(f"<details><summary>{esc(summary)}</summary>"
                         f"<p>順位 {case['rank_index']}/{case['pool_count']}。"
                         f"保存済み結果との差 {case['replay_abs_difference']*scale:.5f} {unit}。"
                         f"画像 {len(case['frames'])} 枚。</p><div class='frame-grid'>")
            for frame in case["frames"]:
                path = OUT / REPRESENTATIVE_DIR / frame["image"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                src = REPRESENTATIVE_DIR + "/" + frame["image"].replace("\\", "/")
                state_note = (f"Palm球内 {'はい' if frame['on_palm'] else 'いいえ'}"
                              if frame["on_palm"] is not None else
                              f"落下 {'あり' if frame['dropped'] else 'なし'}・"
                              f"転倒 {'あり' if frame['tipped'] else 'なし'}")
                caption = (f"Step {frame['step']}（{frame['time_seconds']:.2f} s）、"
                           f"ツール状態 {frame['tool']*scale:.3f} {unit}、{state_note}")
                parts.append(f"<figure><img src='{esc(src)}' loading='lazy' width='512' height='384' "
                             f"alt='{esc(labels[group] + ' ' + rank_label + ' ' + caption)}'>"
                             f"<figcaption>{esc(caption)}</figcaption></figure>")
            parts.append("</div></details>")
    parts.append("<p class='source'>選定条件・再生一致・各Stepの値: "
                 f"<a href='{REPRESENTATIVE_DIR}/manifest.json'>画像マニフェスト</a>。"
                 "画像再生は評価結果の再計算・モデル更新ではない。</p></section>")
    return "".join(parts)


def experiment_map_section() -> str:
    """Link each reported experiment to its live source and frozen result."""
    def artifact(path: str) -> str:
        source = ROOT / path
        if not source.is_file():
            raise FileNotFoundError(f"Paper experiment map points to missing file: {source}")
        href = "../" + path.removeprefix("analysis/") if path.startswith("analysis/") else "../../" + path
        return f"<a href='{esc(href)}'><code>{esc(path)}</code></a>"

    rows = [
        (
            "15タスクのSAC学習",
            "Scissors・Push・Spray各5タスクを統一キャンペーンで学習し、検証値でベストポリシーを選ぶ。成功最終姿勢DBの再収集は次行のTDS処理で行う。",
            ["scripts/run_paper15_unified.py", "src/improved_tds/experiments/train_sb3.py"],
            "runs/paper15_unified_3m_slew_20260927/",
            ["runs/paper15_unified_3m_slew_20260927/manifest.json",
             "analysis/paper15_paper_bundle_20260928/manifest.json"],
            "学習ステップ・検証成功率は「学習条件」に掲載。成功姿勢DB件数は学習中の姿勢数と混同しない。",
        ),
        (
            "15タスクのPCA/PLS・1自由度TDS（主結果）",
            "成功姿勢DBから軸を求め、較正17点、未知目標9点×初期姿勢8件を評価する。ScissorsはPalm球半径1.5倍のTDS採用条件を用いる。",
            ["scripts/run_paper15_tds_corrected.py", "scripts/run_scissors_palm_tds_campaign.py",
             "scripts/run_reach_tds_direct.py",
             "scripts/integrate_scissors_palm_radius_1p5.py", "scripts/build_paper15_paper_bundle.py",
             "scripts/summarize_paper_one_dof.py"],
            "analysis/paper15_no_contact_floor_on_20260928/、analysis/scissors_tds_palm_radius_1p5_20260928/、analysis/paper15_paper_bundle_20260928/",
            ["analysis/paper15_no_contact_floor_on_20260928/manifest.json",
             "analysis/scissors_tds_palm_radius_1p5_20260928/manifest.json",
             "analysis/paper15_paper_bundle_20260928/scissors_lite_3/report.json"],
            "Coverage・保持MAE・較正採用数の15タスク表は「FF結果」。選定元と置換履歴はbundleのmanifestを参照。",
        ),
        (
            "共通9目標でのFF再評価",
            "主結果と未知ツール実験で異なった目標条件を揃え、選定済み軸を共通目標で再実行する。",
            ["scripts/run_matched_nine_campaign.py", "scripts/replay_matched_nine_ff.py"],
            "analysis/matched_nine_goals_clipped_20261008/ff/",
            ["analysis/matched_nine_goals_clipped_20261008/ff/scissors_lite_1.json",
             "analysis/matched_nine_goals_clipped_20261008/summary.json"],
            "共通条件の保持MAE・終端MAE・端点制限・到達幅は「共通9目標結果」。主FFの元レポートを上書きしない。",
        ),
        (
            "第1・第2軸FB（選定4タスク）",
            "Scissors 3/4、Push 3、Spray 3でFFと第1軸FB・第2軸FBを比較し、共通9目標でも照合する。",
            ["scripts/run_second_axis_feedback.py", "scripts/run_matched_nine_campaign.py"],
            "analysis/second_axis_feedback_effect_20261006/、analysis/matched_nine_goals_clipped_20261008/fb/",
            ["analysis/second_axis_feedback_effect_20261006/scissors_lite_3/report.json",
             "analysis/matched_nine_goals_clipped_20261008/fb/scissors_lite_3/report.json"],
            "条件付きMAE、有限終端MAE、保持率、転倒・補正停止は「FB結果」「共通9目標結果」。15タスク全体のFB結果ではない。",
        ),
        (
            "未知ツール・初期姿勢PCA2＋NN",
            "カテゴリ内の1ツールを除外し、既知4ツールの初期姿勢と操作軸から未知初期姿勢の近傍軸を選ぶ。",
            ["scripts/evaluate_initial_pca_nn.py", "scripts/simulate_initial_pca_nn.py"],
            "analysis/initial_pca2_nn_20261008/",
            ["analysis/initial_pca2_nn_20261008/scissors_lite_1_physical.json",
             "analysis/initial_pca2_nn_20261008/index.html"],
            "元の5目標評価は探索結果として「汎化」に表示。共通9目標の直接比較とは区別する。",
        ),
        (
            "未知ツール・3点較正とFew-shot選択",
            "共通軸と候補軸を未知ツールで試し、較正応答と停止規則に基づき制御軸を選択する。",
            ["scripts/few_shot_axis_adaptation.py", "scripts/validate_guarded_few_shot.py",
             "scripts/evaluate_guarded_task_pls.py"],
            "analysis/few_shot_axis_adaptation_20261008/、analysis/few_shot_guarded_20261008/",
            ["analysis/few_shot_axis_adaptation_20261008/summary.json",
             "analysis/few_shot_guarded_20261008/scissors_lite_1.json"],
            "プローブ回数・採用軸・保持MAEは「汎化」。選択規則は事後設計であり独立検証ではない。",
        ),
        (
            "未知ツール・共通9目標再評価",
            "ゼロショット・Few-shot・自ツールDB参照を同じ9目標と初期姿勢群で再実行する。",
            ["scripts/run_matched_nine_campaign.py", "scripts/replay_matched_nine_generalization.py"],
            "analysis/matched_nine_goals_clipped_20261008/generalization/",
            ["analysis/matched_nine_goals_clipped_20261008/generalization/scissors_lite_1.json",
             "analysis/matched_nine_goals_clipped_20261008/summary.json"],
            "共通9目標での保持MAE・全有限終端MAE・Coverageは「共通9目標結果」。旧5目標の数値と混合しない。",
        ),
        (
            "図・動画フレーム・統合報告",
            "初期状態、シナジースイープ、代表3試行の25ステップ間隔画像、学習報酬履歴を描画し、本HTMLを構築する。",
            ["scripts/render_task_initial_states.py", "scripts/render_matched_representatives.py",
             "scripts/export_training_reward_history.py", "scripts/build_paper_ff_fb_generalization_report.py"],
            "analysis/paper_ff_fb_generalization_20261008/、analysis/tds_viability_admitted_20260929/",
            ["analysis/paper_ff_fb_generalization_20261008/summary.json",
             "analysis/paper_ff_fb_generalization_20261008/representative_frames_nonzero/manifest.json"],
            "図は保存済み実験結果の可視化であり、新たな学習・性能評価ではない。",
        ),
    ]
    body = []
    for name, procedure, scripts, folders, files, result in rows:
        script_links = "<br>".join(artifact(path) for path in scripts)
        result_links = "<br>".join(artifact(path) for path in files)
        body.append("<tr>" + "".join(
            f"<td>{value}</td>" for value in
            (esc(name), esc(procedure), script_links, f"<code>{esc(folders)}</code>",
             result_links, esc(result))) + "</tr>")
    headers = ("実験・処理", "実施内容", "実行・集計スクリプト", "結果ディレクトリ", "代表結果ファイル", "本HTMLでの結果・注意")
    return ("<section id='experiment-map'><h2>実験・スクリプト・結果ファイルの対応</h2>"
            "<p>下表のパスはこのリポジトリを基準とする。リンクした代表結果ファイルだけでなく、"
            "各ディレクトリ内の同名規則の15タスク（FBは4タスク）のJSONを用いる。"
            "学習・評価結果は保存済みスナップショットとmanifestに固定されており、現行コードの単純再実行と同一であるとは限らない。</p>"
            "<div class='table-wrap'><table aria-label='論文実験とコード・結果の対応'><thead><tr>"
            + "".join(f"<th scope='col'>{esc(h)}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>"
            "<p class='source'>実機操作の試作コード " + artifact("scripts/real_experiment/operate_tds_jointtrajectory.py")
            + " は将来実験用であり、この表のシミュレーション結果には使っていない。"
            "退避した探索的コード・結果の一覧はGit管理外の <code>Obsoleted/move_manifest.json</code> にのみ保存する。</p></section>")


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
    sections[-1] = sections[-1].replace("<a href='#visual-guide'>手順・誤差の図解</a>",
        "<a href='#visual-guide'>手順・誤差の図解</a><a href='#protocol-comparison'>実験間の条件差</a>")
    sections[-1] = sections[-1].replace("<a href='#protocol-comparison'>実験間の条件差</a>",
        "<a href='#protocol-comparison'>実験間の条件差</a><a href='#matched-nine'>共通9目標結果</a><a href='#initial-states'>初期状態・姿勢</a><a href='#synergy-sweeps'>シナジースイープ</a><a href='#representatives'>代表画像</a>")
    sections[-1] = sections[-1].replace("<a href='#training'>学習条件</a>",
        "<a href='#training'>学習条件</a><a href='#training-detail'>報酬・学習曲線</a>")
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
                 "新条件の主要図表は本HTMLの<a href='#matched-nine'>共通9目標節</a>に統合した。旧条件の数値とは混ぜない。"))
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
        "<a href='#matched-nine'>このHTML内の共通9目標節</a>に示す。</p>"
        "<h3>目標と実測の『差』がMAEになる</h3>"
        "<p>下図は共通9目標の保存済み実測値を用いた計算例である。横軸の目標値と縦軸の実測値が一致すれば"
        "黒破線上に点が載る。点から黒破線までの縦方向の距離が、その試行の絶対誤差である。"
        "Pushでは角度[rad]の代わりに変位[m]を使い、分母は各ツールのRL目標幅に置き換える。</p>"
        + mae_illustration(matched["mae_illustration"])
        + "<p>各方式の主MAEは、完走して有限の終端値が得られた試行の正規化誤差を平均する。"
        "保持に失敗しても終端が有限なら含め、途中終了・数値異常は誤差0とせず欠測件数として別に示す。"
        "保持試行だけのMAE、保持率、ツール状態の到達幅は別指標である。"
        "この実測例の9点平均は1姿勢の説明値であり、全8姿勢の方式別MAEと同一視しない。</p></section>")
    sections.append(protocol_comparison(ff, zero, matched))
    sections.append(matched_nine_section(matched, plots))
    sections.append(initial_state_section())
    sections.append(synergy_sweep_section())
    sections.append(representative_section())
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
    sections.append(training_reward_section(data["training_reward"], data["training_protocol"]))

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
    sections[1] = sections[1].replace("</nav>", "<a href='#experiment-map'>実験・ファイル対応</a></nav>")
    sections.insert(3, experiment_map_section())
    sections = [section.replace(
        "共通9目標の新規再評価は別の報告に記す。",
        "共通9目標の再評価は本HTMLに旧条件とは分けて示す。")
        .replace("共通9目標の再評価は別報告で提示し、",
                 "共通9目標の再評価は本HTMLの別節で提示し、")
        for section in sections]

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
#protocol-comparison th,#protocol-comparison td{white-space:normal;min-width:125px;vertical-align:top;line-height:1.55}
#protocol-comparison th:first-child,#protocol-comparison td:first-child{min-width:190px}
#experiment-map th,#experiment-map td{white-space:normal;min-width:150px;vertical-align:top;line-height:1.5;overflow-wrap:anywhere}
#experiment-map td:first-child{min-width:170px}
.frame-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;margin:12px 0}
.frame-grid figure{margin:0;padding:4px;border:1px solid #ddd}.frame-grid img{display:block;width:100%;height:auto}
#representatives details{margin:8px 0 18px}
#initial-states details{margin:10px 0 18px}
.training-chart-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:10px 0 24px}
.training-chart-pair figure{margin:0;min-width:0;border:1px solid #d7dde2;padding:6px}
.training-chart{display:block;width:100%;height:auto}
.training-chart-pair figcaption{text-align:center;font-size:13px;line-height:1.5}
.initial-task-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px;margin:14px 0}
.initial-task{border:1px solid #d7dde2;padding:10px;min-width:0;background:#fff}
.initial-task h3{margin:0 0 8px;font-size:18px}.initial-task h3 small{font-size:13px;font-weight:normal;color:#445d70}
.initial-pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
.initial-pair figure{margin:0;min-width:0}.initial-pair img{display:block;width:100%;height:auto}
.initial-pair figcaption{font-size:13px;line-height:1.45;margin-top:4px}
.sweep-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px;margin:12px 0 24px}
.sweep-frame{margin:0;padding:5px;border:1px solid #ddd;min-width:0}.sweep-frame img{display:block;width:100%;height:auto;background:#f6f8fa}
.sweep-frame figcaption{font-size:12px;line-height:1.45;margin-top:4px}
#synergy-sweeps details{margin:10px 0}#synergy-sweeps h3{font-size:18px;margin:22px 0 8px}
@media(max-width:780px){.flow-branches{grid-template-columns:1fr}.formula-block{font-size:15px}.mae-figure,.procedure-figure{padding:10px}}
@media(max-width:900px){.sweep-grid{grid-template-columns:repeat(3,minmax(0,1fr))}}
@media(max-width:900px){.training-chart-pair{grid-template-columns:1fr}}
@media(max-width:600px){.initial-task-grid,.initial-pair{grid-template-columns:1fr}}
@media(max-width:540px){.sweep-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
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
