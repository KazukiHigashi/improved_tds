"""Build a self-contained Japanese HTML report for initial-pose transfer tests."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pose_conditioning import CATEGORIES


ARMS = ("global_no_offset", "initial_projection", "initial_pls",
        "initial_nearest_axis", "task_specific_reference")
LABELS = {
    "global_no_offset": "共通操作軸・補正なし",
    "initial_projection": "初期姿勢の操作軸射影→第2軸",
    "initial_pls": "初期13関節PLS→第2軸",
    "initial_nearest_axis": "初期姿勢最近傍→第1軸選択",
    "task_specific_reference": "対象タスク専用PLS（参照）",
}
COLORS = dict(zip(ARMS, ("#555555", "#d55e00", "#0072b2", "#cc79a7", "#009e73")))


def _mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def summarize(root):
    offline = json.loads((root / "offline_report.json").read_text(encoding="utf-8"))
    summary = dict(protocol="initial_pose_conditioning_physical_v1",
                   offline_protocol=offline["protocol"],
                   source=offline["source"], new_training=False,
                   tool_holdout="fit on other four variants only for shared arms",
                   reset_poses_per_tool=3, targets_fraction=[.1,.3,.5,.7,.9],
                   trajectory_seconds=5, hold_seconds=2,
                   scissors_palm_radius_multiplier=1.5,
                   push_spray_support="no drop/tip; no contact-count gate; Spray floor enabled",
                   caveats=["three authored reset poses per held-out tool",
                            "success-pose DB linear inverse, not simulator calibration",
                            "local reference uses the held-out tool's successes",
                            "second axis is a fixed direction; initial posture selects only its coordinate"],
                   categories={})
    for category in CATEGORIES:
        rows = []
        for variant in range(1, 6):
            stem = f"{category}_lite_{variant}"
            base = json.loads((root / f"{stem}_physical.json").read_text(encoding="utf-8"))
            nearest = json.loads((root / f"{stem}_nearest.json").read_text(encoding="utf-8"))
            rows.extend(base + nearest)
        expected = 5 * len(ARMS) * 3 * 5
        if len(rows) != expected:
            raise ValueError(f"{category}: expected {expected} physical trial rows, got {len(rows)}")
        by_arm = {}
        by_tool = {}
        for arm in ARMS:
            group = [r for r in rows if r["arm"] == arm]
            by_arm[arm] = dict(mae_fraction=_mean(group, "normalized_abs_error"),
                               held_fraction=_mean(group, "supported"),
                               joint_clip_fraction=_mean(group, "joint_clipped"),
                               inverse_clip_fraction=_mean(group, "inverse_clipped"),
                               correction_clip_fraction=_mean(group, "correction_clipped"))
        for variant in range(1, 6):
            task = f"{category}_lite_{variant}"
            by_tool[str(variant)] = {
                arm:dict(mae_fraction=_mean([r for r in rows if r["task"] == task
                                              and r["arm"] == arm], "normalized_abs_error"),
                         held_fraction=_mean([r for r in rows if r["task"] == task
                                                and r["arm"] == arm], "supported"))
                for arm in ARMS}
        base = by_arm["global_no_offset"]["mae_fraction"]
        improvement = {arm:base - by_arm[arm]["mae_fraction"] for arm in ARMS}
        tool_deltas = np.array([
            by_tool[str(i)]["initial_pls"]["mae_fraction"]
            - by_tool[str(i)]["global_no_offset"]["mae_fraction"]
            for i in range(1, 6)])
        rng = np.random.default_rng(421 + len(category))
        resampled = tool_deltas[rng.integers(0, 5, size=(20000, 5))].mean(axis=1)
        bootstrap_interval = np.quantile(resampled, [.025, .975]).tolist()
        baseline = {(r["task"], r["pose"], r["target_fraction"]):r
                    for r in rows if r["arm"] == "global_no_offset"}
        selected = [r for r in rows if r["arm"] == "initial_pls"]
        target_slope = {task:float(np.polyfit(
            [r["target_fraction"] for r in selected if r["task"] == task],
            [r["target_angle"] for r in selected if r["task"] == task], 1)[0])
            for task in {r["task"] for r in selected}}
        physical_axis2_shift = float(np.mean([
            abs(r["achieved_angle"] - baseline[(r["task"], r["pose"],
                  r["target_fraction"])]["achieved_angle"]) / target_slope[r["task"]]
            for r in selected]))
        improved_tool_count = {
            arm:sum(by_tool[str(i)][arm]["mae_fraction"] <
                    by_tool[str(i)]["global_no_offset"]["mae_fraction"]
                    for i in range(1,6)) for arm in ARMS}
        summary["categories"][category] = dict(
            offline=offline["categories"][category], physical=rows,
            by_arm=by_arm, by_tool=by_tool,
            improvement_percentage_points=improvement,
            improved_tool_count=improved_tool_count,
            initial_pls_minus_no_offset_tool_bootstrap95=bootstrap_interval,
            initial_pls_mean_physical_output_shift_fraction=physical_axis2_shift)
        print(category, "MAE", {a:round(by_arm[a]["mae_fraction"], 3) for a in ARMS},
              "tools improved", improved_tool_count, flush=True)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    (root / "index.html").write_text(render(summary), encoding="utf-8")
    return summary


def _table(headers, rows):
    h = "".join(f"<th>{html.escape(str(cell))}</th>" for cell in headers)
    b = "".join("<tr>" + "".join(f"<td>{html.escape(str(cell))}</td>" for cell in row)
                + "</tr>" for row in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def _mae_svg(category, item):
    width, height = 660, 310
    xmax = 565
    y_max = 60.
    x = lambda task: 85 + 120 * (task - 1)
    y = lambda error: 235 - 175 * (100 * error / y_max)
    out = [f"<svg role='img' aria-label='{category} のタスク別角度誤差' "
           f"viewBox='0 0 {width} {height}'>",
           "<line x1='85' y1='60' x2='85' y2='235' stroke='#222'/>",
           f"<line x1='85' y1='235' x2='{xmax}' y2='235' stroke='#222'/>",
           "<text x='8' y='25' font-size='13'>MAE [% of target range]</text>",
           "<text x='270' y='290' font-size='13'>Tool variant</text>"]
    for tick in (0, 20, 40, 60):
        yy = y(tick / 100)
        out.extend((f"<line x1='80' y1='{yy:.1f}' x2='{xmax}' y2='{yy:.1f}' stroke='#ddd'/>",
                    f"<text x='42' y='{yy+4:.1f}' font-size='12'>{tick}</text>"))
    for variant in range(1, 6):
        out.append(f"<text x='{x(variant)-3}' y='255' font-size='12'>{variant}</text>")
    for arm in ARMS:
        values = [item["by_tool"][str(i)][arm]["mae_fraction"] for i in range(1, 6)]
        color = COLORS[arm]
        points = " ".join(f"{x(i)},{y(value):.1f}" for i, value in enumerate(values,1))
        out.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        out.extend(f"<circle cx='{x(i)}' cy='{y(value):.1f}' r='3.7' fill='{color}'/>"
                   for i, value in enumerate(values,1))
    return "".join(out) + "</svg>"


def _target_svg(rows, category, variant):
    task = f"{category}_lite_{variant}"
    selected = [r for r in rows if r["task"] == task]
    x = lambda fraction: 70 + 250 * fraction
    y = lambda fraction: 270 - 240 * fraction
    targets = sorted({r["target_fraction"] for r in selected})
    # All currently used tasks have a zero lower target bound. Recover bounds
    # from commanded target/fraction pairs instead of assuming it implicitly.
    target_rows = [r for r in selected if r["arm"] == "global_no_offset"]
    slope, intercept = np.polyfit([r["target_fraction"] for r in target_rows],
                                  [r["target_angle"] for r in target_rows], 1)
    out = [f"<svg role='img' aria-label='{task} の目標対実角度' viewBox='0 0 410 350'>",
           "<line x1='70' y1='30' x2='70' y2='270' stroke='#222'/>",
           "<line x1='70' y1='270' x2='320' y2='270' stroke='#222'/>",
           "<line x1='70' y1='270' x2='320' y2='30' stroke='#aaa' stroke-dasharray='4 3'/>",
           "<text x='125' y='318' font-size='12'>目標角度 [% of target range]</text>",
           "<text x='5' y='20' font-size='12'>実角度 [%]</text>"]
    for tick in (0, .25, .5, .75, 1):
        out.append(f"<text x='{x(tick)-7:.1f}' y='288' font-size='10'>{tick*100:.0f}</text>")
        out.append(f"<text x='36' y='{y(tick)+3:.1f}' font-size='10'>{tick*100:.0f}</text>")
    for arm in ARMS:
        color = COLORS[arm]
        points = []
        for fraction in targets:
            group = [r for r in selected if r["arm"] == arm
                     and np.isclose(r["target_fraction"], fraction)]
            achieved = float(np.mean([r["achieved_angle"] for r in group]))
            scaled = float(np.clip((achieved - intercept) / slope, -.1, 1.1))
            points.append((x(fraction), y(scaled)))
        path = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in points)
        out.append(f"<polyline points='{path}' fill='none' stroke='{color}' stroke-width='2'/>")
        out.extend(f"<circle cx='{xx:.1f}' cy='{yy:.1f}' r='3' fill='{color}'/>"
                   for xx, yy in points)
    return "".join(out) + "</svg>"


def render(report):
    start = ["<!doctype html><html lang='ja'><meta charset='utf-8'>",
             "<title>初期姿勢に基づくタスク横断TDS</title>",
             "<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;color:#16212b}h1,h2{line-height:1.3}table{border-collapse:collapse;width:100%;margin:1em 0}th,td{border:1px solid #c8d0d8;padding:.4em;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf3f8}svg{max-width:100%;height:auto;border:1px solid #e4e9ee;background:#fff}details{margin:1em 0;padding:.7em;border:1px solid #c8d0d8;border-radius:.3em}code{background:#f1f4f6;padding:.1em .25em}small{color:#536171}</style>",
             "<h1>初期姿勢に基づくタスク横断TDSの検証</h1>",
             "<p>Scissors、Spray、Pushそれぞれ5ツール×200件の成功姿勢DBを使用。"
             "強化学習はやり直していません。各ツール固有の最良ポリシーで得た既存DBを再利用し、"
             "未知ツールを1つずつ除いて、残り4ツールから共通操作軸・姿勢補正を学習しました。</p>",
             "<h2>仮説と妥当性</h2>",
             "<p>第1軸 <code>v1</code> は成功姿勢のうち、各ツール内での角度変化と共変するPLS方向です。"
             "第2軸を『初期姿勢をv1に射影した得点の分散方向』とだけ定めることはできません。"
             "射影得点は1値であり、新しい関節方向を持たないからです。そこで、"
             "成功姿勢からv1方向を除いた残差 <code>r = q成功 − μ − [(q成功 − μ)·v1]v1</code> を"
             "初期姿勢から予測する問題と定義しました。各初期姿勢について角度別成功姿勢の残差を"
             "平均し、同じ初期姿勢の重複を数えません。</p>",
             "<p>比較A（原案の具体化）：<code>s0=(q初期−μ初期)·v1</code> と残差のPLSから第2軸"
             " <code>v2</code> を得て、<code>g=a+b s0</code> とする。"
             "比較B（最小限の改良）：初期13関節と残差との交差共分散の最大モードから、"
             "初期側の1軸 <code>w</code> と成功姿勢側の第2軸 <code>v2</code> を得て、"
             "<code>g=a+b(q初期−μ初期)·w</code> とする。"
             "どちらも実行時は固定した <code>g</code> と操作スコア <code>ρ</code> のみで、"
             "目標姿勢 <code>q*=μ+ρv1+gv2</code> を指定します。"
             "第2軸方向そのものを未知初期姿勢ごとに再学習する方式ではありません。</p>",
             "<h2>オフライン：成功姿勢の残差を初期姿勢から予測できるか</h2>",
             "<p>値は第2軸補正なしに対する残差姿勢二乗誤差の改善率。負値は悪化です。"
             "『既知ツール・未使用初期姿勢』と『未知ツール』は別々に評価しました。</p>"]
    offline_rows = []
    for category in CATEGORIES:
        item = report["categories"][category]["offline"]
        s = item["summary"]
        offline_rows.append([category.title(),
            f"{100*s['held_out_initial_poses']['op_projection']['improvement_vs_zero']:.1f}",
            f"{100*s['held_out_initial_poses']['full_initial']['improvement_vs_zero']:.1f}",
            f"{100*s['leave_one_tool_out']['op_projection']['improvement_vs_zero']:.1f}",
            f"{100*s['leave_one_tool_out']['full_initial']['improvement_vs_zero']:.1f}",
            f"{item['median_initial_nn_distance_ratio']:.1f}×"])
    start.append(_table(["カテゴリ", "既知・射影値 [%]", "既知・初期13関節 [%]",
                         "未知・射影値 [%]", "未知・初期13関節 [%]",
                         "未知初期姿勢の最近傍距離比"], offline_rows))
    for category in CATEGORIES:
        item = report["categories"][category]["offline"]
        rows = [[f"{category.title()} {i}",
                 f"{100*fold['methods']['op_projection']['improvement_vs_zero']:+.1f}",
                 f"{100*fold['methods']['full_initial']['improvement_vs_zero']:+.1f}",
                 f"{fold['initial_nn_distance_to_training_ratio']:.1f}×",
                 f"{fold['operation_axis_cosine_to_heldout']:.2f}"]
                for i, fold in enumerate(item["leave_one_tool_out"], start=1)]
        start.append(f"<details><summary>{category.title()}：未知ツール別のオフライン結果</summary>"
                     + _table(["除外ツール", "射影値の残差改善 [%]", "初期13関節の残差改善 [%]",
                               "初期姿勢距離比", "操作軸余弦"], rows) + "</details>")
    start.append("<p>Sprayの未知初期姿勢は、学習側の初期姿勢同士の通常の最近傍距離に比べ"
                 "約7.5倍離れています。これは初期姿勢の条件付き写像にとって外挿です。"
                 "ただし距離だけで物理制御の成功を予測できるとは限りません。</p>")
    start.append("<h2>MuJoCo：未知ツールへの直接軌道制御</h2><p>各ツール3初期姿勢×5目標角度。"
                 "目標は範囲の10、30、50、70、90%。成功姿勢DBからの線形逆写像で操作スコアを決め、"
                 "初期姿勢から再構成姿勢まで5秒で移動し2秒保持します。ScissorsのPalm球は"
                 "既存TDS評価と同じ1.5倍半径、Push/Sprayは転倒・落下を拒否し接触数を要求しません。"
                 "Sprayの床接触は有効です。</p>")
    for category in CATEGORIES:
        item = report["categories"][category]
        start.append(f"<h3>{category.title()}</h3>")
        summary_rows = [[LABELS[a], f"{100*v['mae_fraction']:.1f}",
                         f"{100*v['held_fraction']:.1f}",
                         f"{100*v['joint_clip_fraction']:.1f}",
                         f"{item['improved_tool_count'][a]}/5"]
                        for a,v in item["by_arm"].items()]
        start.append(_table(["方式", "正規化角度MAE [%]", "2秒保持 [%]",
                             "関節クリップ [%]", "補正なしより改善したツール"], summary_rows))
        start.append(_mae_svg(category, item))
        start.append("<p>" + "　".join(
            f"<span style='color:{COLORS[a]}'>■</span> {LABELS[a]}" for a in ARMS) + "</p>")
        start.append("<details><summary>タスク別の数値と目標・実角度グラフ（各3初期姿勢の平均）</summary>")
        task_rows = [[f"{category.title()} {i}"] + [
            f"{100*item['by_tool'][str(i)][a]['mae_fraction']:.1f}" for a in ARMS]
            for i in range(1,6)]
        start.append(_table(["ツール"] + [LABELS[a] + " [%]" for a in ARMS],task_rows))
        for variant in range(1,6):
            start.append(f"<h4>{category.title()} {variant}</h4>"
                         + _target_svg(item["physical"],category,variant))
        start.append("</details>")
        raw_rows = [[r["task"], LABELS[r["arm"]], r["pose"],
                     f"{100*r['target_fraction']:.0f}",
                     f"{r['achieved_angle']:.4f}",
                     f"{100*r['normalized_abs_error']:.1f}",
                     "可" if r["supported"] else "不可",
                     "有" if r["joint_clipped"] else "無"]
                    for r in sorted(item["physical"],
                                    key=lambda r:(r["task"],r["pose"],r["target_fraction"],
                                                  ARMS.index(r["arm"]))) ]
        start.append(f"<details><summary>{category.title()}：全375試行の終端データ</summary>"
                     + _table(["ツール", "方式", "初期姿勢", "目標 [%]", "実角度 [rad]",
                               "正規化誤差 [%]", "保持", "関節クリップ"], raw_rows)
                     + "</details>")
    diagnostic_rows = []
    for category in CATEGORIES:
        item = report["categories"][category]
        cosines = [r["operation_axis_cosine_to_heldout"] for r in
                   item["offline"]["leave_one_tool_out"]]
        delta = item["by_arm"]["initial_pls"]["mae_fraction"] - item["by_arm"]["global_no_offset"]["mae_fraction"]
        low, high = item["initial_pls_minus_no_offset_tool_bootstrap95"]
        diagnostic_rows.append([category.title(),
                                f"{np.mean(cosines):.2f} / {np.min(cosines):.2f}",
                                f"{item['offline']['median_initial_nn_distance_ratio']:.1f}×",
                                f"{100*item['initial_pls_mean_physical_output_shift_fraction']:.1f}",
                                f"{100*delta:+.1f} [{100*low:+.1f}, {100*high:+.1f}]"])
    start.append("<h2>汎化を表す補助尺度</h2><p>角度MAEに加え、"
                 "既知ツールの軸と未知ツール専用軸の向き、初期姿勢の学習分布からの距離、"
                 "第2軸補正が実際のツール角度を変える大きさを確認しました。"
                 "専用軸との余弦は未知ツールDBを評価時だけ使う診断値で、実行時には使いません。"
                 "区間は5ツール単位で再標本化した記述的95%範囲であり、有意差検定ではありません。</p>")
    start.append(_table(["カテゴリ", "操作軸余弦 平均/最小", "初期姿勢最近傍距離比",
                         "第2軸による実角度変化 [%]", "第2軸MAE差 [pt] と95%範囲"],
                        diagnostic_rows))
    sc = report["categories"]["scissors"]
    sp = report["categories"]["spray"]
    pu = report["categories"]["push"]
    start.append("<h2>解釈と次の実装判断</h2>"
                 f"<p>初期13関節PLS補正はScissorsの平均MAEを"
                 f"{100*sc['by_arm']['global_no_offset']['mae_fraction']:.1f}%から"
                 f"{100*sc['by_arm']['initial_pls']['mae_fraction']:.1f}%へ下げましたが、"
                 f"改善は{sc['improved_tool_count']['initial_pls']}/5ツールのみで、"
                 "主にScissors 3による効果です。"
                 f"Sprayは{100*sp['by_arm']['global_no_offset']['mae_fraction']:.1f}%→"
                 f"{100*sp['by_arm']['initial_pls']['mae_fraction']:.1f}%の小幅改善、"
                 f"Pushは{100*pu['by_arm']['global_no_offset']['mae_fraction']:.1f}%→"
                 f"{100*pu['by_arm']['initial_pls']['mae_fraction']:.1f}%と悪化しました。"
                 "原案の単一射影値も、最近傍による第1軸選択も、一貫して共通軸を上回りません。"
                 f"Scissorsの関節指令クリップは補正なしの"
                 f"{100*sc['by_arm']['global_no_offset']['joint_clip_fraction']:.1f}%から"
                 f"補正ありの{100*sc['by_arm']['initial_pls']['joint_clip_fraction']:.1f}%へ増えました。"
                 "よって第2軸を標準の未知ツール制御器として採用する根拠は現段階では不足します。</p>"
                 "<p>第2軸は既知ツールの未使用初期姿勢で成功姿勢の残差を説明しますが、"
                 "関節空間で第1軸と直交しても、接触を介する実際のツール角度応答は直交しません。"
                 "初期姿勢最近傍から軸を選ぶ方法はタスクIDを使わずに済みますが、"
                 "姿勢が似ていてもツールの力学・角度写像が同じとは限りません。</p>"
                 "<p><strong>次回の実装候補：</strong>まず共通の第1軸だけを維持し、"
                 "新規ツールで安全な少数点の物理較正を行って角度逆写像の傾き・切片を合わせる。"
                 "それをゼロショット方式と明確に分けて、較正点数対MAE・Coverageを評価する。"
                 "また、初期姿勢の最近傍距離を外挿の警告指標とし、操作軸余弦と物理応答勾配は"
                 "評価用診断指標に留める。追加の第2軸を使うなら、較正によって目標角度以外の"
                 "把持・姿勢品質を独立に改善することを確認した場合に限る。</p>"
                 "<p>判定上の制限：3初期姿勢/ツールの予備実験であり、"
                 "線形逆写像は成功姿勢DB上の回帰で、物理Calibrationをしていません。"
                 "タスク専用参照は未知ツールのDBを用いるためゼロショットではありません。"
                 "未知ツールに適切な初期姿勢を人が設計できるという前提も残ります。"
                 "Scissorsの緩和Palm球による保持を実機の把持安全性と同一視しません。</p>")
    return "".join(start) + "</html>"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("analysis/initial_pose_conditioned_tds_20261007"))
    summarize(parser.parse_args().root)
