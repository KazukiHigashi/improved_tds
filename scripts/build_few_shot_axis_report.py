"""Build a self-contained Japanese report for few-shot TDS axis adaptation."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from few_shot_axis_adaptation import ARMS
from evaluate_initial_pose_conditioning import CATEGORIES


LABELS = {
    "common_db_ff": "共通軸・既知DB逆写像",
    "common_fewshot": "共通軸・3点較正",
    "selected_fewshot": "5候補から実測選択・3点較正",
    "local_oracle": "対象ツール専用DB（参考）",
}
COLORS = {"common_db_ff":"#68737d", "common_fewshot":"#008f88",
          "selected_fewshot":"#e56b35", "local_oracle":"#7355a7"}
UNITS = {"scissors":"rad", "spray":"rad", "push":"m"}


def esc(value):
    return html.escape(str(value))


def table(headers, rows):
    head = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>"+"".join(f"<td>{esc(x)}</td>" for x in row)+"</tr>"
                   for row in rows)
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def valid_coverage(rows, lo, hi):
    angles = [r["achieved_angle"] for r in rows if r.get("supported")]
    if not angles:
        return 0.
    return max(0., min(hi, max(angles))-max(lo, min(angles)))/(hi-lo)


def metric(rows, lo, hi):
    good = [r for r in rows if r.get("available")]
    if not good:
        return dict(available=0, mae_fraction=None, coverage_fraction=0.,
                    strict_hold_fraction=None, final_support_fraction=None,
                    joint_clip_fraction=None, inverse_clip_fraction=None)
    return dict(available=len(good),
        mae_fraction=float(np.mean([r["normalized_abs_error"] for r in good])),
        coverage_fraction=float(valid_coverage(good,lo,hi)),
        strict_hold_fraction=float(np.mean([r["supported"] for r in good])),
        final_support_fraction=float(np.mean([r["final_supported"] for r in good])),
        joint_clip_fraction=float(np.mean([r["joint_clipped"] for r in good])),
        inverse_clip_fraction=float(np.mean([r["inverse_clipped"] for r in good])))


def summarize_guarded(root):
    if not all((root/f"{category}_lite_{variant}.json").is_file()
               for category in CATEGORIES for variant in range(1,6)):
        return None
    tasks={}
    categories={}
    has_local_pls=all((root/f"{category}_lite_{variant}_local_pls.json").is_file()
                      for category in CATEGORIES for variant in range(1,6))
    for category in CATEGORIES:
        for variant in range(1,6):
            name=f"{category}_lite_{variant}"
            item=json.loads((root/f"{name}.json").read_text(encoding="utf-8"))
            if len(item["physical"])!=15 or item["calibration_pose"] in item["test_poses"]:
                raise ValueError(f"invalid guarded validation: {name}")
            rows=item["physical"]
            lo,hi=item["target_range"]
            item["metrics"]={arm:dict(
                mae_fraction=float(np.mean([r[f"{arm}_mae_fraction"] for r in rows])),
                strict_hold_fraction=float(np.mean([r[f"{arm}_supported"] for r in rows])),
                joint_clip_fraction=float(np.mean([r[f"{arm}_joint_clipped"] for r in rows])),
                coverage_fraction=valid_coverage([
                    dict(achieved_angle=r[f"{arm}_achieved"],supported=r[f"{arm}_supported"])
                    for r in rows],lo,hi)) for arm in ("baseline","adapted")}
            if has_local_pls:
                local=json.loads((root/f"{name}_local_pls.json").read_text(encoding="utf-8"))
                local_rows=local["rows"]
                if len(local_rows)!=len(rows) or any(
                    a["pose"]!=b["pose"] or a["target_fraction"]!=b["target_fraction"]
                    or abs(a["target_angle"]-b["target_angle"])>1e-12
                    for a,b in zip(local_rows,rows)):
                    raise ValueError(f"task-specific PLS trials not paired: {name}")
                item["metrics"]["local_pls"]=dict(
                    mae_fraction=float(np.mean([r["mae_fraction"] for r in local_rows])),
                    strict_hold_fraction=float(np.mean([r["supported"] for r in local_rows])),
                    joint_clip_fraction=float(np.mean([r["joint_clipped"] for r in local_rows])),
                    coverage_fraction=valid_coverage([
                        dict(achieved_angle=r["achieved"],supported=r["supported"])
                        for r in local_rows],lo,hi))
                if "calibrated_mae_fraction" in local_rows[0]:
                    item["metrics"]["local_pls_calibrated"]=dict(
                        mae_fraction=float(np.mean([
                            r["calibrated_mae_fraction"] for r in local_rows])),
                        strict_hold_fraction=float(np.mean([
                            r["calibrated_supported"] for r in local_rows])),
                        joint_clip_fraction=float(np.mean([
                            r["calibrated_joint_clipped"] for r in local_rows])),
                        coverage_fraction=valid_coverage([
                            dict(achieved_angle=r["calibrated_achieved"],
                                 supported=r["calibrated_supported"])
                            for r in local_rows],lo,hi),
                        calibration_available=local["calibration_response"] is not None)
            tasks[name]=item
        selected=[tasks[f"{category}_lite_{i}"] for i in range(1,6)]
        differences=np.asarray([x["metrics"]["adapted"]["mae_fraction"]
                                -x["metrics"]["baseline"]["mae_fraction"] for x in selected])
        rng=np.random.default_rng(99100+len(category))
        boot=differences[rng.integers(0,5,(20000,5))].mean(axis=1)
        categories[category]=dict(
            baseline_mae_fraction=float(np.mean([x["metrics"]["baseline"]["mae_fraction"]
                                              for x in selected])),
            adapted_mae_fraction=float(np.mean([x["metrics"]["adapted"]["mae_fraction"]
                                             for x in selected])),
            baseline_coverage_fraction=float(np.mean([x["metrics"]["baseline"]["coverage_fraction"]
                                                   for x in selected])),
            adapted_coverage_fraction=float(np.mean([x["metrics"]["adapted"]["coverage_fraction"]
                                                  for x in selected])),
            baseline_hold_fraction=float(np.mean([x["metrics"]["baseline"]["strict_hold_fraction"]
                                               for x in selected])),
            adapted_hold_fraction=float(np.mean([x["metrics"]["adapted"]["strict_hold_fraction"]
                                              for x in selected])),
            mean_probes=float(np.mean([x["probes_attempted"] for x in selected])),
            three_probe_tools=sum(x["probes_attempted"]==3 for x in selected),
            improved_tool_count=int(np.sum(differences<0)),
            mean_mae_difference=float(differences.mean()),
            bootstrap95_tool_mean_difference=np.quantile(boot,[.025,.975]).tolist())
        if has_local_pls:
            local_diff=np.asarray([x["metrics"]["adapted"]["mae_fraction"]
                                   -x["metrics"]["local_pls"]["mae_fraction"]
                                   for x in selected])
            local_boot=local_diff[rng.integers(0,5,(20000,5))].mean(axis=1)
            categories[category].update(
                local_pls_mae_fraction=float(np.mean([x["metrics"]["local_pls"]["mae_fraction"]
                                                      for x in selected])),
                local_pls_coverage_fraction=float(np.mean([
                    x["metrics"]["local_pls"]["coverage_fraction"] for x in selected])),
                local_pls_hold_fraction=float(np.mean([
                    x["metrics"]["local_pls"]["strict_hold_fraction"] for x in selected])),
                local_pls_joint_clip_fraction=float(np.mean([
                    x["metrics"]["local_pls"]["joint_clip_fraction"] for x in selected])),
                adapted_better_than_local_count=int(np.sum(local_diff<0)),
                adapted_minus_local_mae_fraction=float(local_diff.mean()),
                adapted_minus_local_bootstrap95=np.quantile(local_boot,[.025,.975]).tolist())
            if all("local_pls_calibrated" in x["metrics"] for x in selected):
                calibrated_diff=np.asarray([
                    x["metrics"]["adapted"]["mae_fraction"]
                    -x["metrics"]["local_pls_calibrated"]["mae_fraction"]
                    for x in selected])
                calibrated_boot=calibrated_diff[
                    rng.integers(0,5,(20000,5))].mean(axis=1)
                categories[category].update(
                    local_pls_calibrated_mae_fraction=float(np.mean([
                        x["metrics"]["local_pls_calibrated"]["mae_fraction"]
                        for x in selected])),
                    local_pls_calibrated_coverage_fraction=float(np.mean([
                        x["metrics"]["local_pls_calibrated"]["coverage_fraction"]
                        for x in selected])),
                    local_pls_calibrated_available_tools=sum(
                        x["metrics"]["local_pls_calibrated"]["calibration_available"]
                        for x in selected),
                    adapted_better_than_calibrated_local_count=int(np.sum(calibrated_diff<0)),
                    adapted_minus_calibrated_local_mae_fraction=float(calibrated_diff.mean()),
                    adapted_minus_calibrated_local_bootstrap95=np.quantile(
                        calibrated_boot,[.025,.975]).tolist())
    return dict(protocol="few_shot_guarded_new_pose_v1",tasks=tasks,categories=categories,
                task_specific_pls_complete=has_local_pls,
                calibrated_task_pls_complete=has_local_pls and all(
                    "local_pls_calibrated" in x["metrics"] for x in tasks.values()))


def summarize(root):
    tasks = {}
    legacy_root = root.parent/"initial_pca2_nn_20261008"
    baseline_audits = {}
    for category in CATEGORIES:
        for variant in range(1,6):
            name = f"{category}_lite_{variant}"
            path = root/f"{name}.json"
            item = json.loads(path.read_text(encoding="utf-8"))
            if item["task"] != name or len(item["physical"]) != 3*5*len(ARMS):
                raise ValueError(f"incomplete or mismatched physical experiment: {path}")
            if item["calibration_pose"] in item["test_poses"]:
                raise ValueError(f"calibration/test pose leakage: {name}")
            legacy_path = legacy_root/f"{name}_physical.json"
            if legacy_path.is_file():
                legacy = [r for r in json.loads(legacy_path.read_text(encoding="utf-8"))
                          if r["arm"]=="common_axis"]
                present = [r for r in item["physical"] if r["arm"]=="common_db_ff"]
                if len(legacy) != len(present) or any(
                    a["pose"] != b["pose"] or a["target_fraction"] != b["target_fraction"]
                    or abs(a["achieved_angle"]-b["achieved_angle"]) > 1e-10
                    for a,b in zip(legacy,present)):
                    raise ValueError(f"common-axis baseline differs from earlier experiment: {name}")
                baseline_audits[name] = True
            lo, hi = item["target_range"]
            item["metrics"] = {arm:metric([r for r in item["physical"] if r["arm"]==arm],lo,hi)
                               for arm in ARMS}
            # Predeclared safe fallback preserves intention-to-treat denominators.
            baseline = {(r["pose"],r["target_fraction"]):r for r in item["physical"]
                        if r["arm"]=="common_db_ff"}
            selected = [r if r.get("available") else baseline[(r["pose"],r["target_fraction"])]
                        for r in item["physical"] if r["arm"]=="selected_fewshot"]
            item["metrics"]["selected_with_fallback"] = metric(selected,lo,hi)
            tasks[name] = item
    category_summary = {}
    for category in CATEGORIES:
        items = [tasks[f"{category}_lite_{variant}"] for variant in range(1,6)]
        by_arm = {}
        for arm in (*ARMS,"selected_with_fallback"):
            physical = []
            if arm == "selected_with_fallback":
                for item in items:
                    baseline = {(r["pose"],r["target_fraction"]):r for r in item["physical"]
                                if r["arm"]=="common_db_ff"}
                    physical.extend(r if r.get("available") else baseline[(r["pose"],r["target_fraction"])]
                                    for r in item["physical"] if r["arm"]=="selected_fewshot")
            else:
                physical = [r for item in items for r in item["physical"] if r["arm"]==arm]
            # Coverage must be averaged by tool; pooled min/max falsely bridges tools.
            by_arm[arm] = dict(
                mean_mae_fraction=float(np.mean([r["normalized_abs_error"] for r in physical
                                                  if r.get("available")] ))
                    if any(r.get("available") for r in physical) else None,
                mean_tool_coverage_fraction=float(np.mean([
                    item["metrics"][arm]["coverage_fraction"] for item in items])),
                available_trials=sum(bool(r.get("available")) for r in physical),
                total_trials=len(physical),
                strict_hold_fraction=float(np.mean([r["supported"] for r in physical
                    if r.get("available")])) if any(r.get("available") for r in physical) else None,
                joint_clip_fraction=float(np.mean([r["joint_clipped"] for r in physical
                    if r.get("available")])) if any(r.get("available") for r in physical) else None)
        differences = np.asarray([
            item["metrics"]["selected_with_fallback"]["mae_fraction"]
            -item["metrics"]["common_db_ff"]["mae_fraction"] for item in items])
        rng = np.random.default_rng(8000+len(category))
        boot = differences[rng.integers(0,5,(20000,5))].mean(axis=1)
        category_summary[category] = dict(by_arm=by_arm,
            selected_calibratable_tools=sum(item["selected_candidate"] is not None for item in items),
            common_calibratable_tools=sum(item["models"]["common"]["response"] is not None
                                         for item in items),
            improved_tool_count=int(np.sum(differences < 0)),
            tool_mean_mae_difference=float(np.mean(differences)),
            bootstrap95_tool_mean_difference=np.quantile(boot,[.025,.975]).tolist())
    result = dict(protocol="few_shot_axis_selection_v1", training="none",
        selection_uses="one held-out-tool calibration reset pose, three measured points per known-axis package",
        heldout_success_database="used only by local_oracle reference",
        target_trials_per_tool=15, calibration_probes_per_tool=15,
        fallback="if no package can be calibrated, use common-axis DB FF",
        matched_earlier_baseline_tasks=sorted(baseline_audits),
        categories=category_summary, tasks=tasks,
        new_pose_validation=summarize_guarded(root.parent/"few_shot_guarded_20261008"))
    (root/"summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    (root/"index.html").write_text(render(result),encoding="utf-8")
    return result


def mae_svg(category, tasks):
    xs = [105+130*i for i in range(5)]
    values = [tasks[f"{category}_lite_{i}"]["metrics"] for i in range(1,6)]
    arms = ("common_db_ff","selected_with_fallback","local_oracle")
    colors = [COLORS["common_db_ff"],COLORS["selected_fewshot"],COLORS["local_oracle"]]
    high = max(60., 110*max(m[a]["mae_fraction"] for m in values for a in arms))
    y = lambda v: 250-200*(100*v/high)
    out = [f"<svg viewBox='0 0 740 315' role='img' aria-label='{category} タスク別MAE'>",
           "<line x1='75' y1='50' x2='75' y2='250' stroke='#333'/>",
           "<line x1='75' y1='250' x2='670' y2='250' stroke='#333'/>",
           "<text x='8' y='26'>MAE［目標範囲に対する%］</text>",
           "<text x='330' y='305'>ツール番号</text>"]
    for tick in np.linspace(0,high,5):
        yy=250-200*tick/high
        out.append(f"<line x1='75' y1='{yy:.1f}' x2='670' y2='{yy:.1f}' stroke='#e3e7ea'/>")
        out.append(f"<text x='29' y='{yy+4:.1f}'>{tick:.0f}</text>")
    for i,xx in enumerate(xs,1):
        out.append(f"<text x='{xx-3}' y='270'>{i}</text>")
    for arm,color in zip(arms,colors):
        points = " ".join(f"{xx},{y(m[arm]['mae_fraction']):.1f}"
                          for xx,m in zip(xs,values))
        out.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        for xx,m in zip(xs,values):
            out.append(f"<circle cx='{xx}' cy='{y(m[arm]['mae_fraction']):.1f}' r='4' fill='{color}'/>")
    return "".join(out)+"</svg>"


def task_svg(item):
    lo,hi=item["target_range"]
    normalize=lambda value:(value-lo)/(hi-lo)
    x=lambda value:60+220*normalize(value)
    y=lambda value:255-220*normalize(value)
    out=["<svg viewBox='0 0 350 320' role='img' aria-label='目標に対する実測応答'>",
         "<line x1='60' y1='35' x2='60' y2='255' stroke='#333'/>",
         "<line x1='60' y1='255' x2='280' y2='255' stroke='#333'/>",
         "<line x1='60' y1='255' x2='280' y2='35' stroke='#bbb' stroke-dasharray='4 4'/>",
         "<text x='95' y='305'>目標［目標範囲%］</text>",
         "<text x='4' y='22'>実測［目標範囲%］</text>"]
    for tick in (0,.25,.5,.75,1):
        out.append(f"<text x='{60+220*tick-5:.1f}' y='274'>{100*tick:.0f}</text>")
        out.append(f"<text x='28' y='{255-220*tick+4:.1f}'>{100*tick:.0f}</text>")
    for arm in ("common_db_ff","selected_fewshot","local_oracle"):
        rows=[r for r in item["physical"] if r["arm"]==arm and r.get("available")]
        for fraction in sorted({r["target_fraction"] for r in rows}):
            subset=[r for r in rows if r["target_fraction"]==fraction]
            achieved=np.mean([r["achieved_angle"] for r in subset])
            out.append(f"<circle cx='{x(subset[0]['target_angle']):.1f}' cy='{y(achieved):.1f}' "
                       f"r='4' fill='{COLORS[arm]}' opacity='.8'/>")
    return "".join(out)+"</svg>"


def validation_mae_svg(category, tasks):
    xs=[105+130*i for i in range(5)]
    values=[tasks[f"{category}_lite_{i}"]["metrics"] for i in range(1,6)]
    arms=("baseline","adapted")
    if "local_pls" in values[0]:
        arms+=("local_pls",)
    if "local_pls_calibrated" in values[0]:
        arms+=("local_pls_calibrated",)
    high=max(60.,110*max(m[a]["mae_fraction"] for m in values for a in arms))
    y=lambda v:250-200*(100*v/high)
    out=[f"<svg viewBox='0 0 740 315' role='img' aria-label='{category} 別姿勢追試MAE'>",
         "<line x1='75' y1='50' x2='75' y2='250' stroke='#333'/>",
         "<line x1='75' y1='250' x2='670' y2='250' stroke='#333'/>",
         "<text x='8' y='26'>MAE［目標範囲に対する%］</text>",
         "<text x='330' y='305'>ツール番号</text>"]
    for tick in np.linspace(0,high,5):
        yy=250-200*tick/high
        out.append(f"<line x1='75' y1='{yy:.1f}' x2='670' y2='{yy:.1f}' stroke='#e3e7ea'/>")
        out.append(f"<text x='29' y='{yy+4:.1f}'>{tick:.0f}</text>")
    for i,xx in enumerate(xs,1):
        out.append(f"<text x='{xx-3}' y='270'>{i}</text>")
    for arm,color in (("baseline",COLORS["common_db_ff"]),
                      ("adapted",COLORS["selected_fewshot"]),
                      ("local_pls",COLORS["local_oracle"]),
                      ("local_pls_calibrated","#b84c91")):
        if arm not in arms:
            continue
        points=" ".join(f"{xx},{y(m[arm]['mae_fraction']):.1f}"
                        for xx,m in zip(xs,values))
        out.append(f"<polyline points='{points}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        out.extend(f"<circle cx='{xx}' cy='{y(m[arm]['mae_fraction']):.1f}' r='4' fill='{color}'/>"
                   for xx,m in zip(xs,values))
    return "".join(out)+"</svg>"


def render(report):
    out=["<!doctype html><html lang='ja'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>未知ツールへの少数試行TDS適応</title>",
         "<style>body{font:16px/1.65 system-ui,sans-serif;color:#1a2933;max-width:1180px;margin:2rem auto;padding:0 1rem}h1,h2,h3{line-height:1.3}table{border-collapse:collapse;width:100%;font-size:.93rem}th,td{border:1px solid #ccd6dd;padding:.4rem;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#eef3f6}svg{max-width:100%;height:auto}details{border:1px solid #cbd5dc;border-radius:6px;padding:.65rem;margin:1rem 0}summary{cursor:pointer;font-weight:600}.note{background:#f2f7f9;padding:.8rem;border-left:4px solid #008f88}.warn{background:#fff7ec;padding:.8rem;border-left:4px solid #e56b35}.legend span{display:inline-block;margin-right:1rem}.dot{display:inline-block;width:.8rem;height:.8rem;margin-right:.3rem;border-radius:50%}code{background:#eff3f6;padding:.1rem .2rem}</style></head><body>",
         "<h1>未知ツールに対する少数試行によるTDS操作軸適応</h1>",
         "<p class='note'>既存の成功姿勢DBと学習済みモデルを固定し、新たな強化学習は行っていません。未知ツールの成功姿勢DBは候補作成・軸選択・逆写像較正には使わず、参考用の対象ツール専用モデルにだけ使用しました。</p>",
         "<h2>問いと実験設計</h2>",
         "<p>各カテゴリ5ツールのうち1つを未知ツールとして除外し、既知4ツールの成功姿勢から共通PLS操作軸と各既知ツール専用PLS操作軸を作りました。各候補について未知ツールの較正用初期姿勢1種類で、既知DBに由来する3スコアを独立に試行します。ツール応答はこの実測値のみから線形に近似し、目標ツール量の範囲に入る実測範囲が最大の候補を選択しました。較正と評価の初期姿勢は分離しています。</p>",
         "<p>フィッティングには最終時点で支持され、数値異常や転倒がない3点のみを使用します。途中の一時的な保持判定不成立だけでは除外しません。ただし3点が単調でない候補、1点でも最終支持を失った候補は選びません。選択後はその候補軸のスコアだけを変える1自由度FF制御です。各試行は初期姿勢から5秒線形軌道、終端2秒保持。Scissorsは1.5倍Palm球、Push/Sprayは接触数でゲートせず転倒・落下を除外し、Sprayの床接触は有効です。</p>",
         "<p>較正予算は1ツール当たり最大15試行（5候補×3点）、評価は別の初期姿勢3種類×目標5点×方式です。1試行は7秒のシミュレーション時間です。対象ツールのタスクIDは候補選択の数値特徴ではありません。</p>",
         f"<p>従来実験の共通軸FFと同一条件・同一実測値であることを確認したタスク：{len(report['matched_earlier_baseline_tasks'])}/15。</p>",
         "<h2>集計</h2>",
         "<p class='legend'>"+"".join(f"<span><i class='dot' style='background:{COLORS[a]}'></i>{LABELS[a]}</span>"
                                   for a in ("common_db_ff","selected_fewshot","local_oracle"))+"</p>"]
    overview=[]
    for cat in CATEGORIES:
        c=report["categories"][cat]
        b=c["by_arm"]
        fmt=lambda arm,key: f"{100*b[arm][key]:.1f}" if b[arm][key] is not None else "—"
        overview.append([cat.title(),f"{c['selected_calibratable_tools']}/5",
            f"{c['common_calibratable_tools']}/5",
            fmt("common_db_ff","mean_mae_fraction"),
            fmt("common_fewshot","mean_mae_fraction")+f" ({b['common_fewshot']['available_trials']}/75)",
            fmt("selected_with_fallback","mean_mae_fraction"),
            fmt("local_oracle","mean_mae_fraction"),
            f"{c['improved_tool_count']}/5"])
    out.append(table(["カテゴリ","軸選択可能","共通軸較正可能","共通DB FF MAE [%]",
        "共通軸較正MAE [%]（利用試行）","選択＋安全fallback MAE [%]",
        "対象専用参考MAE [%]","改善ツール数"],overview))
    out.append("<p class='warn'>較正不能時にその条件を集計から落とすと結果が楽観的になります。主比較は、選択不能なら共通軸DB FFへ戻す事前定義のfallbackを含む全75評価試行／カテゴリで行います。共通軸較正単独のMAEは利用可能試行に限る条件付き値です。対象専用モデルは未知ツールの成功姿勢DBを使うため実用的な比較条件ではなく参考値です。</p>")
    for cat in CATEGORIES:
        c=report["categories"][cat]
        out.append(f"<h2>{cat.title()}</h2>")
        out.append(mae_svg(cat,report["tasks"]))
        diff=c["tool_mean_mae_difference"]
        ci=c["bootstrap95_tool_mean_difference"]
        out.append(f"<p>選択＋fallback − 共通DB FF のツール平均MAE差：{100*diff:+.1f}ポイント（ツール単位bootstrap 95%区間 {100*ci[0]:+.1f}～{100*ci[1]:+.1f}）。5ツールしかないため、区間は探索的な不確実性表示です。</p>")
        cov=[]
        for arm in ("common_db_ff","common_fewshot","selected_fewshot","selected_with_fallback","local_oracle"):
            m=c["by_arm"][arm]
            label=LABELS.get(arm,"選択＋共通DB FF fallback")
            cov.append([label,f"{m['available_trials']}/{m['total_trials']}",
                "—" if m["mean_mae_fraction"] is None else f"{100*m['mean_mae_fraction']:.1f}",
                f"{100*m['mean_tool_coverage_fraction']:.1f}",
                "—" if m["strict_hold_fraction"] is None else f"{100*m['strict_hold_fraction']:.1f}",
                "—" if m["joint_clip_fraction"] is None else f"{100*m['joint_clip_fraction']:.1f}"])
        out.append(table(["方式","利用評価試行","MAE [%]","実測Coverage [%]",
                          "終端2秒採用率 [%]","関節指令clip [%]"],cov))
        detail=[]
        for i in range(1,6):
            item=report["tasks"][f"{cat}_lite_{i}"]
            a=item["metrics"]
            detail.append([i,item["selected_candidate"] or "較正不能",
                f"{sum(x['response'] is not None for x in item['models'].values())}/5",
                f"{100*a['common_db_ff']['mae_fraction']:.1f}",
                "—" if a["selected_fewshot"]["mae_fraction"] is None else
                    f"{100*a['selected_fewshot']['mae_fraction']:.1f}",
                f"{100*a['selected_with_fallback']['mae_fraction']:.1f}",
                f"{100*a['local_oracle']['mae_fraction']:.1f}"])
        out.append(table(["ツール","採用した候補","較正可能候補","共通DB MAE [%]",
                          "選択時MAE [%]","選択＋fallback MAE [%]","対象専用MAE [%]"],detail))
        for i in range(1,6):
            item=report["tasks"][f"{cat}_lite_{i}"]
            out.append(f"<details><summary>{cat.title()} {i}：較正3点と目標応答</summary>")
            out.append(task_svg(item))
            probes=[]
            for name,model in item["models"].items():
                response=model["response"]
                probes.append([name,
                    ", ".join(f"{p['score']:.3f}→{p['achieved_angle']:.4g}"
                              for p in model["probes"]),
                    "/".join("○" if p["admitted"] else "×" for p in model["probes"]),
                    "不可" if response is None else f"{100*response['coverage_fraction']:.1f}",
                    "—" if response is None else f"{response['slope']:.4g}"])
            out.append(table(["候補軸","スコア→実測ツール量［"+UNITS[cat]+"］",
                              "最終採用","較正Coverage [%]","実測傾き［"+UNITS[cat]+"/score］"],probes))
            out.append(f"<p>目標量・実測量の単位：{UNITS[cat]}。散布図はツールごとの目標範囲に正規化。採用候補：{esc(item['selected_candidate'] or 'なし')}。</p></details>")
    validation=report["new_pose_validation"]
    if validation is not None:
        out.append("<h2>別の初期姿勢での追試：共通軸優先方式</h2>")
        out.append("<p>初回実験では応答範囲が広い既知軸を常に選ぶと、共通軸の較正だけで十分な例まで切り替えて悪化することがありました。そこで初回結果を踏まえて『共通軸3点の実測Coverageが75%以上なら探索を止める』規則を設定し、初回の較正・評価姿勢をすべて除いた別の姿勢群で追試しました。75%という閾値は初回データを見て設定した探索的な値であり、未知のツール群による独立検証ではありません。</p>")
        rows=[]
        for cat in CATEGORIES:
            x=validation["categories"][cat]
            row=[cat.title(),f"{100*x['baseline_mae_fraction']:.1f}",
                f"{100*x['adapted_mae_fraction']:.1f}"]
            if validation["task_specific_pls_complete"]:
                row.append(f"{100*x['local_pls_mae_fraction']:.1f}")
            if validation["calibrated_task_pls_complete"]:
                row.append(f"{100*x['local_pls_calibrated_mae_fraction']:.1f}")
            row.extend([f"{100*x['baseline_coverage_fraction']:.1f}",
                f"{100*x['adapted_coverage_fraction']:.1f}"])
            if validation["task_specific_pls_complete"]:
                row.append(f"{100*x['local_pls_coverage_fraction']:.1f}")
            if validation["calibrated_task_pls_complete"]:
                row.append(f"{100*x['local_pls_calibrated_coverage_fraction']:.1f}")
            row.extend([f"{x['mean_probes']:.1f}",f"{x['three_probe_tools']}/5",
                f"{x['improved_tool_count']}/5"])
            rows.append(row)
        heads=["カテゴリ","共通FF MAE [%]","適応MAE [%]"]
        if validation["task_specific_pls_complete"]:
            heads.append("タスク専用PLS MAE [%]")
        if validation["calibrated_task_pls_complete"]:
            heads.append("タスク専用PLS＋3点較正 MAE [%]")
        heads.extend(["共通FF Coverage [%]","適応Coverage [%]"])
        if validation["task_specific_pls_complete"]:
            heads.append("タスク専用PLS Coverage [%]")
        if validation["calibrated_task_pls_complete"]:
            heads.append("専用PLS＋較正 Coverage [%]")
        heads.extend(["平均較正試行数","3試行で停止","適応が共通FFに勝つツール数"])
        out.append(table(heads,rows))
        out.append("<p class='legend'><span><i class='dot' style='background:#68737d'></i>共通軸FF</span><span><i class='dot' style='background:#e56b35'></i>段階的適応</span><span><i class='dot' style='background:#7355a7'></i>タスク専用PLS</span><span><i class='dot' style='background:#b84c91'></i>専用PLS＋3点較正</span></p>")
        if validation["task_specific_pls_complete"]:
            out.append("<p class='warn'>タスク専用PLSは当該ツールの成功姿勢200件から推定した操作軸・平均姿勢とDB線形逆写像を使用します。未知ツールの成功DBを使用するためFew-shot適応法ではありません。また物理応答の3点較正を行わないため、性能の数学的上限でもありません。各方式は同じ評価初期姿勢・目標値で比較しています。</p>")
        if validation["calibrated_task_pls_complete"]:
            out.append("<p>比較を補うため、タスク専用PLS軸にも同じ較正姿勢で3点の物理応答を測り、逆写像を作り直しました。これも未知ツールの成功DBを利用する参考条件であり、Few-shot法には利用していません。較正不成立ならそのツール専用DB逆写像に戻します。</p>")
        for cat in CATEGORIES:
            x=validation["categories"][cat]
            ci=x["bootstrap95_tool_mean_difference"]
            out.append(f"<h3>{cat.title()}：別姿勢追試</h3>")
            out.append(validation_mae_svg(cat,validation["tasks"]))
            out.append(f"<p>適応 − 共通FF のツール平均MAE差：{100*x['mean_mae_difference']:+.1f}ポイント。ツール単位bootstrap 95%区間：{100*ci[0]:+.1f}～{100*ci[1]:+.1f}ポイント（探索的表示）。</p>")
            if validation["task_specific_pls_complete"]:
                ci_local=x["adapted_minus_local_bootstrap95"]
                out.append(f"<p>適応 − タスク専用PLS のツール平均MAE差：{100*x['adapted_minus_local_mae_fraction']:+.1f}ポイント。95%区間：{100*ci_local[0]:+.1f}～{100*ci_local[1]:+.1f}ポイント。適応のMAEが小さいツール：{x['adapted_better_than_local_count']}/5。</p>")
            if validation["calibrated_task_pls_complete"]:
                ci_cal=x["adapted_minus_calibrated_local_bootstrap95"]
                out.append(f"<p>適応 − 専用PLS＋3点較正 のツール平均MAE差：{100*x['adapted_minus_calibrated_local_mae_fraction']:+.1f}ポイント。95%区間：{100*ci_cal[0]:+.1f}～{100*ci_cal[1]:+.1f}ポイント。適応が低MAE：{x['adapted_better_than_calibrated_local_count']}/5。専用PLS較正成立：{x['local_pls_calibrated_available_tools']}/5。</p>")
        detail=[]
        for cat in CATEGORIES:
            for i in range(1,6):
                x=validation["tasks"][f"{cat}_lite_{i}"]
                row=[f"{cat.title()} {i}",x["selected_candidate"] or "共通FF fallback",
                    x["probes_attempted"],f"{100*x['metrics']['baseline']['mae_fraction']:.1f}",
                    f"{100*x['metrics']['adapted']['mae_fraction']:.1f}"]
                if validation["task_specific_pls_complete"]:
                    row.append(f"{100*x['metrics']['local_pls']['mae_fraction']:.1f}")
                if validation["calibrated_task_pls_complete"]:
                    row.append(f"{100*x['metrics']['local_pls_calibrated']['mae_fraction']:.1f}")
                row.extend([
                    f"{100*x['metrics']['baseline']['coverage_fraction']:.1f}",
                    f"{100*x['metrics']['adapted']['coverage_fraction']:.1f}"])
                if validation["task_specific_pls_complete"]:
                    row.append(f"{100*x['metrics']['local_pls']['coverage_fraction']:.1f}")
                if validation["calibrated_task_pls_complete"]:
                    row.append(f"{100*x['metrics']['local_pls_calibrated']['coverage_fraction']:.1f}")
                detail.append(row)
        detail_heads=["タスク","選択候補","較正試行数","共通FF MAE [%]","適応MAE [%]"]
        if validation["task_specific_pls_complete"]:
            detail_heads.append("タスク専用PLS MAE [%]")
        if validation["calibrated_task_pls_complete"]:
            detail_heads.append("専用PLS＋較正 MAE [%]")
        detail_heads.extend(["共通FF Coverage [%]","適応Coverage [%]"])
        if validation["task_specific_pls_complete"]:
            detail_heads.append("タスク専用PLS Coverage [%]")
        if validation["calibrated_task_pls_complete"]:
            detail_heads.append("専用PLS＋較正 Coverage [%]")
        out.append("<details><summary>追試15タスクの個別結果</summary>"+
                   table(detail_heads,detail)+
                   "</details>")
    out.append("<h2>解釈と次の検証</h2><p>これは未知ツールでの少数試行を使うfew-shot適応であり、初期姿勢だけからのzero-shot汎化とは区別します。候補探索に最大15回を要するため、実機ではこのまま無検証で実行せず、接触と関節余裕を監視し、危険な候補を早期中止する段階的探索が必要です。較正は1種類の初期姿勢のみなので、別姿勢での再現性と較正試行回数の削減をさらに検証します。候補の実測ツール量の範囲が小さい場合は逆写像だけでは改善できず、把持基準姿勢または候補方向を変える必要があります。</p>")
    out.append("</body></html>")
    return "".join(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("analysis/few_shot_axis_adaptation_20261008"))
    args=p.parse_args()
    result=summarize(args.root)
    for name,c in result["categories"].items():
        print(name,"baseline",round(c["by_arm"]["common_db_ff"]["mean_mae_fraction"],3),
              "adapted",round(c["by_arm"]["selected_with_fallback"]["mean_mae_fraction"],3),
              "improved",c["improved_tool_count"],flush=True)


if __name__=="__main__":
    main()
