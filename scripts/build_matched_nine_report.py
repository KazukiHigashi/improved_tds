"""Build a plain-Japanese paper audit from completed matched-nine JSON only."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path
from statistics import mean

from build_paper_ff_fb_generalization_report import figure, label, table

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "analysis/matched_nine_goals_clipped_20261008"
TASKS = tuple(f"{category}_lite_{i}" for category in ("scissors", "push", "spray")
              for i in range(1, 6))
FB_TASKS = ("scissors_lite_3", "scissors_lite_4", "push_lite_3", "spray_lite_3")
FRACTIONS = tuple(i / 8 for i in range(9))
UNKNOWN = ("common_zero", "pca2_nn_zero", "common_three_probe",
           "selected_three_probe", "local_pls_oracle", "local_pls_three_probe_oracle")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(rows: list[dict], *, fb: bool = False) -> dict:
    if len(rows) != 72:
        raise ValueError(f"expected 72 paired trials, got {len(rows)}")
    finite = ([r for r in rows if r["abs_error"] is not None
               and r["complete"] and not r["numerical_failure"]] if fb else
              [r for r in rows if r["finite"] and not r["terminated"]
               and not r["truncated"]])
    held = [r for r in finite if r["valid" if fb else "held"]]
    def error(r):
        return (r["abs_error"] / r["span"] if fb else r["normalized_abs_error"])
    target_key = "target" if fb else "target_angle"
    achieved_key = "achieved" if fb else "achieved_angle"
    low = min(r[target_key] for r in rows)
    high = max(r[target_key] for r in rows)
    held_values = [r[achieved_key] for r in held]
    reached = (100 * max(0., min(high,max(held_values))-max(low,min(held_values)))
               / (high-low) if held_values and high > low else 0.)
    return dict(trials=len(rows), complete_finite=len(finite), held=len(held),
        complete_finite_mae_percent=100*mean(error(r) for r in finite) if finite else None,
        held_mae_percent=100*mean(error(r) for r in held) if held else None,
        clipped_complete_finite=sum(bool(r.get("inverse_clipped", False)) for r in finite),
        clipped_mae_percent=(100*mean(error(r) for r in finite
                             if r.get("inverse_clipped", False))
                             if any(r.get("inverse_clipped", False) for r in finite) else None),
        unclipped_mae_percent=(100*mean(error(r) for r in finite
                               if not r.get("inverse_clipped", False))
                               if any(not r.get("inverse_clipped", False) for r in finite) else None),
        held_reached_span_percent=reached,
        inverse_clipped=sum(bool(r.get("inverse_clipped", False)) for r in rows),
        joint_clipped=sum(bool(r.get("joint_clipped", r.get("ff_clipped", False))) for r in rows),
        early_or_numerical=len(rows)-len(finite))


def paired_delta(first: list[dict], second: list[dict], *, fb: bool = False,
                 span: float | None = None) -> dict:
    key = ((lambda r:(r["pose"],r["target"])) if fb else
           (lambda r:(r["pose"],r["target_fraction"])))
    left, right = ({key(r):r for r in group} for group in (first, second))
    if set(left) != set(right) or len(left) != 72:
        raise ValueError("paired comparison does not share all 72 pose/target keys")
    differences = []
    for k in left:
        a, b = left[k], right[k]
        if fb:
            good = (a["abs_error"] is not None and b["abs_error"] is not None
                    and a["complete"] and b["complete"]
                    and not a["numerical_failure"] and not b["numerical_failure"])
            delta = (b["abs_error"]-a["abs_error"])/span if good else None
        else:
            good = (a["finite"] and b["finite"] and not a["terminated"]
                    and not b["terminated"] and not a["truncated"]
                    and not b["truncated"])
            delta = ((b["normalized_abs_error"]-a["normalized_abs_error"])
                     if good else None)
        if good:
            differences.append(delta)
    return dict(paired_complete_finite=len(differences),
                second_minus_first_mae_percent=(100*mean(differences)
                                                 if differences else None),
                second_better_pairs=sum(delta < 0 for delta in differences))


def load() -> dict:
    status = read(RESULT / "status.json")
    if status["expected_jobs"] != 34 or status["completed_jobs"] != 34:
        raise ValueError("matched campaign is not complete")
    if any(job["status"] not in ("complete", "already_complete") for job in status["jobs"]):
        raise ValueError("matched campaign has a failed job")
    for name, checksum in status["code_sha256"].items():
        if digest(ROOT / name) != checksum:
            raise ValueError(f"experiment source changed: {name}")
    result = dict(protocol=status["protocol"], source_hashes=status["code_sha256"],
                  raw_result_hashes={},
                  target_ranges={}, ff={}, unknown={}, fb={},
                  paired={"ff":{},"unknown":{},"fb":{}})
    for task in TASKS:
        ff_path = RESULT / "ff" / f"{task}.json"
        un_path = RESULT / "generalization" / f"{task}.json"
        result["raw_result_hashes"][str(ff_path.relative_to(RESULT)).replace("\\", "/")] = digest(ff_path)
        result["raw_result_hashes"][str(un_path.relative_to(RESULT)).replace("\\", "/")] = digest(un_path)
        ff = read(ff_path)
        un = read(un_path)
        if (ff["task"] != task or un["task"] != task
                or ff["protocol"] != "frozen_ff_common_full_rl_nine_v1"
                or un["protocol"] != "unknown_tool_common_full_rl_nine_v1"
                or len(ff["test_poses"]) != 8 or ff["test_poses"] != un["test_poses"]
                or ff["target_fractions"] != un["target_fractions"]
                or not all(abs(a-b) < 1e-12 for a,b in zip(FRACTIONS,ff["target_fractions"]))
                or ff["source_report_sha256"] != un["source_report_sha256"]):
            raise ValueError(f"FF/unknown pairing differs: {task}")
        ff_keys = {(r["pose"],r["target_fraction"]) for r in ff["rows"]}
        un_keys = {(r["pose"],r["target_fraction"]) for r in un["rows"]}
        if len(ff_keys) != 72 or ff_keys != un_keys:
            raise ValueError(f"target/pose pairing differs: {task}")
        result["target_ranges"][task] = ff["target_range"]
        result["ff"][task] = {method:metrics([r for r in ff["rows"] if r["method"] == method])
                              for method in ("pca", "pls")}
        result["unknown"][task] = dict(
            arms={arm:metrics([r for r in un["rows"] if r["arm"] == arm]) for arm in UNKNOWN},
            probes_attempted=un["probes_attempted"],
            selected_candidate=un["selected_candidate"],
            common_probe_coverage_percent=(100*un["candidate_models"]["common"]["response"]
                                           ["coverage_fraction"]
                                           if un["candidate_models"]["common"]["response"]
                                           is not None else None),
            selected_probe_coverage_percent=(100*un["candidate_models"]
                                             [un["selected_candidate"]]["response"]
                                             ["coverage_fraction"]
                                             if un["selected_candidate"] is not None else None),
            oracle_probe_coverage_percent=(100*un["oracle_response"]["coverage_fraction"]
                                           if un["oracle_response"] is not None else None),
            common_probe_failed=un["candidate_models"]["common"]["response"] is None,
            selected_probe_failed=(un["selected_candidate"] is None or
                                   un["candidate_models"][un["selected_candidate"]]["response"] is None),
            calibration_pose=un["calibration_pose"],test_poses=un["test_poses"])
        result["paired"]["ff"][task] = paired_delta(
            [r for r in ff["rows"] if r["method"] == "pca"],
            [r for r in ff["rows"] if r["method"] == "pls"])
        result["paired"]["unknown"][task] = paired_delta(
            [r for r in un["rows"] if r["arm"] == "common_zero"],
            [r for r in un["rows"] if r["arm"] == "selected_three_probe"])
    for task in FB_TASKS:
        report_path = RESULT / "fb" / task / "report.json"
        result["raw_result_hashes"][str(report_path.relative_to(RESULT)).replace("\\", "/")] = digest(report_path)
        report = read(report_path)
        ff_poses = read(RESULT / "ff" / f"{task}.json")["test_poses"]
        ff_source = read(RESULT / "ff" / f"{task}.json")
        if (report["task"] != task or report["test_poses"] != ff_poses
                or report["protocol"] != "second_axis_feedback_full_rl_nine_v1"
                or report["source_report_sha256"] != ff_source["source_report_sha256"]
                or not report["full_rl_target_grid"] or len(report["targets"]) != 9):
            raise ValueError(f"FB pairing differs: {task}")
        low, high = report["target_range"]
        if any(abs(target-(low+fraction*(high-low))) > 1e-10
               for target,fraction in zip(report["targets"],FRACTIONS)):
            raise ValueError(f"FB target grid differs: {task}")
        by_method = {}
        for method in ("pca", "pls"):
            trials = report["methods"][method]["trials"]
            fb_keys = {(r["pose"], r["target"], r["condition"]) for r in trials}
            if len(fb_keys) != 216 or len(trials) != 216:
                raise ValueError(f"FB pose/goal/condition keys differ: {task}/{method}")
            ff_rows = {(r["pose"], r["target_fraction"]):r for r in ff_source["rows"]
                       if r["method"] == method}
            for row in trials:
                if row["condition"] != "ff":
                    continue
                fraction = (row["target"]-low)/(high-low)
                base = ff_rows[(row["pose"], min(FRACTIONS, key=lambda f:abs(f-fraction)))]
                if (abs(row["target"]-base["target_angle"]) > 1e-10 or
                    (row["abs_error"] is None) != (base["normalized_abs_error"] is None) or
                    (row["abs_error"] is not None and
                     abs(row["abs_error"]/(high-low)-base["normalized_abs_error"]) > 1e-10)):
                    raise ValueError(f"FF/FB baseline differs: {task}/{method}")
            groups = {}
            for condition in ("ff", "axis1_fb", "axis2_fb"):
                rows = [dict(r, span=high-low) for r in trials if r["condition"] == condition]
                groups[condition] = metrics(rows, fb=True)
                groups[condition]["feedback_enabled"] = sum(
                    bool(r["sensitivity_enabled"]) for r in rows) if condition != "ff" else None
                groups[condition]["tipped"] = sum(bool(r["tipped"]) for r in rows)
            by_method[method] = groups
            result["paired"]["fb"][task+"_"+method] = {
                condition: paired_delta(
                    [r for r in trials if r["condition"] == "ff"],
                    [r for r in trials if r["condition"] == condition],
                    fb=True, span=high-low)
                for condition in ("axis1_fb", "axis2_fb")}
        result["fb"][task] = by_method
    return result


def pct(value):
    return "—" if value is None else f"{value:.2f}"


def category_mean(result, stage, category, arm):
    if stage == "ff":
        values = [result["ff"][f"{category}_lite_{i}"][arm]["complete_finite_mae_percent"]
                  for i in range(1,6)]
    else:
        values = [result["unknown"][f"{category}_lite_{i}"]["arms"][arm]
                  ["complete_finite_mae_percent"] for i in range(1,6)]
    if any(v is None for v in values):
        raise ValueError(f"category mean has missing task: {stage}/{category}/{arm}")
    return mean(values)


def build_html(data: dict) -> str:
    plots = []
    labels = [label(task) for task in TASKS]
    ff_plot, spec = figure("ff-matched", labels, [
        ("PCA",[data["ff"][t]["pca"]["complete_finite_mae_percent"] for t in TASKS]),
        ("PLS",[data["ff"][t]["pls"]["complete_finite_mae_percent"] for t in TASKS])],
        "完走有限終端MAE / RL目標幅 [%]",
        "図1. 15タスク・共通8姿勢×9目標。保持失敗も完走・有限なら誤差へ含める。")
    plots.append(spec)
    ff_reach, spec = figure("ff-reached-span", labels, [
        ("PCA",[data["ff"][t]["pca"]["held_reached_span_percent"] for t in TASKS]),
        ("PLS",[data["ff"][t]["pls"]["held_reached_span_percent"] for t in TASKS])],
        "保持できた実測到達幅 / RL目標幅 [%]",
        "図2. 共通9目標の保持終点が実際に張る角度・変位幅。独立スイープCoverageとは異なる。",105)
    plots.append(spec)
    arm_labels = {"common_zero":"共通軸・実測なし", "pca2_nn_zero":"姿勢NN・実測なし",
        "common_three_probe":"共通軸＋3計測", "selected_three_probe":"軸選択＋3～15計測",
        "local_pls_oracle":"自ツールDB参照", "local_pls_three_probe_oracle":"自ツールDB＋3計測"}
    u_plot, spec = figure("unknown-matched", labels, [
        (arm_labels[a],[data["unknown"][t]["arms"][a]["complete_finite_mae_percent"]
                        for t in TASKS]) for a in ("common_zero", "pca2_nn_zero",
                           "common_three_probe", "selected_three_probe",
                           "local_pls_three_probe_oracle")],
        "完走有限終端MAE / RL目標幅 [%]",
        "図3. 未知ツールの同一姿勢・同一9目標。自ツールDB参照は実運用できる未知ツール手法ではない。")
    plots.append(spec)
    u_reach, spec = figure("unknown-reached-span", labels, [
        (arm_labels[a],[data["unknown"][t]["arms"][a]["held_reached_span_percent"]
                        for t in TASKS]) for a in ("common_zero", "selected_three_probe",
                                                 "local_pls_three_probe_oracle")],
        "保持できた実測到達幅 / RL目標幅 [%]",
        "図4. 同じ9目標での実測幅。大きい幅でも追従誤差が小さいとは限らない。",105)
    plots.append(spec)
    fb_labels = [label(t)+"-"+m.upper() for t in FB_TASKS for m in ("pca", "pls")]
    fb_plot, spec = figure("fb-matched", fb_labels, [
        (name,[data["fb"][t][m][condition]["complete_finite_mae_percent"]
               for t in FB_TASKS for m in ("pca", "pls")])
        for name,condition in (("FF＋保持","ff"),("第1軸FB","axis1_fb"),
                               ("第2軸FB","axis2_fb"))],
        "完走有限終端MAE / RL目標幅 [%]",
        "図6. 4タスクのみ。各目標の感度は8較正姿勢で別途計測した。")
    plots.append(spec)
    clip_plot, spec = figure("unknown-clipping",labels,[
        (arm_labels[a],[100*data["unknown"][t]["arms"][a]["inverse_clipped"]/72
                        for t in TASKS]) for a in ("common_zero","selected_three_probe")],
        "逆写像の端点制限率 [%]",
        "図5. 較正・DBが示す操作域外の目標は端点へ制限した。低MAEだけでなくこの率を確認する。",105)
    plots.append(spec)
    ff_table = table(["ツール", "PCA MAE [%]", "PLS MAE [%]", "PCA完走", "PLS完走",
                      "PCA保持", "PLS保持", "PCA到達幅 [%]", "PLS到達幅 [%]",
                      "PCA端点制限", "PLS端点制限", "PCA関節制限", "PLS関節制限"], [
        [t, pct(data["ff"][t]["pca"]["complete_finite_mae_percent"]),
         pct(data["ff"][t]["pls"]["complete_finite_mae_percent"]),
         f'{data["ff"][t]["pca"]["complete_finite"]}/72',
         f'{data["ff"][t]["pls"]["complete_finite"]}/72',
         f'{data["ff"][t]["pca"]["held"]}/72', f'{data["ff"][t]["pls"]["held"]}/72',
         pct(data["ff"][t]["pca"]["held_reached_span_percent"]),
         pct(data["ff"][t]["pls"]["held_reached_span_percent"]),
         f'{data["ff"][t]["pca"]["inverse_clipped"]}/72',
         f'{data["ff"][t]["pls"]["inverse_clipped"]}/72',
         f'{data["ff"][t]["pca"]["joint_clipped"]}/72',
         f'{data["ff"][t]["pls"]["joint_clipped"]}/72'] for t in TASKS],
        name="全15ツールのPCA/PLS共通目標結果")
    u_table = table(["ツール", "共通軸 [%]", "姿勢NN [%]", "共通軸＋3計測 [%]",
        "選択軸＋計測 [%]", "自ツールDB [%]", "自ツールDB＋3計測 [%]", "試行プローブ数",
        "共通軸較正幅 [%]", "採用軸較正幅 [%]", "共通較正失敗", "採用較正失敗",
        "選択軸完走", "選択軸保持", "選択軸到達幅 [%]", "選択軸端点制限",
        "選択軸関節制限"], [
        [t,*[pct(data["unknown"][t]["arms"][a]["complete_finite_mae_percent"])
              for a in UNKNOWN], data["unknown"][t]["probes_attempted"],
         pct(data["unknown"][t]["common_probe_coverage_percent"]),
         pct(data["unknown"][t]["selected_probe_coverage_percent"]),
         "有" if data["unknown"][t]["common_probe_failed"] else "無",
         "有" if data["unknown"][t]["selected_probe_failed"] else "無",
         f'{data["unknown"][t]["arms"]["selected_three_probe"]["complete_finite"]}/72',
         f'{data["unknown"][t]["arms"]["selected_three_probe"]["held"]}/72',
         pct(data["unknown"][t]["arms"]["selected_three_probe"]["held_reached_span_percent"]),
         f'{data["unknown"][t]["arms"]["selected_three_probe"]["inverse_clipped"]}/72',
         f'{data["unknown"][t]["arms"]["selected_three_probe"]["joint_clipped"]}/72']
        for t in TASKS], name="全15未知ツールの共通目標結果")
    fb_table = table(["タスク・方式", "FF [%]", "第1軸FB [%]", "第2軸FB [%]",
                      "FF完走", "第1軸FB完走", "第2軸FB完走",
                      "FF保持", "第1軸FB保持", "第2軸FB保持",
                      "第1軸較正成立", "第2軸較正成立", "FF転倒診断", "第2軸転倒診断"], [
        [t+" / "+m.upper(), *[pct(data["fb"][t][m][c]["complete_finite_mae_percent"])
                                for c in ("ff","axis1_fb","axis2_fb")],
         *[f'{data["fb"][t][m][c]["complete_finite"]}/72'
           for c in ("ff","axis1_fb","axis2_fb")],
         *[f'{data["fb"][t][m][c]["held"]}/72' for c in ("ff","axis1_fb","axis2_fb")],
         f'{data["fb"][t][m]["axis1_fb"]["feedback_enabled"]}/72',
         f'{data["fb"][t][m]["axis2_fb"]["feedback_enabled"]}/72',
         f'{data["fb"][t][m]["ff"]["tipped"]}/72',
         f'{data["fb"][t][m]["axis2_fb"]["tipped"]}/72']
        for t in FB_TASKS for m in ("pca","pls")], name="FB共通目標結果")
    cat_table = table(["カテゴリ", "PCA FF [%]", "PLS FF [%]", "未知共通軸 [%]",
                       "未知共通軸＋3計測 [%]", "未知選択軸＋計測 [%]",
                       "自ツールDB＋3計測 [%]"], [
        [category.title(), pct(category_mean(data,"ff",category,"pca")),
         pct(category_mean(data,"ff",category,"pls")),
         pct(category_mean(data,"unknown",category,"common_zero")),
         pct(category_mean(data,"unknown",category,"common_three_probe")),
         pct(category_mean(data,"unknown",category,"selected_three_probe")),
         pct(category_mean(data,"unknown",category,"local_pls_three_probe_oracle"))]
        for category in ("scissors","push","spray")], name="カテゴリ平均")
    range_table = table(["ツール", "RL目標下限", "RL目標上限", "目標幅の1%"], [
        [t, f'{data["target_ranges"][t][0]*scale:.3f} {unit}',
         f'{data["target_ranges"][t][1]*scale:.3f} {unit}',
         f'{(data["target_ranges"][t][1]-data["target_ranges"][t][0])*scale/100:.3f} {unit}']
        for t in TASKS for scale,unit in [(1000.,"mm") if t.startswith("push") else (1.,"rad")]],
        name="各ツールの物理的なRL目標幅")
    ff_pair_table = table(["ツール", "対応完走/72", "PLS − PCA MAE差 [%]",
                           "PLS低誤差の対応数"], [
        [t, data["paired"]["ff"][t]["paired_complete_finite"],
         pct(data["paired"]["ff"][t]["second_minus_first_mae_percent"]),
         data["paired"]["ff"][t]["second_better_pairs"]] for t in TASKS],
        name="同一姿勢・同一目標におけるPCA対PLS")
    unknown_pair_table = table(["ツール", "対応完走/72",
                                "選択＋計測 − 共通軸MAE差 [%]", "選択＋計測が低誤差の対応数"], [
        [t, data["paired"]["unknown"][t]["paired_complete_finite"],
         pct(data["paired"]["unknown"][t]["second_minus_first_mae_percent"]),
         data["paired"]["unknown"][t]["second_better_pairs"]] for t in TASKS],
        name="同一姿勢・同一目標における未知ツール適応の差")
    fb_pair_table = table(["タスク・方式", "第1軸FB−FF [%]", "第1軸対応完走",
                           "第2軸FB−FF [%]", "第2軸対応完走"], [
        [t+" / "+m.upper(),
         pct(data["paired"]["fb"][t+"_"+m]["axis1_fb"]["second_minus_first_mae_percent"]),
         data["paired"]["fb"][t+"_"+m]["axis1_fb"]["paired_complete_finite"],
         pct(data["paired"]["fb"][t+"_"+m]["axis2_fb"]["second_minus_first_mae_percent"]),
         data["paired"]["fb"][t+"_"+m]["axis2_fb"]["paired_complete_finite"]]
        for t in FB_TASKS for m in ("pca","pls")], name="同一姿勢・同一目標のFB差")
    clip_strata = table(["ツール", "共通軸:非制限MAE [%]", "共通軸:制限MAE [%]",
                         "選択軸:非制限MAE [%]", "選択軸:制限MAE [%]",
                         "選択軸:制限完走/72"], [
        [t, *[pct(data["unknown"][t]["arms"][a][key])
              for a in ("common_zero", "selected_three_probe")
              for key in ("unclipped_mae_percent", "clipped_mae_percent")],
         f'{data["unknown"][t]["arms"]["selected_three_probe"]["clipped_complete_finite"]}/72']
        for t in TASKS], name="逆写像端点制限の有無による誤差")
    improved_tools = [t for t in TASKS if data["unknown"][t]["arms"]
                      ["selected_three_probe"]["complete_finite_mae_percent"] <
                      data["unknown"][t]["arms"]["common_zero"]
                      ["complete_finite_mae_percent"]]
    full_probe_tools = [t for t in TASKS if data["unknown"][t]["probes_attempted"] == 15]
    spray_midpoint_or_worse = [t for t in TASKS if t.startswith("spray") and
        data["unknown"][t]["arms"]["selected_three_probe"]
        ["complete_finite_mae_percent"] >= 100*2.5/9]
    missing_ff = sum(72-v["complete_finite"] for task in data["ff"].values()
                     for v in task.values())
    missing_unknown = sum(72-v["complete_finite"] for task in data["unknown"].values()
                          for v in task["arms"].values())
    missing_fb = sum(72-v["complete_finite"] for task in data["fb"].values()
                     for method in task.values() for v in method.values())
    main_results = (
        "<section><h2>数値結果の読み取り</h2>"
        f"<p>共通9目標でのFFカテゴリ平均MAEは、ScissorsでPCA {pct(category_mean(data,'ff','scissors','pca'))}%・PLS {pct(category_mean(data,'ff','scissors','pls'))}%、"
        f"PushでPCA {pct(category_mean(data,'ff','push','pca'))}%・PLS {pct(category_mean(data,'ff','push','pls'))}%、"
        f"SprayでPCA {pct(category_mean(data,'ff','spray','pca'))}%・PLS {pct(category_mean(data,'ff','spray','pls'))}%だった。"
        "これは同じ目標集合での比較だが、各方式の較正写像そのものは従来の保存値である。</p>"
        f"<p>未知ツールでは共通軸のカテゴリ平均がScissors {pct(category_mean(data,'unknown','scissors','common_zero'))}%、"
        f"Push {pct(category_mean(data,'unknown','push','common_zero'))}%、Spray {pct(category_mean(data,'unknown','spray','common_zero'))}%であり、"
        f"3～15回の計測後に軸を選ぶ条件はそれぞれ{pct(category_mean(data,'unknown','scissors','selected_three_probe'))}%、"
        f"{pct(category_mean(data,'unknown','push','selected_three_probe'))}%、"
        f"{pct(category_mean(data,'unknown','spray','selected_three_probe'))}%だった。"
        f"単純平均の大小では{len(improved_tools)}/15ツールで低誤差だったが、これは有意差ではなく、"
        f"{len(full_probe_tools)}/15ツールは15回のプローブを要した。"
        f"Sprayでは{len(spray_midpoint_or_worse)}/5ツールが中央固定の数式上の参照誤差27.78%以上であり、"
        "端点制限率も高い。カテゴリ平均だけで未知Sprayへの汎化が成立したとは言えない。</p>"
        f"<p>完走・有限終端を得られなかった件数はFF表で{missing_ff}、未知ツール表で{missing_unknown}、"
        f"FB表で{missing_fb}だった。FB表のFF条件はFF表と同じ試行なので合算しない。これらを誤差0として平均に入れていない。"
        "保持率と転倒診断を別表で示し、MAEの低下だけから実機安全性を判断しない。"
        "Scissorsのカテゴリ平均改善にはScissors 3の大きな変化が強く寄与する。</p></section>")
    sections = [
        "<header><h1>共通9目標によるTDS再評価</h1><p>15ツールのPCA/PLS 1自由度FF、未知ツール適応、4タスクのFBを同じ目標角度割合で評価した。旧条件の数値とは別の新規シミュレーションである。</p></header>",
        "<section><h2>先に結論を読む前の注意</h2><p>全ツールでRL目標範囲の0、12.5、25、37.5、50、62.5、75、87.5、100%を目標とし、各ツールの同じ8初期姿勢で72試行/方式を実施した。5秒で姿勢を移動し2秒保持する。途中終了や数値異常を誤差0として扱わず、完走・有限試行のMAE、保持件数、未観測件数を分けて示す。目標域外では較正済み逆写像の端点へ指令を制限し、その頻度を隠さない。</p><p>この結果は既存15ツールと既存姿勢プールの再評価であり、独立な新形状・新姿勢での確認実験ではない。比較表のMAEが小さくても保持率・端点制限率が悪ければ実用性能が高いとは結論しない。</p></section>",
        "<section><h2>用語と測定単位</h2><p>TDSはTool-DoF Synergy（ツール操作自由度シナジー）の略で、多数のハンド関節を一つの得点で動かす方向をいう。RLは強化学習、DBはデータベース、PCAは主成分分析、PLSは部分最小二乗法、NNは最近傍法である。FFはツール状態の誤差を測って指令を修正しない制御、FBは移動後の誤差で指令を修正する制御。Few-shotの『プローブ』は一つのシナジー得点を試して実際の動きを測る1試行を指す。</p><p>PCA/PLSの軸は成功した最終ハンド姿勢200件/ツールから求める。初期姿勢は別データで、各ツールに32件あり、そのうち主実験の較正8件・評価8件を使った。MAEは目標と実測のずれの平均絶対値を、各ツールのRL目標範囲幅で割って百分率にした値。Pushのツール自由度は変位[m]、Scissors/Sprayは角度[rad]であり、図ではすべて同じ正規化百分率へ変換する。『保持』は各タスクの既定の保持条件を満たした完走試行であり、誤差の小ささとは別の指標である。Push/Sprayの保持判定は接触数ではなく落下・転倒の有無に基づく。『到達幅』は保持できた試行の実測最大値から最小値を引き、RL目標範囲内に制限してから幅で割った値であり、全区間の到達保証ではない。</p><details><summary>15ツールの目標範囲と物理単位</summary>"+range_table+"</details></section>",
        "<section><h2>手続きと情報境界</h2><p>FFのPCA/PLS軸と等張回帰の逆写像は、保存済み成功姿勢DBと従来の8較正姿勢から得た値を凍結し、再学習・再フィットしない。未知ツールでは同一カテゴリの他4ツール（計800成功姿勢）から共通軸と候補軸を作る。実測なしの条件は未知ツールの操作結果を使わない。3点計測条件はFF較正姿勢8件のうち1件で共通軸を試し、応答幅が75%未満なら他4軸も各3回試す。未知ツール自身の200成功姿勢を使う『自ツールDB参照』は情報量の異なる比較対象で、ゼロショットの上限保証ではない。</p><p>FBの4タスクはScissors 3/4、Push 3、Spray 3。FF・第1軸FB・第2軸FBで同じ8評価姿勢と9目標を共有する。ただし各目標で8較正姿勢×3得点×2補正軸×2手法の感度試験を別途行う。これは単純なゼロショット制御ではなく、較正コストの大きい探索実験である。75%停止則と4タスクの選択は過去の探索後に固定したため、事前登録済みの独立確認とみなさない。</p></section>",
        "<section><h2>主要結果：カテゴリ平均</h2><p>平均はカテゴリ内5ツールの等重み平均である。各ツールの試行を独立な15×72件としてp値は計算しない。目標を問わず中央値だけを返す数式上の参照誤差は27.78%だが、これは物理的に把持・操作した対照実験ではない。</p>"+cat_table+"</section>",
        main_results,
        "<section><h2>PCAとPLSの1自由度FF</h2>"+ff_plot+ff_reach+"<p>差分表は同じ姿勢・目標で両方式が完走した組だけを引き算する。負ならPLSの誤差が小さい。ただし対応完走数が72未満なら、除かれた試行と保持件数も表で確認する。</p><details><summary>全15ツールの誤差・保持・制限件数</summary>"+ff_table+"</details><details><summary>対応した試行だけのPCA対PLS差</summary>"+ff_pair_table+"</details></section>",
        "<section><h2>未知ツールへの転用と少数計測</h2><p>図3は同じ目標・初期姿勢での結果。共通軸をそのまま使う場合と、その軸を3回実測して補正する場合は、軸選択の効果と較正の効果を分けて読む。未知ツールのDBを使う二条件は研究上の参照であり、Few-shotに使える情報とは異なる。表中の『—』は較正応答が成立しなかったことを表し、実測幅0%ではない。</p>"+u_plot+u_reach+clip_plot+"<details><summary>全15ツールの誤差とプローブ費用</summary>"+u_table+"</details><details><summary>対応した試行だけの適応効果</summary>"+unknown_pair_table+"</details></section>",
        "<section><h2>4タスクのFF・FB比較</h2><p>第2軸はPCAではPC2、PLSではPLS第2潜在成分ではなく第1操作軸を除いた姿勢残差のPC1である。感度が無効な試行も条件の分母に残す。終端のPalm球判定と転倒診断を同じ安全性指標として扱わない。</p>"+fb_plot+"<details><summary>全4タスク×2手法の数値</summary>"+fb_table+"</details><details><summary>同じ姿勢・目標のFF比誤差差</summary>"+fb_pair_table+"</details></section>",
        "<section><h2>限界と研究倫理上の位置付け</h2><p>ScissorsのTDS評価球は学習・成功姿勢DBより大きく、Push/Sprayでは接触数ゲートを用いない。これらは事後に変更された採用基準であり、成功率を一種類の値として統合しない。Sprayの床接触は有効である。今回の共通9目標は評価目標を揃えるが、各方式の較正情報量までは等しくしない。得られた差を新規の未知形状への統計的有意差や実機安全性の証拠とは主張しない。</p><p>Claudeによる外部監査の指摘、採否、修正内容は別途記録する。元論文PDFは外部送信していない。</p></section>",
    ]
    sections.insert(3, "<section><h2>実験で使う判定を混同しないために</h2>"
        "<p>既存15ツールのうち1ツールを評価時だけ除外する『未知ツール』設定であり、新しい形状を追加した実験ではない。"
        "プローブ採用には2秒保持の終点が条件を満たし、途中終了・数値異常がないことを求める。"
        "本評価の『保持』は2秒の保持区間を通じて条件を満たした試行である。Scissorsでは関節中心がPalmに追従する球に入る基準、"
        "Push/Sprayでは落下・転倒を除く基準を使う。ただし転倒診断とScissorsのPalm球保持は同義ではなく、"
        "Scissors 3/4では終端保持が多くても軌道中の転倒診断が多い。FF・未知ツールの生データは転倒診断を保存していないため、"
        "それらの保持率を安全性指標とは呼べない。</p>"
        "<p>『端点制限』は較正済み逆写像の得点域から外れた目標を端に丸めた回数、"
        "『関節制限』は再構成姿勢が駆動関節の可動域に当たった回数である。"
        "非較正の線形逆写像では端点制限が少なくても操作できているとは限らない。"
        "『到達幅』は全8初期姿勢をまとめた保持終点の最大−最小で、1つの外れ値にも敏感である。"
        "『—』は値が定義できないことを示し、応答幅0%とは区別する。</p></section>")
    sections.insert(-1, "<section><h2>端点制限で分けた誤差</h2><p>同じ72試行を、"
        "実行時に逆写像の端点へ丸めた組と丸めない組に分ける。群の目標値分布は異なるため、"
        "この2列の大小を手法効果とは解釈しない。実測なしの線形逆写像で端点制限0でも、"
        "到達幅がほぼ0の例がある。</p>"+clip_strata+"</section>")
    sections.insert(-1, "<section><h2>実験履歴と監査範囲</h2>"
        "<p>最初の共通9目標キャンペーンは、FFの逆写像域外での指令端点制限が適用されていない実装を実行中に発見して中断した。"
        "その部分結果は本集計に一切含めず、修正後に別ディレクトリで全34ジョブを最初から実行した。"
        "したがって旧部分結果との性能差は主張しない。コード・生JSONのSHA-256はsummary.jsonに記録した。</p>"
        "<p>Claude Opus 5.5/High Effortには指定したコード・集計JSON・状態JSONだけを読み取り専用で確認させた。"
        "較正不成立時の代替、転倒診断との不一致、評価姿勢の再利用、事後設定の75%閾値、"
        "端点・関節制限の区別を指摘され、採否と対応を"
        "<a href='../../docs/claude_external_audit_20261008.md'>監査記録</a>へ記した。"
        "生の試行JSONとHTMLの表示はClaudeの今回の監査対象外であり、不正がないことの証明でもない。</p></section>")
    sections = [s.replace("同じ目標角度割合", "同じツール状態目標割合")
                .replace("共通の9角度", "共通の9ツール状態値")
                .replace("共通9目標での結果", "共通9目標での結果")
                .replace("3点計測条件はFF較正姿勢8件のうち1件で共通軸を試し、",
                         "3点計測条件はFF較正姿勢8件のうち1件で共通軸を試し、")
                for s in sections]
    sections = [s.replace("表中の『—』は較正応答が成立しなかったことを表し、実測幅0%ではない。",
        "表中の『—』は数値が定義できないことを表し、実測幅0%ではない。"
        "共通軸の3点較正が不成立だったSpray 1では、当該条件の評価指令は実測なしの線形逆写像へ戻る。"
        "この条件を『3点較正が成功した』とは解釈しない。採用軸の較正は全15ツールで成立した。"
        "試行プローブ数は成功数ではなく、別途表示する自ツールDBの3点計測を含まない。")
        for s in sections]
    css = """<style>body{margin:0;background:#f5f5f3;color:#111;font:16px/1.7 'Times New Roman','Yu Mincho',serif}header,section{max-width:1200px;margin:0 auto;background:#fff;padding:26px 32px}header{padding-top:50px;border-bottom:3px solid #111}section{margin-top:16px}h1{font-size:34px}h2{font-size:25px;border-bottom:1px solid #999;padding-bottom:7px}p{max-width:1080px}figure{margin:20px 0;border-top:1px solid #ddd;border-bottom:1px solid #ddd;padding:10px 0}.graph{min-height:340px}.graph svg{display:block;width:100%;height:auto}.legend{display:flex;gap:12px;flex-wrap:wrap;font-size:14px}.swatch{display:inline-block;width:11px;height:11px;border:2px solid var(--series);margin-right:4px}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:6px 9px;border-bottom:1px solid #bbb;text-align:left;white-space:nowrap}th{border-top:2px solid #111;background:#f7f7f7}details{margin:20px 0;padding:8px 14px;border:1px solid #bbb}summary{cursor:pointer}figcaption{font-size:14px;margin-top:8px}@media(max-width:700px){header,section{padding:18px 14px}.graph{min-height:220px}}</style>"""
    js = """<script src='https://cdn.plot.ly/plotly-4.1.1.min.js'></script><script>
const plots=__PLOTS__;if(window.Plotly){for(const p of plots){let traces=p.series.map((s,j)=>({type:'scatter',mode:'markers',name:s.name,x:p.labels.map((_,i)=>i+(j-(p.series.length-1)/2)*.13),y:s.values,marker:{size:9,color:s.color,symbol:s.symbol},hovertemplate:'%{y:.2f}<extra>'+s.name+'</extra>'}));let axis={ticks:'inside',ticklen:6,minor:{ticks:'inside',ticklen:3,showgrid:false},showline:true,linecolor:'#111',showgrid:false,zeroline:false,automargin:true};Plotly.newPlot(p.id,traces,{template:'plotly_white',margin:{l:3,r:3,t:2,b:3},font:{family:'Times New Roman',size:15},height:360,showlegend:false,xaxis:{...axis,tickmode:'array',tickvals:p.labels.map((_,i)=>i),ticktext:p.labels,tickangle:-40},yaxis:{...axis,title:{text:p.ylabel},range:[0,p.maximum]}},{responsive:true,displayModeBar:false});}}
</script>""".replace("__PLOTS__",json.dumps(plots,ensure_ascii=False,separators=(",",":")))
    return "<!doctype html><html lang='ja'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>共通9目標TDS再評価</title>"+css+"</head><body>"+"\n".join(sections)+"<footer style='max-width:1200px;margin:0 auto;padding:25px;background:white'>機械可読集計と生結果のSHA-256: <a href='summary.json'>summary.json</a>。34ジョブの実行状態: <a href='status.json'>status.json</a>。生の試行JSONはローカルの ff・generalization・fb 各ディレクトリに保存。</footer>"+js+"</body></html>"


def main() -> None:
    data = load()
    (RESULT / "summary.json").write_text(json.dumps(data,ensure_ascii=False,indent=2),
                                         encoding="utf-8")
    (RESULT / "index.html").write_text(build_html(data),encoding="utf-8")
    print(RESULT / "index.html")


if __name__ == "__main__":
    main()
