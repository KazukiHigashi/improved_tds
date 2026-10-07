"""Self-contained Japanese HTML for PCA2-nearest operation-axis transfer."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from evaluate_initial_pca_nn import (CATEGORIES, METHODS, coordinates,
                                     fit_pca, read_category, select_nearest,
                                     unique_initials)
from simulate_initial_pca_nn import ARMS


NAMES = {
    "common_axis":"4ツール共通・1軸",
    "raw13_nn_package":"初期13関節NN・軸＋較正",
    "pca2_nn_package":"初期PCA2 NN・軸＋較正",
    "pca2_selected_axis_only":"初期PCA2 NN・軸のみ",
    "task_specific_reference":"対象ツール専用・参照",
}
COLORS = dict(zip(ARMS,("#555555","#d55e00","#0072b2","#cc79a7","#009e73")))
TOOL_COLORS = ("#0072b2","#e69f00","#009e73","#d55e00","#cc79a7")


def table(headers, rows):
    h="".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b="".join("<tr>"+"".join(f"<td>{html.escape(str(x))}</td>" for x in row)+"</tr>"
              for row in rows)
    return f"<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>"


def mean(rows, key):
    return float(np.mean([r[key] for r in rows]))


def summarize(root):
    offline=json.loads((root/"offline.json").read_text(encoding="utf-8"))
    output=dict(protocol="initial_pca2_nn_physical_v1",offline_protocol=offline["protocol"],
                source=offline["source_bundle"],new_training=False,
                pca_rule="fit PCA on unique initial postures of four known tools only",
                trajectory_seconds=5,hold_seconds=2,reset_poses_per_tool=3,
                target_fractions=[.1,.3,.5,.7,.9],categories={})
    for category in CATEGORIES:
        rows=[]
        for variant in range(1,6):
            path=root/f"{category}_lite_{variant}_physical.json"
            values=json.loads(path.read_text(encoding="utf-8"))
            if len(values)!=3*5*len(ARMS):
                raise ValueError(f"{path}: incomplete experiment")
            rows.extend(values)
        by_arm={}
        by_tool={}
        for arm in ARMS:
            chosen=[r for r in rows if r["arm"]==arm]
            by_arm[arm]=dict(mae_fraction=mean(chosen,"normalized_abs_error"),
                             held_fraction=mean(chosen,"supported"),
                             joint_clip_fraction=mean(chosen,"joint_clipped"),
                             inverse_clip_fraction=mean(chosen,"inverse_clipped"),
                             mean_axis_cosine=mean(chosen,"selected_axis_cosine_to_target"))
        for variant in range(1,6):
            by_tool[str(variant)]={arm:dict(
                mae_fraction=mean([r for r in rows if r["task"]==f"{category}_lite_{variant}"
                                   and r["arm"]==arm],"normalized_abs_error"),
                held_fraction=mean([r for r in rows if r["task"]==f"{category}_lite_{variant}"
                                    and r["arm"]==arm],"supported")) for arm in ARMS}
        rng=np.random.default_rng(68180+len(category))
        ci={}
        for comparator in ("raw13_nn_package","common_axis"):
            d=np.array([by_tool[str(i)]["pca2_nn_package"]["mae_fraction"]
                        -by_tool[str(i)][comparator]["mae_fraction"] for i in range(1,6)])
            sampled=d[rng.integers(0,5,(20000,5))].mean(axis=1)
            ci[comparator]=dict(mean_difference=float(d.mean()),
                                bootstrap95=np.quantile(sampled,[.025,.975]).tolist(),
                                improved_tool_count=int(np.sum(d<0)))
        axis_only_delta=np.array([
            by_tool[str(i)]["pca2_selected_axis_only"]["mae_fraction"]
            -by_tool[str(i)]["common_axis"]["mae_fraction"] for i in range(1,6)])
        axis_only_sampled=axis_only_delta[rng.integers(0,5,(20000,5))].mean(axis=1)
        axis_only_ci=dict(mean_difference=float(axis_only_delta.mean()),
                          bootstrap95=np.quantile(axis_only_sampled,[.025,.975]).tolist(),
                          improved_tool_count=int(np.sum(axis_only_delta<0)))
        output["categories"][category]=dict(offline=offline["categories"][category],
             physical=rows,by_arm=by_arm,by_tool=by_tool,pca2_vs=ci,
             pca2_axis_only_vs_common=axis_only_ci)
        print(category,"MAE",{a:round(by_arm[a]["mae_fraction"],3) for a in ARMS},
              "PCA2 vs raw",ci["raw13_nn_package"],flush=True)
    (root/"summary.json").write_text(json.dumps(output,ensure_ascii=False,indent=2),
                                     encoding="utf-8")
    (root/"index.html").write_text(render(output,root),encoding="utf-8")
    return output


def mae_graph(category,item):
    x=lambda i:85+120*(i-1)
    y=lambda value:240-180*(100*value/65.)
    out=[f"<svg role='img' aria-label='{category} taskwise MAE' viewBox='0 0 670 310'>",
         "<line x1='85' y1='60' x2='85' y2='240' stroke='#222'/>",
         "<line x1='85' y1='240' x2='565' y2='240' stroke='#222'/>",
         "<text x='7' y='25' font-size='13'>MAE [% of target range]</text>",
         "<text x='275' y='290' font-size='13'>Tool variant</text>"]
    for tick in (0,20,40,60):
        yy=y(tick/100)
        out.append(f"<line x1='80' y1='{yy:.1f}' x2='565' y2='{yy:.1f}' stroke='#ddd'/>")
        out.append(f"<text x='42' y='{yy+4:.1f}' font-size='12'>{tick}</text>")
    for i in range(1,6):
        out.append(f"<text x='{x(i)-4}' y='258' font-size='12'>{i}</text>")
    for arm in ARMS:
        values=[item["by_tool"][str(i)][arm]["mae_fraction"] for i in range(1,6)]
        color=COLORS[arm]
        points=" ".join(f"{x(i)},{y(v):.1f}" for i,v in enumerate(values,1))
        out.append(f"<polyline points='{points}' stroke='{color}' stroke-width='2.5' fill='none'/>")
        out.extend(f"<circle cx='{x(i)}' cy='{y(v):.1f}' r='3.5' fill='{color}'/>"
                   for i,v in enumerate(values,1))
    return "".join(out)+"</svg>"


def response_graph(rows,category,variant):
    task=f"{category}_lite_{variant}"
    chosen=[r for r in rows if r["task"]==task]
    target_fractions=sorted({r["target_fraction"] for r in chosen})
    coef=np.polyfit([r["target_fraction"] for r in chosen if r["arm"]=="common_axis"],
                    [r["target_angle"] for r in chosen if r["arm"]=="common_axis"],1)
    x=lambda f:70+250*f
    y=lambda f:270-240*f
    out=[f"<svg role='img' aria-label='{task} target response' viewBox='0 0 410 350'>",
         "<line x1='70' y1='30' x2='70' y2='270' stroke='#222'/>",
         "<line x1='70' y1='270' x2='320' y2='270' stroke='#222'/>",
         "<line x1='70' y1='270' x2='320' y2='30' stroke='#aaa' stroke-dasharray='4 3'/>",
         "<text x='120' y='318' font-size='12'>目標角度 [%]</text>",
         "<text x='5' y='20' font-size='12'>実角度 [%]</text>"]
    for tick in (0,.25,.5,.75,1.):
        out.append(f"<text x='{x(tick)-7:.1f}' y='288' font-size='10'>{tick*100:.0f}</text>")
        out.append(f"<text x='36' y='{y(tick)+3:.1f}' font-size='10'>{tick*100:.0f}</text>")
    for arm in ARMS:
        points=[]
        for f in target_fractions:
            subset=[r for r in chosen if r["arm"]==arm
                    and np.isclose(r["target_fraction"],f)]
            achieved=float(np.mean([r["achieved_angle"] for r in subset]))
            normalized=float(np.clip((achieved-coef[1])/coef[0],-.1,1.1))
            points.append((x(f),y(normalized)))
        path=" ".join(f"{xx:.1f},{yy:.1f}" for xx,yy in points)
        out.append(f"<polyline points='{path}' stroke='{COLORS[arm]}' stroke-width='2' fill='none'/>")
        out.extend(f"<circle cx='{xx:.1f}' cy='{yy:.1f}' r='3' fill='{COLORS[arm]}'/>"
                   for xx,yy in points)
    return "".join(out)+"</svg>"


def pca_scatter(bundle,category,heldout):
    blocks=read_category(bundle,category)
    target=blocks[heldout-1]
    sources=[b for i,b in enumerate(blocks) if i!=heldout-1]
    train,labels,_=unique_initials(sources)
    test,_,test_poses=unique_initials([target])
    model=fit_pca(train)
    ztrain=coordinates(train,model,"pca2")
    ztest=coordinates(test,model,"pca2")
    mins=np.minimum(ztrain.min(0),ztest.min(0))
    maxs=np.maximum(ztrain.max(0),ztest.max(0))
    span=np.maximum(maxs-mins,1e-9)
    x=lambda value:55+450*(value-mins[0])/span[0]
    y=lambda value:245-200*(value-mins[1])/span[1]
    out=[f"<svg role='img' aria-label='{category} heldout {heldout} PCA2 initial poses' "
         "viewBox='0 0 570 310'>",
         "<line x1='55' y1='45' x2='55' y2='245' stroke='#222'/>",
         "<line x1='55' y1='245' x2='505' y2='245' stroke='#222'/>",
         "<text x='260' y='290' font-size='12'>Initial-pose PC1 [rad]</text>",
         "<text x='5' y='20' font-size='12'>PC2 [rad]</text>"]
    for i,b in enumerate(blocks):
        if i==heldout-1:continue
        color=TOOL_COLORS[i]
        out.extend(f"<circle cx='{x(p[0]):.1f}' cy='{y(p[1]):.1f}' r='2.6' "
                   f"fill='{color}' opacity='.68'/>" for p,label in zip(ztrain,labels)
                   if label==b["name"])
    for p in ztest:
        out.append(f"<circle cx='{x(p[0]):.1f}' cy='{y(p[1]):.1f}' r='3.5' "
                   "fill='#111' opacity='.8'/>")
    chosen_poses=np.random.default_rng(59320+heldout).choice(32,size=3,replace=False)
    for pose in chosen_poses:
        loc=np.flatnonzero(test_poses==pose)
        if not len(loc):continue
        p=ztest[int(loc[0])]
        neighbour,_=select_nearest(test[int(loc[0])][None],train,model,"pca2")
        q=ztrain[int(neighbour[0])]
        out.append(f"<line x1='{x(p[0]):.1f}' y1='{y(p[1]):.1f}' "
                   f"x2='{x(q[0]):.1f}' y2='{y(q[1]):.1f}' stroke='#111' "
                   "stroke-width='1.2' stroke-dasharray='3 2'/>")
        out.append(f"<circle cx='{x(p[0]):.1f}' cy='{y(p[1]):.1f}' r='5' "
                   "fill='none' stroke='#111' stroke-width='1.5'/>")
    out.append("<text x='55' y='267' font-size='11'>黒：未知ツール、線：評価初期姿勢→選択された近傍</text>")
    return "".join(out)+"</svg>"


def render(report,root):
    bundle=Path(report["source"])
    out=["<!doctype html><html lang='ja'><meta charset='utf-8'>",
         "<title>初期姿勢PCA2 NNによるTDS軸転移</title>",
         "<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1120px;margin:2em auto;padding:0 1em;color:#17222b}table{border-collapse:collapse;width:100%;margin:1em 0}td,th{border:1px solid #c7d0d8;padding:.38em;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#edf2f7}svg{max-width:100%;height:auto;border:1px solid #dfe6eb;background:#fff}details{padding:.7em;margin:1em 0;border:1px solid #c7d0d8;border-radius:.3em}code{background:#f1f4f6;padding:.1em .25em}</style>",
         "<h1>初期姿勢の2D PCA空間を用いた最近傍TDS軸選択</h1>",
         "<p>各カテゴリ5ツール×200成功姿勢。再学習なし。各成功姿勢に保存された初期13関節を、"
         "同じ初期姿勢の重複を除いてPCA。未知ツールを1つずつ除き、既知4ツールの初期姿勢だけで"
         "PCA2と最近傍辞書を構成しました。未知ツールの初期姿勢を投影し、最近傍に紐づく"
         "既知ツールの角度監督PLS第1軸を選びます。数値のツールIDは入力していません。</p>",
         "<h2>妥当性と実装上の懸念</h2>",
         "<p><code>z=(q初期−μ初期)W2</code> を2D座標とし、"
         "<code>j*=argmin_j ||z−z_j||₂</code> で既知初期姿勢を選びます。"
         "その姿勢に対応するツールの操作軸 <code>v1,j*</code> を使います。"
         "1初期姿勢あたりの成功姿勢は少ないため、初期姿勢ごとにPLS軸を再学習せず、"
         "既知ツールの200成功姿勢から得た軸を対応付けました。"
         "PCA2が大きな初期姿勢分散を保持しても、ツール操作軸に重要な小分散方向を捨てる可能性があります。"
         "関節ごとの可動域や設計時のばらつきが異なるため、関節角をそのままPCAする尺度の選択にも注意が必要です。"
         "また、初期手姿勢が似てもツールの形状・接触・角度較正が同じとは限りません。</p>",
         "<h2>オフライン幾何評価</h2><p>既知ツール内で未使用初期姿勢を25%除外した10分割で、"
         "最近傍が同一ツール由来だった割合を示します。ラベルは評価にだけ使用。"
         "未知ツールの軸余弦は、対象ツールのDBを後から用いた診断値で実行時には使いません。</p>"]
    geometry=[]
    for category in CATEGORIES:
        item=report["categories"][category]["offline"]
        geometry.append([category.title(),f"{100*item['mean_cumulative_pca2_variance']:.1f}",
            f"{100*item['mean_known_label_accuracy']['raw13']:.1f}",
            f"{100*item['mean_known_label_accuracy']['pca2']:.1f}",
            f"{item['mean_unseen_axis_cosine']['raw13']:.2f}",
            f"{item['mean_unseen_axis_cosine']['pca2']:.2f}",
            f"{100*item['mean_same_source_tool_as_raw13']['pca2']:.1f}"])
    out.append(table(["カテゴリ","PCA2分散保持 [%]","13D同一ツール検索 [%]",
        "PCA2同一ツール検索 [%]","未知軸余弦13D","未知軸余弦PCA2",
        "PCA2/13D同じ既知ツール選択 [%]"],geometry))
    out.append("<details><summary>初期姿勢空間の次元・ホワイトニング比較（オフラインのみ）</summary>")
    for category in CATEGORIES:
        item=report["categories"][category]["offline"]
        out.append(f"<h3>{category.title()}</h3>"+table(
            ["初期姿勢の距離空間","既知ツールの検索正解率 [%]",
             "未知ツール専用軸との余弦","13Dと同じ選択元 [%]"],
            [[method,f"{100*item['mean_known_label_accuracy'][method]:.1f}",
              f"{item['mean_unseen_axis_cosine'][method]:.2f}",
              f"{100*item['mean_same_source_tool_as_raw13'][method]:.1f}"]
             for method in METHODS]))
    out.append("<p>ホワイトニングと3Dは近傍検索の感度分析であり、物理制御はPCA2と13Dを比較しました。</p></details>")
    for category in CATEGORIES:
        out.append(f"<details><summary>{category.title()}：未知ツール各foldの初期姿勢PCA2図</summary>")
        for variant in range(1,6):
            out.append(f"<h3>{category.title()} {variant} を未知ツールとした場合</h3>"
                       +pca_scatter(bundle,category,variant))
        out.append("</details>")
    out.append("<h2>MuJoCo：目標ツール角度への制御</h2><p>各ツール3初期姿勢×5目標角度"
               "（範囲の10、30、50、70、90%）。再構成ハンド姿勢へ5秒の直線軌道、終端2秒保持。"
               "13D/PCA2の『軸＋較正』は選択元ツールの平均成功姿勢・角度逆写像も使用。"
               "『軸のみ』は軸だけを選び、平均姿勢と逆写像を4ツール共通に固定。"
               "タスク専用参照だけは未知ツールの成功DBを使うためゼロショットではありません。"
               "Scissorsは従来TDSと同じ1.5倍Palm球、Push/Sprayは落下・転倒を拒否し接触数を要求せず、"
               "Sprayの床接触は有効です。</p>")
    for category in CATEGORIES:
        item=report["categories"][category]
        out.append(f"<h3>{category.title()}</h3>")
        rows=[[NAMES[a],f"{100*v['mae_fraction']:.1f}",
               f"{100*v['held_fraction']:.1f}",f"{100*v['joint_clip_fraction']:.1f}",
               f"{v['mean_axis_cosine']:.2f}"] for a,v in item["by_arm"].items()]
        out.append(table(["方式","正規化角度MAE [%]","2秒保持 [%]",
                          "関節指令クリップ [%]","対象ツール専用軸との余弦"],rows))
        out.append(mae_graph(category,item))
        out.append("<p>"+"　".join(f"<span style='color:{COLORS[a]}'>■</span> {NAMES[a]}"
                                 for a in ARMS)+"</p>")
        compare=[]
        for reference in ("raw13_nn_package","common_axis"):
            result=item["pca2_vs"][reference]
            lo,hi=result["bootstrap95"]
            compare.append([NAMES[reference],
                f"{100*result['mean_difference']:+.1f}",
                f"[{100*lo:+.1f},{100*hi:+.1f}]",
                f"{result['improved_tool_count']}/5"])
        axis_only=item["pca2_axis_only_vs_common"]
        axis_lo,axis_hi=axis_only["bootstrap95"]
        compare.append(["軸のみPCA2 NN − 共通軸",
                        f"{100*axis_only['mean_difference']:+.1f}",
                        f"[{100*axis_lo:+.1f},{100*axis_hi:+.1f}]",
                        f"{axis_only['improved_tool_count']}/5"])
        out.append("<p>上2行はPCA2『軸＋較正』との差、最下行はPCA2『軸のみ』と共通軸の差"
                   "（負＝PCA2側が良い）。"
                   "95%範囲は5ツール単位の再標本化による記述値で、有意差判定ではありません。</p>")
        out.append(table(["比較相手","平均MAE差 [pt]","95%記述範囲 [pt]",
                          "PCA2が勝ったツール"],compare))
        out.append("<details><summary>ツール別MAEと目標→実角度グラフ</summary>")
        task_rows=[[f"{category.title()} {i}"]+
                   [f"{100*item['by_tool'][str(i)][a]['mae_fraction']:.1f}" for a in ARMS]
                   for i in range(1,6)]
        out.append(table(["ツール"]+[NAMES[a]+" [%]" for a in ARMS],task_rows))
        for variant in range(1,6):
            out.append(f"<h4>{category.title()} {variant}</h4>"+
                       response_graph(item["physical"],category,variant))
        out.append("</details>")
        raw_rows=[[r["task"],NAMES[r["arm"]],r["pose"],
                   f"{100*r['target_fraction']:.0f}",f"{r['achieved_angle']:.4f}",
                   f"{100*r['normalized_abs_error']:.1f}",
                   "可" if r["supported"] else "不可",
                   "有" if r["joint_clipped"] else "無",
                   r["selected_source"] or "—"]
                  for r in sorted(item["physical"],key=lambda r:(r["task"],r["pose"],
                      r["target_fraction"],ARMS.index(r["arm"]))) ]
        out.append("<details><summary>全375試行の終端データ</summary>"+
                   table(["対象","方式","初期姿勢","目標 [%]","実角度 [rad]",
                          "正規化誤差 [%]","保持","関節クリップ","選択元"],raw_rows)+
                   "</details>")
    sc=report["categories"]["scissors"]
    sp=report["categories"]["spray"]
    pu=report["categories"]["push"]
    out.append("<h2>結論と次の研究・実装判断</h2>"
               "<p><strong>2D PCA-NNを標準の未知ツール制御器へ採用する根拠は得られませんでした。</strong>"
               f"『軸＋較正』の平均MAEはScissorsで"
               f"{100*sc['by_arm']['pca2_nn_package']['mae_fraction']:.1f}%"
               f"（13D NN {100*sc['by_arm']['raw13_nn_package']['mae_fraction']:.1f}%）、"
               f"Sprayで{100*sp['by_arm']['pca2_nn_package']['mae_fraction']:.1f}%"
               f"（13D NN {100*sp['by_arm']['raw13_nn_package']['mae_fraction']:.1f}%）、"
               f"Pushで{100*pu['by_arm']['pca2_nn_package']['mae_fraction']:.1f}%"
               f"（13D NN {100*pu['by_arm']['raw13_nn_package']['mae_fraction']:.1f}%）でした。"
               "PCA2はScissors・Sprayの初期姿勢分散を高率で保持しますが、"
               "それは角度制御に重要な方向を保持する保証ではありません。</p>"
               "<p>軸だけをPCA2-NNで選び、平均姿勢と逆写像を共通に固定した場合のMAEは"
               f"Scissors {100*sc['by_arm']['pca2_selected_axis_only']['mae_fraction']:.1f}%、"
               f"Spray {100*sp['by_arm']['pca2_selected_axis_only']['mae_fraction']:.1f}%、"
               f"Push {100*pu['by_arm']['pca2_selected_axis_only']['mae_fraction']:.1f}%でした。"
               "Spray 3には有効な例がありますが、カテゴリ全体での一貫した改善はありません。"
               "逆にScissors 3の『軸＋較正』の改善は『軸のみ』では再現しません。"
               "これは操作軸の近さより、選択元の平均姿勢や角度逆写像の適合が支配的な場合を示します。</p>"
               "<p>主評価指標は未知ツールの正規化角度MAE、副指標は保持率・関節クリップ・"
               "目標対実角度曲線とすべきです。PCA分散保持率、同一ツール検索率、"
               "操作軸余弦は原因分析には有用ですが、単独で制御の汎化を表しません。"
               "次の実装候補は、共通1自由度軸を出発点として新規ツールの少数の安全な角度較正点で"
               "逆写像を更新し、ゼロショットと少数点適応を分けて評価することです。"
               "初期姿勢から軸を予測する研究を続けるなら、形状変化を系統的に増やした学習ツール群と、"
               "初期姿勢が既知PCA空間から外れた場合の外挿判定が必要です。</p>"
               "<p>制限：未知ツールに適切な初期姿勢を人が用意できる前提、5ツール/カテゴリ・"
               "3初期姿勢/ツールの予備評価、DB上の線形逆写像、シミュレーションの保持判定。"
               "実機の把持安全性や真の新形状群への統計的汎化は未検証です。</p></html>")
    return "".join(out)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path("analysis/initial_pca2_nn_20261008"))
    summarize(parser.parse_args().root)
