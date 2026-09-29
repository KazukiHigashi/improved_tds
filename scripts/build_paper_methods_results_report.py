#!/usr/bin/env python3
"""Rebuild the evidence-linked manuscript preparation report from frozen JSONs."""
from __future__ import annotations

import html
import json
import os
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/paper_methods_results_20260926"
SC = ROOT / "analysis/scissors_palm_all5_tds_direct_20260926"
OLD = ROOT / "analysis/strict_synergy_paper_v4_fullrange"
RUN_SC = ROOT / "runs/scissors_palm_all5_3m_20260926"
RUN_OLD = ROOT / "runs/unified_reward_fullrange_4x4_v1"
FULL = ROOT / "analysis/scissors4_fullscore_unseen_20260926"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def relative(path: Path) -> str:
    return html.escape(Path(os.path.relpath(path, OUT)).as_posix(), quote=True)


def link(path: Path, label: str) -> str:
    return f'<a href="{relative(path)}">{html.escape(label)}</a>'


def table(headings, rows):
    top = "".join(f"<th>{html.escape(str(x))}</th>" for x in headings)
    body = "".join("<tr>" + "".join(f"<td>{x}</td>" for x in row) + "</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{top}</tr></thead><tbody>{body}</tbody></table></div>"


def pct(x):
    return f"{100 * x:.1f}"


def f3(x):
    return "—" if x is None else f"{x:.3f}"


def make_charts(sc_rows, old_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    ids = np.arange(1, 6)
    for ax, metric, ylabel, multiplier in (
        (axes[0], "independent_coverage", "Independent coverage of 0–1 rad [%]", 100),
        (axes[1], "continuous_coverage", "Continuous sweep coverage [%]", 100),
        (axes[2], "inverse_mae", "Inverse angle MAE [rad]", 1),
    ):
        for method, marker in (("pca", "o"), ("pls", "s")):
            vals = [sc_rows[f"scissors_lite_{i}"][method][metric] * multiplier for i in ids]
            deviations = [sc_rows[f"scissors_lite_{i}"][method][metric + "_sd"] * multiplier for i in ids]
            ax.errorbar(ids, vals, yerr=deviations, marker=marker, label=method.upper(), linewidth=2,
                        capsize=3)
        ax.set_xticks(ids)
        ax.set_xlabel("Scissors variant")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(OUT / "fig_scissors_pca_pls.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), constrained_layout=True)
    for j, category in enumerate(("push", "spray")):
        ax = axes[j]
        for method, marker in (("pca", "o"), ("pls", "s")):
            vals = [100 * old_rows[f"{category}_lite_{i}"][method]["reachable"] for i in ids]
            deviations = [100 * old_rows[f"{category}_lite_{i}"][method]["reachable_sd"] for i in ids]
            ax.errorbar(ids, vals, yerr=deviations, marker=marker, label=method.upper(),
                        linewidth=2, capsize=3)
        ax.set_title(category.capitalize() + " (legacy protocol)")
        ax.set_xlabel("Variant")
        ax.set_ylabel("Targets inside fitted domain [%]")
        ax.set_ylim(0, 105)
        ax.set_xticks(ids)
        ax.grid(alpha=.25)
    axes[0].legend()
    fig.savefig(OUT / "fig_push_spray_legacy.png", dpi=180)
    plt.close(fig)

    full = load(FULL / "summary.json")
    fig, ax = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)
    for method, marker in (("pca", "o"), ("pls", "s")):
        vals = full[method]["by_unseen_score"]
        ax.plot([v["score"] for v in vals], [v["held"] / v["trials"] * 100 for v in vals],
                marker=marker, label=method.upper())
    ax.set(xlabel="Unseen midpoint score", ylabel="Palm-sphere hold rate [%]",
           title="Scissors 4: disconnected low-score branch")
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(OUT / "fig_scissors4_fullscore.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.6), constrained_layout=True)
    tasks = list(sc_rows) + list(old_rows)
    best = [x["training"]["validation"]["best"]["success_rate"] * 100
            for x in list(sc_rows.values()) + list(old_rows.values())]
    ax.bar(np.arange(len(tasks)), best, color=["#35759f"] * 5 + ["#499f81"] * 5 + ["#b87947"] * 5)
    ax.axhline(95, color="#b53939", linestyle="--", linewidth=1, label="Early-stop threshold")
    ax.set_xticks(np.arange(len(tasks)), [x.replace("_lite_", " ") for x in tasks], rotation=45, ha="right")
    ax.set_ylabel("Best validation ever-success [%]")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=.25)
    ax.legend()
    fig.savefig(OUT / "fig_training_validation.png", dpi=180)
    plt.close(fig)


def main():
    sc_rows, old_rows = {}, {}
    for i in range(1, 6):
        task = f"scissors_lite_{i}"
        train = load(RUN_SC / task / "model_training_summary.json")
        reports = [load(p) for p in sorted((SC / "results" / task).glob("20*/report.json"))]
        if len(reports) != 5:
            raise RuntimeError(f"expected five Scissors split reports for {task}")
        db = load(RUN_SC / task / "success_postures.progress.json")
        if db["rows"] != 200 or not db["complete"] or not db["success_only"]:
            raise RuntimeError(f"incomplete success-only database: {task}")
        item = {"training": train, "database": db, "reports": reports}
        for method in ("pca", "pls"):
            configs = [r["configurations"][f"{method}_1d"] for r in reports]
            summaries = [c["summary"] for c in configs]
            item[method] = {
                "independent_coverage": statistics.mean(s["independent_coverage"] for s in summaries),
                "independent_coverage_sd": statistics.stdev(s["independent_coverage"] for s in summaries),
                "continuous_coverage": statistics.mean(s["continuous_coverage"] for s in summaries),
                "continuous_coverage_sd": statistics.stdev(s["continuous_coverage"] for s in summaries),
                "held_fraction": statistics.mean(s["held_fraction"] for s in summaries),
                "inverse_mae": statistics.mean(s["controller_comparison"]["isotonic"]["mae"] for s in summaries),
                "inverse_mae_sd": statistics.stdev(s["controller_comparison"]["isotonic"]["mae"] for s in summaries),
                "ramp_mae": statistics.mean(s["controller_comparison"]["ramp"]["mae"] for s in summaries),
                "domain": [statistics.mean(c["inverse_target_range"][edge] for c in configs)
                           for edge in (0, 1)],
            }
        sc_rows[task] = item
    for category in ("push", "spray"):
        for i in range(1, 6):
            task = f"{category}_lite_{i}"
            meta = load(OLD / "databases" / f"{task}.json")
            reports = [load(p) for p in sorted((OLD / "reports").glob(f"{task}_seed*.json"))]
            if len(reports) != 5 or meta["rows"] != 672:
                raise RuntimeError(f"incomplete legacy result: {task}")
            train = load(RUN_OLD / task / "model_training_summary.json")
            item = {"training": train, "database": meta, "target_high": reports[0]["target_values"][-1]}
            for method in ("pca", "pls"):
                outcomes = [o for r in reports for o in r["configurations"][f"{method}__none"]["outcomes"]]
                reachable_by_split = [statistics.mean(o["reachable"] for o in
                    r["configurations"][f"{method}__none"]["outcomes"]) for r in reports]
                item[method] = {
                    "mae": statistics.mean(o["abs_error"] for o in outcomes),
                    "held": statistics.mean(o["held"] for o in outcomes),
                    "reachable": statistics.mean(o["reachable"] for o in outcomes),
                    "reachable_sd": statistics.stdev(reachable_by_split),
                    "n": len(outcomes),
                }
            old_rows[task] = item
    make_charts(sc_rows, old_rows)

    train_rows = []
    for category, rows in (("Scissors", sc_rows), ("Push/Spray", old_rows)):
        for task, data in rows.items():
            t = data["training"]
            b = t["validation"]["best"]
            train_rows.append((html.escape(task), f"{t['total_timesteps']:,}",
                               f"{b['timesteps']:,}", pct(b["success_rate"]),
                               pct(b["final_success_rate"]) if "final_success_rate" in b else "未記録",
                               f3(b["mean_abs_error"]),
                               "rad" if task.startswith("scissors") or task.startswith("spray") else "m"))
    sc_table = []
    for task, data in sc_rows.items():
        for method in ("pca", "pls"):
            m = data[method]
            sc_table.append((html.escape(task), method.upper(), pct(m["independent_coverage"]),
                             pct(m["continuous_coverage"]), pct(m["held_fraction"]),
                             f"{m['domain'][0]:.3f}–{m['domain'][1]:.3f}",
                             f3(m["inverse_mae"]), f3(m["ramp_mae"])))
    old_table = []
    for task, data in old_rows.items():
        for method in ("pca", "pls"):
            m = data[method]
            old_table.append((html.escape(task), method.upper(),
                              f3(data["target_high"]), f"{data['database']['successful_rows']}/672", pct(m["reachable"]),
                              pct(m["held"]), f3(m["mae"]),
                              "rad" if task.startswith("spray") else "m"))
    reward_rows = [
        ("Scissors 1–5", "2 = 粗 1 + 細 1", "1", "1", "0.25", "5", "25", "0.1", "0.002", "0 + gap 1.0"),
        ("Push 1–5（保存 run）", "2", "1", "2", "0.25", "5", "250", "0.02", "0.002", "なし"),
        ("Spray 1–5（保存 run）", "2", "1", "2", "0.25", "5", "250", "0.02*", "0.002*", "なし"),
    ]
    sc_links = "".join(f"<li>{link(SC / 'results' / f'scissors_lite_{i}' / 'index.html', f'Scissors {i}: sweep, score–angle, PCA/PLS')}</li>" for i in range(1, 6))
    sources = [
        ("学習設定", ROOT / "src/improved_tds/experiments/train_sb3.py"),
        ("成功判定と Palm 球", ROOT / "src/improved_tds/environments/scissors_reach.py"),
        ("Scissors 行動平滑化", ROOT / "src/improved_tds/environments/scissors_smooth.py"),
        ("Push/Spray の成功・報酬", ROOT / "src/improved_tds/environments/pose_pool.py"),
        ("成功姿勢 200 件の収集", ROOT / "src/improved_tds/experiments/collect_reach_successes.py"),
        ("Scissors TDS 評価", ROOT / "src/improved_tds/experiments/reach_tds_direct.py"),
        ("Push/Spray 旧評価", ROOT / "src/improved_tds/experiments/evaluate_strict_synergy_control.py"),
        ("実機操作スクリプト", ROOT / "scripts/real_experiment/operate_tds_jointtrajectory.py"),
    ]
    source_list = "".join(f"<li>{link(path, label)}</li>" for label, path in sources)
    report = f"""<!doctype html><html lang='ja'><head><meta charset='utf-8'><title>Improved TDS: 論文手法・実験監査</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1180px;margin:auto;padding:24px;color:#183041;background:#f6f8fa}}
h1,h2,h3{{line-height:1.3}}section{{background:white;border:1px solid #dce4e9;border-radius:12px;margin:18px 0;padding:18px 22px}}
table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #dbe2e7;padding:7px 9px;text-align:left;white-space:nowrap}}th{{background:#eef4f7;position:sticky;top:0}}
.scroll{{overflow:auto}}img{{max-width:100%;height:auto;border:1px solid #e1e7ea;border-radius:7px}}.warn{{border-left:4px solid #c35e2c;padding-left:12px}}
.ok{{border-left:4px solid #26816a;padding-left:12px}}code{{background:#eef1f3;padding:1px 4px}}.small{{font-size:13px;color:#526671}}</style></head><body>
<h1>Improved TDS：論文手法・実験結果・実機計画</h1>
<p>2026-09-26 時点の凍結済み成果物から再集計。<strong>現在の 15 タスクは同一プロトコルの比較ではない。</strong> Scissors は新方式、Push/Spray は旧方式の完了済み個別レポートを使用した。数値はシミュレーションであり、実機結果は未取得。</p>
<section><h2>1. 論文原案との関係と研究命題</h2>
<p>参照原稿：<em>Tool DoF Synergy: A Framework for Synergy Acquisition and Control in Articulated Tool Use</em>（添付 <code>submit_opt.pdf</code>、8 頁）。原案は RL から得た成功姿勢を PCA で解析し、PC1 スコアをツール角度の 1 自由度入力として用いる。12 点の同定とランプ関数の逆写像、5 秒の姿勢軌道、Scissors 3/4 の Shadow Hand 実機検証を扱う。原稿中の実機誤差 ≤2° は<strong>旧研究の結果</strong>であり、以下の新方式の実機成績ではない。</p>
<p>現行研究の問いは、成功姿勢から推定した PCA/PLS の操作軸によって、ハンド全関節を独立に制御せずにツール自由度を再現できるか、である。1 自由度を主評価とし、Push/Spray で必要な把持軸を追加する 2 自由度を対照とする。</p></section>
<section><h2>2. 現行手法（実装と照合済み）</h2>
<ol><li>32 種の保存初期姿勢と全目標範囲を用い、SAC の Best Policy を学習・選択する。検証は 10 万 aggregate steps ごと、成功率 95% が 2 回連続で早期停止、checkpoint と Best Policy を別保存する。</li>
<li>Scissors は目標角度 0–1 rad、Palm に剛体追従する半径 0.05 m の球内にヒンジ中心があることを把持条件とし、角度誤差 ≤0.03 rad の 5 連続ステップを成功とする。接触数・落下・姿勢は診断情報で、成功の追加ゲートではない。学習アクションの 1 step 変化は 0.05 に制限し、接触補助報酬の重みは 0。</li>
<li>Scissors の Best Policy で達成角度 10 bin × 成功姿勢 20 = 200 件を再収集する。失敗姿勢は DB に入れない。PCA/PLS は保存最終姿勢を<strong>全体平均で中心化</strong>し、PCA の最大分散軸または PLS の角度関連軸を求める。姿勢を <code>q(ρ,g)=clip(μ+ρ a v_op+g b v_grasp)</code> で再構成する。</li>
<li>DB スコアの観測最小・最大をそれぞれ 1.5 倍した範囲に 17 点を置く。保存初期姿勢から再構成姿勢へ 5 秒で線形に移動し、2 秒保持する。逆写像は<strong>保持が成立する連続した概ね単調な区間</strong>に限定し、等張回帰とランプ関数を比較する。区間外の目標は成功範囲に含めない。</li>
<li>5 分割の初期姿勢分割を用い、各分割で calibration 8 姿勢、test 8 姿勢を分離する。ただし PCA/PLS 軸は 200 件 DB 全体から学習され、<strong>完全な未知姿勢汎化試験ではない</strong>。2 自由度は固定把持スコアを選ぶ方式であり、連続的な把持・操作の同時最適化ではない。</li></ol>
<p class='small'>原案との差：DDPG+HER→SAC、学習中 FIFO 2000 成功姿勢→Best Policy から均衡 200 件、PCA のみ→PCA/PLS、12 点→17 点、原案のランプ単独→等張/ランプ比較。論文で原案のパラメータを現行方式と混同しないこと。</p></section>
<section><h2>3. 全タスクの条件と適用範囲</h2>
{table(['カテゴリ','変数','目標範囲','学習成功判定','TDS データ','現行評価'],[
('Scissors','1–5','0–1 rad','Palm 球 + |角度誤差|≤0.03 rad、5 step','200 件、10 bin×20、成功のみ','直接姿勢、5 s+2 s、17 score、PCA/PLS、5 分割'),
('Spray','1–5','0–0.4 rad','目標到達 + 指接触≥2 + 非落下/非転倒、5 step','672 試行中の成功行から各分割 100 件','旧 direct absolute command、別 calibration/preload'),
('Push','1–5','0–7～12 mm（変数別）','目標到達 + 指接触≥2 + 非落下/非転倒、5 step','672 試行中の成功行から各分割 100 件','旧 direct absolute command、別 calibration/preload')])}
<p>Push の上限は変数ごとに異なる。後表の値を参照。全カテゴリとも SAC、32 初期姿勢、100k step 検証の系統だが、<strong>学習上限・実際の報酬重み・TDS 再収集法は実験 run により異なる</strong>。Scissors と Push/Spray の Coverage/MAE を同列で統計検定しない。</p></section>
<section><h2>3a. 保存 run の学習設定</h2>
{table(['run 群','SAC 環境数 / 並行学習','batch / replay','勾配更新 / warm-up','entropy','上限 aggregate step','HER'],[
('Scissors 1–5','4 / campaign 最大 4','256 / 1,000,000','2 / 1,000','固定 0.02','3,000,000','なし'),
('Push 1–5','4 / campaign 最大 4','256 / 1,000,000','2 / 1,000','auto','1,000,000','保存 summary に比率未記録'),
('Spray 1–5','4 / campaign 最大 4','256 / 1,000,000','2 / 1,000','auto','1,000,000','保存 summary に比率未記録')])}
<p class='small'>保存された <code>model_training_summary.json</code> を優先。Push/Spray の旧 run と現行 <code>train_unified_reward_campaign.ps1</code> の既定引数が一致しないため、現在のスクリプト既定値を過去 run の設定と推定しない。</p></section>
<section><h2>3b. 保存済み run の報酬係数（タスクカテゴリ内共通）</h2>
{table(['カテゴリ','tracking','progress','grasp','interaction','success','failure','Δaction cost','|action| cost','contact auxiliary'],reward_rows)}
<p class='small'>正の値は報酬、failure と control cost は減点の絶対値。Scissors の gap は requested–applied action 差への減点。*Spray の制御コストは Push と共通の学習 CLI パラメータから渡されるため、上段の <code>spray_reward</code> 保存辞書には別キーとして現れない。既存説明文の「全カテゴリ共通係数」と保存済み Push/Spray run は一致せず、表は実際の run metadata を優先した。</p></section>
<section><h2>4. RL 結果：Best Policy の検証</h2>
<figure><img src='fig_training_validation.png' alt='Best validation success by task'><figcaption>図候補 0. 保存 run における Best Policy の検証成功率。Scissors と Push/Spray は成功判定が異なるため、横断順位ではない。</figcaption></figure>
{table(['タスク','停止 step','Best step','ever 5-step 成功 %','最終時刻成功 %','目標誤差 MAE','単位'],train_rows)}
<p class='small'>Scissors の ever-success と最終時刻 success は異なる。Scissors 4/5 は ever-success が高くても最終保持率が低い。Push/Spray の最終時刻指標は保存されていない。Best は学習全期間中の選択値で、停止時モデルと同一とは限らない。</p></section>
<section><h2>5. Scissors：論文掲載候補の主要結果</h2>
<figure><img src='fig_scissors_pca_pls.png' alt='Scissors PCA versus PLS coverage and inverse MAE'><figcaption>図候補 A. 独立姿勢・連続スイープの Coverage と逆写像 MAE（5 分割平均 ± 標本標準偏差、1 DoF）。MAE は各分割の<strong>選択された可逆域内</strong>の目標に限る。</figcaption></figure>
{table(['タスク','軸','独立 Coverage %','連続 Coverage %','保持率 %','逆写像域 rad','等張 MAE rad','ランプ MAE rad'],sc_table)}
<p>Scissors 3/5 はほぼ全域の独立 Coverage、Scissors 1/2 は目標範囲の一部のみ。Scissors 4 の独立 Coverage は広いが、連続 sweep と保持は大幅に低下する。2 DoF では 25 分割中 24 分割で把持スコア 0 が選ばれ、ほぼ 1 DoF と同じ挙動であり、追加自由度の利益を示す証拠は弱い。</p>
<h3>Scissors 4 の全スコア追加検証</h3><figure><img src='fig_scissors4_fullscore.png' alt='Scissors 4 held rate versus score'><figcaption>図候補 B. 未使用の中点 score 16 点における保持率（各 40 試行）。低 score 側の成功島と主な成功区間の間には不安定域がある。</figcaption></figure>
<p>全域順写像フィットの未使用 score MAE は PCA: 線形 0.068、等張 0.073、ランプ 0.072 rad、PLS: 0.080/0.083/0.084 rad（保持失敗も含む各 640 試行）。PCA 483/640、PLS 463/640 のみ保持。低 score 側を含めて曲線を描くことはできても、単調かつ保持可能な<strong>全域逆写像は成立していない</strong>。上記の可逆域限定 MAE と混同しない。</p>
<ul>{sc_links}</ul><p>{link(FULL / 'summary.json','Scissors 4 全スコア追加検証の数値')}</p></section>
<section><h2>6. Push / Spray：旧プロトコルの結果（参考、直接比較不可）</h2>
<figure><img src='fig_push_spray_legacy.png' alt='Push and Spray legacy inverse domain reachability'><figcaption>図候補 C. 旧方式で 11 目標点が同定された逆写像域に入る割合。5 分割平均 ± 標本標準偏差。</figcaption></figure>
{table(['タスク','軸','目標上限','DB 成功/全','可逆域内目標 %','保持率 %','全試行角度/深さ MAE','単位'],old_table)}
<p>各設定は 5 分割×8 test 姿勢×11 目標 = 440 試行。誤差には逆写像域外目標に対する境界値指令も含むため、可逆域内 MAE ではない。特に Push 1、Spray 1–5 は可逆域内目標が少なく、この数値のみで 1 DoF 操作が成立したとは言えない。旧 campaign 全体の status は <code>running</code>、ただし Push/Spray 各 5×5 report は存在する。</p></section>
<section><h2>7. 実装監査：論文に書く前に区別すべき点</h2>
{table(['論点','コード・成果物から確認した実態','論文化時の扱い'],[
('成功姿勢の起源','Scissors は Best Policy による学習後の再収集。学習中候補そのものではない','「RL 学習で得たポリシーから再収集」と明記'),
('Push/Spray DB','672 行に失敗が混在し、fit には各分割 100 成功行を抽出','「全 DB が成功姿勢」と記述しない'),
('制御手順','Scissors は準備オフセットなし、5 秒直接姿勢。Push/Spray 旧評価は reference/preload を使用','15 タスク共通の手法として未統一'),
('実験期間/報酬','現行 Scissors run は最大 3M、Push/Spray 保存 run は最大 1M。docs の統一重みと保存 run 設定が異なる','保存 run の設定を表に使い、再学習なしに同条件と主張しない'),
('評価分割','Scissors の初期姿勢は分割、軸学習 DB は全初期姿勢を含む','「未知姿勢汎化」と主張しない'),
('Scissors 4','低 score 成功島と中央失敗域があり、全域の単値逆写像は存在しない','可逆域を明記し、外挿しない'),
('実機','現在の改良方式の実機データなし。原案の ≤2° は別条件','実機結果欄は未取得とする')])}
<p class='warn'>論文提出前の最優先事項：Push/Spray を Scissors と同じ成功のみ 200 件・直接 5 秒姿勢・同じ calibration/test 分割で再評価し、報酬設定と学習 cap を run ごとに正確に記載する。全 15 タスクの同一プロトコル下の PCA/PLS 優劣は現時点では結論保留。</p></section>
<section><h2>8. 実機実験の手順（計画、未実施）</h2>
<p>参照：添付原稿の実機 Scissors 3/4、提供 ZIP <code>tool_dof_synergy-main/real_experiment/</code>。元コードは ROS1 の <code>/rh_trajectory_controller/follow_joint_trajectory</code> action と Arduino 角度センサ（度出力）を使用する。提供コード中の急速な 0.2 秒指令、ハードコードの PC1 範囲、関節順序変換は現行方式と一致しないため、再利用せず安全前提を見直した。</p>
<p class='small'>実行形式の差：元コードは <code>python operate_lite_pc1_score.py &lt;operateconfig名&gt;</code> と pickle 設定に依存する。新コードは同じ ROS action / 角度センサを使うが、現行 PCA/PLS report、明示的な JSON 関節対応、実機校正 CSV を引数で受ける。旧 pickle の暗黙の関節順序・固定 PC1 値を採用しないための意図的な変更である。</p>
<ol><li>Scissors 3/4 を候補とし、実機の道具寸法、取り付け姿勢、角度センサゼロ、緊急停止、作業空間、指・物体の安全境界を確認する。人が始動姿勢を作り、サンプルごとに同じ始動姿勢に戻す。</li>
<li>シミュレーションの 13 関節順と Shadow Hand の実 joint 名・符号・オフセット・MuJoCo actuator 範囲・実機機械的限界・許容速度を一対一検証し、<code>joint-map.json</code> を作る。再構成姿勢は MuJoCo 範囲で clip してから変換する。自動変換や旧コードの番号削除/符号反転は用いない。</li>
<li>1 目標につき、まず <code>--score</code> で dry-run plan と SHA を出し、承認済み SHA を指定して 5 秒の ROS 軌道を 1 回だけ送る。2 秒保持後、Arduino の角度を記録する。操作中は監視者と物理停止手段を必須とする。</li>
<li>選択済みの保持可能 score 区間を低振幅から段階的に計測し、score–実角度の単調な有効枝を同定する。実機 calibration CSV の単調枝のみを逆写像に使う。Scissors 4 の低 score 島は中央失敗域を跨いで統合しない。</li>
<li>実機 calibration に使わない目標角度を選び、<code>--target-angle --real-calibration</code> から score を逆算する。各目標を複数反復し、角度 MAE、95 パーセンタイル、保持率、可逆域、操作時間、停止件数を報告する。</li></ol>
<p>スクリプト：{link(ROOT / 'scripts/real_experiment/operate_tds_jointtrajectory.py','operate_tds_jointtrajectory.py')} ／ {link(ROOT / 'scripts/real_experiment/joint_map.template.json','joint map テンプレート')}。デフォルトは dry-run で、ROS/シリアルには触れない。実行には検証済み joint map、実機校正 CSV（目標角度モード）、<code>--execute</code> と dry-run の SHA、Arduino ポート、記録先 CSV が必要。実機での動作・安全性は<strong>未検証</strong>。</p>
<h3>実行形式</h3><pre>python scripts/real_experiment/operate_tds_jointtrajectory.py --report analysis/scissors_palm_all5_tds_direct_20260926/results/scissors_lite_3/20260926/report.json --joint-map joint-map.json --score 0.0
python scripts/real_experiment/operate_tds_jointtrajectory.py --report .../report.json --joint-map joint-map.json --target-angle 0.5 --real-calibration calibration.csv
# dry-run 計画・安全項目を確認後のみ、同じ引数へ
# --execute --approve-sha &lt;表示された SHA&gt; --serial-port /dev/ttyACM0 --log-csv real_trials.csv を追加</pre>
<p class='warn'>未確定事項：実機 13 関節の対応、制御器の joint 制限/単位、実機始動姿勢、Arduino センサ校正、Scissors 3/4 のどちらを主実験とするか、反復回数・目標角度列、実機の許容速度/停止閾値。これらは実機コマンド送信前に確定・承認が必要。</p></section>
<section><h2>9. 再現性・証拠</h2><ul>{source_list}</ul><p>{link(SC / 'index.html','Scissors 5 タスク元 HTML')} ／ {link(OLD / 'status.txt','旧 campaign status')} ／ {link(FULL / 'summary.json','Scissors 4 全 score 数値')}</p>
<p class='small'>本ページは <code>scripts/build_paper_methods_results_report.py</code> により JSON から再生成。数値を HTML に手入力していない。旧原稿は方法論と実機手順の背景資料として参照し、旧結果を現行実験に転記していない。</p></section></body></html>"""
    (OUT / "index.html").write_text(report, encoding="utf-8")
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
