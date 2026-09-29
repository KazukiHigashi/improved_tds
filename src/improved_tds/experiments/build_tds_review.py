"""Evidence-backed 15-task review, explicitly distinguishing missing from rejected controllers."""
import argparse
from collections import defaultdict
from datetime import datetime
import html
import json
from pathlib import Path
import numpy as np
from improved_tds.experiments.summarize_adaptive_synergy import main as summarize


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    reports=sorted((root/'reports').glob('*_seed*.json'))
    if not reports:raise ValueError('No completed evaluation reports')
    parsed=defaultdict(list)
    for path in reports:
        d=json.loads(path.read_text(encoding='utf-8'))
        if d.get('pilot') or set(d['configurations'])!={'pca','pls'}:raise ValueError(f'Invalid production report: {path}')
        parsed[path.stem.split('_seed')[0]].append(d)
    summarize(['--report-dir',str(root/'reports'),'--output-dir',str(root/'summary'),
               '--expected-reports',str(len(reports))])
    out=root/'review';out.mkdir(exist_ok=True)
    metrics=[];rows=[];audits=[]
    def pct(x):return '—' if x is None else f'{100*x:.1f}%'
    for task in manifest['tasks']:
        name=task['name'];ds=parsed[name];database=json.loads((root/'databases'/f'{name}.json').read_text(encoding='utf-8'))
        audit=json.loads((root/'fit_databases'/name/'audit.json').read_text(encoding='utf-8'))
        ready=sum(s['complete'] for s in audit['splits'].values())
        if len(ds)==5: state='評価完了'
        elif name=='push_lite_3':state='未完了：把持指標PLSを推定不可'
        elif name=='scissors_lite_5':state='未完了：Fit高角度区間が不足'
        else:state=f'未完了：{len(ds)}/5分割'
        audits.append(f'<tr><td>{name}</td><td>{database["successful_rows"]:,} / {database["rows"]:,}</td><td>{ready}/5</td><td>{len(ds)}/5</td><td>{state}</td></tr>')
        for method in ('pca','pls'):
            arms=[d['configurations'][method]['arms']['joint'] for d in ds]
            calibrated=sum(x['calibrated'] for x in arms)
            coverage=[float(np.mean([q['held_coverage'] for q in x['poses']])) if x['poses'] else 0. for x in arms]
            held=[float(np.mean([o['held'] and not o['failed'] for q in x['poses'] for o in q['outcomes']])) if x['poses'] else 0. for x in arms]
            trials=[o for x in arms for q in x['poses'] for o in q['target_trials']]
            width=ds[0]['target_range'][1]-ds[0]['target_range'][0] if ds else None
            mae=float(np.mean([o['abs_error'] for o in trials])) if trials else None
            entry=dict(task=name,method=method,reports=len(ds),calibrated=calibrated,
                held_coverage=float(np.mean(coverage)) if coverage else None,
                held_rate=float(np.mean(held)) if held else None,mae=mae,
                normalized_mae=mae/width if mae is not None else None,target_trials=len(trials))
            metrics.append(entry)
            label='未実施' if not ds else 'Calibration不成立' if not calibrated else '実行済み'
            unit='m' if name.startswith('push') else 'rad'
            formatted_mae='—' if mae is None else f'{mae:.5f} {unit}'
            rows.append(f'<tr class="{"missing" if not ds else "rejected" if not calibrated else ""}"><td>{name}</td><td>{method.upper()}</td>'
                f'<td>{label}</td><td>{calibrated}/{len(ds)}' + ('（未実施）' if not ds else '') + '</td>'
                f'<td>{pct(entry["held_coverage"])}</td><td>{pct(entry["held_rate"])}</td><td>{formatted_mae}</td>'
                f'<td>{pct(entry["normalized_mae"])}</td><td>{len(trials)}</td></tr>')
    summary=dict(generated_local=datetime.now().isoformat(),evaluation_completed_utc=status['utc'],
        completed_tasks=sum(len(v)==5 for v in parsed.values()),expected_tasks=15,
        completed_reports=len(reports),expected_reports=75,production_complete=len(reports)==75,
        missing_tasks=[t['name'] for t in manifest['tasks'] if len(parsed[t['name']])!=5],joint_metrics=metrics)
    (out/'review.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    document='''<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>全15タスク TDS評価報告 — 2026-09-18</title><style>
body{font-family:system-ui,"Yu Gothic",sans-serif;color:#183047;background:#f5f7fa;margin:0}main{max-width:1250px;margin:auto;padding:32px 24px 70px}h1{font-size:28px}h2{margin-top:36px}p,li{line-height:1.8}a{color:#005ea8}.notice{padding:18px 22px;background:#fff0d1;border-left:5px solid #c87900;border-radius:5px}.cards{display:flex;gap:16px;margin:24px 0;flex-wrap:wrap}.card{background:white;padding:20px;border:1px solid #d8e1e8;border-radius:8px;min-width:180px}.card strong{font-size:28px;display:block}table{border-collapse:collapse;width:100%;background:white;font-size:14px}th,td{padding:10px;border-bottom:1px solid #dce3e9;text-align:left}th{background:#183047;color:white;white-space:nowrap}td{font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}.missing{background:#fff0d1}.rejected{background:#fce8e6}.muted{color:#52697c}nav{display:flex;gap:24px;flex-wrap:wrap}</style></head><body><main>
<p class="muted">IMPROVED TDS / PCA・PLS / 2026-09-18</p><h1>全15タスクのTDS評価報告</h1>
<div class="notice"><strong>部分結果：13タスク・65分割を評価済み／計画は15タスク・75分割。</strong><br>
処理は2026年9月18日09:56 JSTに終了しました。Push 3とScissors 5は未完了です。
未実施は「—」、評価したがCalibration不成立の条件はCoverage・把持維持率「0」として区別します。全15タスクの実験完了とは扱いません。</div>
<div class="cards"><div class="card">姿勢再収集<strong>15 / 15</strong></div><div class="card">評価処理完了<strong>13 / 15</strong></div><div class="card">分割レポート<strong>65 / 75</strong></div></div>
<nav><a href="#joint">Joint方式の結果</a><a href="#status">全タスクの実施状況</a><a href="../summary/index.html">全4条件の詳細・グラフ・姿勢画像</a><a href="review.json">集計JSON</a></nav>
<h2>主な結果</h2><ul>
<li>Push 1・2・4・5のJoint CoverageはPCAで72.4〜95.7%、PLSで67.0〜91.7%。方式の優劣はタスクごとに異なります。</li>
<li>Scissors 1・2はJoint CoverageがPCAで91.0%・96.1%、PLSで98.3%・98.7%です。</li>
<li><strong>Scissors 3・4はPCA／PLSとも全5分割でJointのCalibration不成立。</strong>成功姿勢DBの範囲が広くても、抽出した1自由度軸に沿うFF操作で把持を保ちながら単調応答を得られるとは限りません。</li>
<li>SprayのJoint CoverageはPCAで33.8〜58.2%、PLSで23.9〜64.4%。RLでの高成功率が、そのまま低自由度FF制御に移行していません。</li></ul>
<h2 id="joint">Joint方式：タスク別結果</h2>
<p>5分割平均。Coverage・把持維持率はCalibration不成立を0として含みます。MAEは実際に実行した全域9目標の試行に対する条件付き平均です。目標範囲外の逆写像要求は端点で飽和させ、その誤差も含みます。</p>
<div class="scroll"><table><thead><tr><th>タスク</th><th>方式</th><th>評価状態</th><th>Calibration成立</th><th>Coverage</th><th>把持維持率</th><th>MAE</th><th>正規化MAE</th><th>目標試行数</th></tr></thead><tbody>'''+''.join(rows)+'''</tbody></table></div>
<h2 id="status">データ収集・分割監査・評価状況</h2>
<p>全タスクとも32初期姿勢×101目標＝3,232エピソード。Fit／Calibration／Testを16／8／8初期姿勢に分離しました。各Fitでは成功姿勢を工具目標範囲の10区間から各10件、計100件選択し、PCAとPLSで共用しています。</p>
<div class="scroll"><table><thead><tr><th>タスク</th><th>再収集成功数／試行数</th><th>Fit監査合格</th><th>評価済み分割</th><th>状態</th></tr></thead><tbody>'''+''.join(audits)+'''</tbody></table></div>
<h2>未完了2タスクの理由</h2><ul>
<li><strong>Push 3：</strong>成功姿勢と5分割のFitは確保できていますが、把持指標の初期姿勢内変動が不足し、把持指標PLSを推定できずstrict評価が停止しました。操作用PLS自体の推定不能を意味しません。現実装は分割全体を中断するため、独立に成立し得る他の枝も未報告です。</li>
<li><strong>Scissors 5：</strong>1,380件の成功記録がありますが、seed 20260913・14・15で0.9〜1.0 radの重複除外後Fit姿勢が各8件（必要10件）。seed 20260911・12のFitは充足しています。全5分割の事前監査を通過しないタスクは評価を開始しない設定のため、5分割ともTDS評価は未実施です。</li></ul>
<p>学習中の全方策から集めた候補が全域を網羅することと、単一のBest Coverage Policyの再収集データを固定分割した後に100件を揃えられることは異なります。今回の不足は後者です。</p>
<h2>解釈と次に確認すべきこと</h2><ul>
<li>成功姿勢DBの角度網羅性は必要な検査ですが、TDSの有効性の十分条件ではありません。Scissors 3・4ではCalibrationの支持率・単調性・可動幅を分解して調べる必要があります。</li>
<li>Push 3は推定不能な把持PLSの枝のみを不成立扱いにし、残差PCAなど独立な枝を評価する設計が候補です。本報告では未実行の結果を補完していません。</li>
<li>Scissors 5は不足した高角度区間のFit初期姿勢で追加再収集を行うのが先です。分割変更や件数の水増しで不足を隠していません。</li>
<li>Coverageは把持を維持した工具状態の最小最大区間の幅です。その間の全角度に到達できることの証明ではありません。5分割は重複データを含むため独立な5回のRL学習とは扱いません。</li></ul>
<h2>実験・表示条件</h2><p>Scissors 3〜5は再学習v3、残り12タスクはv2のBest Coverage Policy。TDS実行時は全Scissorsで同じ直接FF指令を使用します。Scissorsは1自由度、Push／Sprayは把持＋操作の2自由度です。把持100Step→操作50Step→保持50Step、スコアごとに独立resetします。</p>
<p>詳細ページはlegacy／direction／range／jointの4条件を掲載し、Fit成功姿勢の散布図・Calibration応答・Test応答、字幕付きスイープ姿勢画像を表示します。Joint成立時は事前指定のTest姿勢・中央目標の20Step間隔画像も含みます。Calibration不成立のJoint画像を成功例として生成していません。</p>
<p>内部名force_plsの教師信号は環境によって接触割合・接触数の代理指標です。特にPushの値をニュートン単位の把持力として解釈しないでください。</p>
<p class="muted">本報告生成時に追加学習・追加評価や成功基準変更は行っていません。元のJSON、学習モデル、失敗ログを保存しています。</p></main></body></html>'''
    (out/'index.html').write_text(document,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='joint_metrics'},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
