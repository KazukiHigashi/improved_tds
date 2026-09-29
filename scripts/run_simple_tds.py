"""Four independent workers, frozen source, incremental HTML; no interactive monitoring."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for task in sorted((root/'results').iterdir()):
        files = sorted(task.glob('*/report.json'))
        if not files:
            continue
        docs = [json.loads(p.read_text()) for p in files]
        cells = []
        details = []
        for method in ('pca', 'pls'):
            configs = [d['configurations'][method] for d in docs]
            trials = [t for c in configs for p in c['test'] for t in p['target_trials']]
            errors = [t['abs_error'] for t in trials if t['abs_error'] is not None]
            valid = sum(t['outcome'] is not None and t['outcome']['valid'] for t in trials)
            width = docs[0]['target_range'][1]-docs[0]['target_range'][0]
            mae = f'{np.mean(errors)/width*100:.1f}%' if errors else '未評価'
            ncal = sum(c['selection'] is not None for c in configs)
            cells.append(f"{np.mean([c['mean_coverage'] for c in configs])*100:.1f}% / {mae} / {ncal}/{len(configs)}")
            details.append(f'<p>{method.upper()}: 逆写像の観測誤差 {len(errors)}/{len(trials)}試行; '
                           f'2秒把持成立 {valid}/{len(trials)}。未到達試行を誤差0としていません。</p>')
        # Representative split is fixed in advance, never chosen by performance.
        d = docs[0]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), sharex=True, sharey=True)
        for ax, method in zip(axes, ('pca','pls')):
            c = d['configurations'][method]
            xs = np.array(c['diagnostic_levels'])
            ax.scatter(c['fit_rho'], c['fit_tool'], s=9, alpha=.3, color='gray', label='Fit100 success postures (projection)')
            for label, records, color in [('Calibration',c['calibration'],'tab:blue'),
                                          ('Test', [p['diagnostic'] for p in c['test']], 'tab:orange')]:
                means=[]
                rejected_x=[]; rejected_y=[]
                for j in range(len(xs)):
                    vals=[r['outcomes'][j]['achieved'] for r in records
                          if r['outcomes'][j] is not None and r['outcomes'][j]['valid']]
                    means.append(np.mean(vals) if vals else np.nan)
                    for r in records:
                        item=r['outcomes'][j]
                        if item is not None and not item['valid'] and np.isfinite(item['achieved']):
                            rejected_x.append(xs[j]);rejected_y.append(item['achieved'])
                ax.plot(xs, means, 'o-', label=label+' held 2s', color=color, markersize=3)
                ax.scatter(rejected_x,rejected_y,marker='x',s=8,alpha=.15,color=color,
                           label=label+' observed but invalid')
            n=len(c['levels'])//2
            for indices, label, style in [(range(n),'Continuous forward','--'),(range(n,2*n),'Continuous reverse',':')]:
                vals=[]
                for j in indices:
                    vv=[p['outcomes'][j]['achieved'] for p in c['test']
                        if p['outcomes'][j] is not None and p['outcomes'][j]['valid']]
                    vals.append(np.mean(vv) if vv else np.nan)
                ax.plot(np.array(c['levels'])[list(indices)], vals, style, label=label, linewidth=2)
            ax.axhspan(*d['target_range'], alpha=.05, color='green')
            ax.set_title(method.upper()+' / fixed open-loop preparation')
            ax.set_xlabel('Physical synergy score rho (not fixed to [-1,1])')
            ax.set_ylabel('Tool state (m for Push; rad otherwise)')
            ax.grid(alpha=.2);ax.legend(fontsize=7)
            if not any(p['preparation']['valid'] for p in c['test']):
                ax.text(.02,.97,'No valid fixed preparation on Test poses',transform=ax.transAxes,
                        va='top',fontsize=8,color='red')
        fig.tight_layout();fig.savefig(task/'comparison.png',dpi=130);plt.close(fig)
        audit_path=files[0].parent/'hold_audit.json'
        audit='保持監査は処理中または未実施です。'
        if audit_path.exists():
            aa=json.loads(audit_path.read_text()); reached=sum(x['rl_stable_success'] for x in aa)
            audit=(f'RL安定成功 {reached}/{len(aa)}、成功後の指令固定2秒：'
                   f'角度・把持等の成功維持 {sum(x["frozen_hold_success"] is True for x in aa)}/{reached}、'
                   f'把持維持 {sum(x["frozen_grasp_success"] is True for x in aa)}/{reached}。'
                   f'対照のRL継続2秒：成功維持 {sum(x.get("continued_rl_hold_success") is True for x in aa)}、'
                   f'把持維持 {sum(x.get("continued_rl_grasp_success") is True for x in aa)}。'
                   '同一seedからの独立対照であり、失敗救済の再試行ではありません。')
        gallery=''.join(f'<details><summary>{m.upper()}：20ステップごとの往復スイープ画像</summary>'+''.join(
            f'<img loading="lazy" src="{files[0].parent.name}/{html.escape(im)}">'
            for im in d['configurations'][m]['images'])+'</details>' for m in ('pca','pls'))
        content=(f'<h1>{html.escape(task.name)}</h1><p>{len(docs)}/5分割完了。代表図は固定の先頭分割。</p>'
                 '<p>Fit100は既存の成功姿勢のみ。RL下の短時間成功と、FFでの2秒保持は別指標です。'
                 '灰色点は成功姿勢の線形射影であり、指令応答そのものではありません。'
                 '線は把持成立例の平均なので、姿勢別結果もJSONで確認してください。</p>'
                 '<img src="comparison.png">'+''.join(details)+f'<p>{audit}</p>'+gallery+
                 '<h2>全分割の記録</h2>'+''.join(f'<p><a href="{p.parent.name}/report.json">{p.parent.name}</a></p>' for p in files))
        (task/'index.html').write_text(page(content),encoding='utf-8')
        rows.append(f'<tr><td><a href="results/{task.name}/index.html">{task.name}</a></td>'
                    f'<td>{len(docs)}/5</td><td>{cells[0]}</td><td>{cells[1]}</td></tr>')
    content=('<h1>単純な1／2自由度FF制御：再実験</h1>'
             '<p>固定把持準備 → 操作スコアの往復。オンラインRL・状態復元・Testごとの把持選択・再試行なし。</p>'
             '<p>把持準備の方向・時間はCalibrationのみで選択し、PCA/PLSで共通。Scissorsは操作1自由度、'
             'Push/Sprayは把持方向と操作方向の2自由度（本評価の把持スコアは1に固定）。'
             'いずれも固定の把持準備動作を含み、全運動を1軸で表現する主張ではありません。</p>'
             '<p>各欄：連続往復Coverage / 正規化MAE / 逆写像成立分割数。'
             'Coverageは実行されたスイープの把持成立角度の最小最大範囲で、全角度の可到達性の証明ではありません。'
             '初期化失敗は手順全体のCoverage 0、逆写像不成立の誤差は未評価。</p>'
             '<table><tr><th>Task</th><th>完了</th><th>PCA</th><th>PLS</th></tr>'+''.join(rows)+'</table>')
    (root/'index.html').write_text(page(content),encoding='utf-8')


def page(content):
    return '<!doctype html><html lang="ja"><meta charset="utf-8"><title>Simple TDS</title><style>body{font:16px sans-serif;margin:2em;line-height:1.6}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}img{max-width:100%}</style>'+content+'</html>'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',default=' '.join(f'{c}_lite_{i}' for c in ('scissors','push','spray') for i in range(1,6)))
    p.add_argument('--seeds',default='20260911 20260912 20260913 20260914 20260915')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();root=a.output.resolve()
    if a.report_only:
        report(root);return
    if root.exists():raise FileExistsError(root)
    project=Path(__file__).resolve().parents[1]
    root.mkdir(parents=True);(root/'results').mkdir()
    shutil.copytree(project/'src',root/'source_snapshot')
    shutil.copy2(__file__,root/'runner_snapshot.py')
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'source_snapshot'),OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    jobs=[(task,int(seed)) for task in a.tasks.split() for seed in a.seeds.split()]
    import hashlib
    input_hashes={}
    for task in a.tasks.split():
        for f in [a.source_root/'databases'/f'{task}.npz',a.source_root/'inputs'/task/'policy.zip']:
            with f.open('rb') as stream:
                input_hashes[str(f.resolve())]=hashlib.file_digest(stream,'sha256').hexdigest()
    manifest=dict(source_root=str(a.source_root.resolve()),jobs=jobs,workers=a.workers,
        input_sha256=input_hashes,
        started_utc=datetime.now(timezone.utc).isoformat(),online_rl=False,state_restoration=False,
        fit_source='existing successful Fit100 and frozen axes',preparation_candidates='Cal success commands + zero',
        preparation_durations_seconds=[.04,.2,.4],preparation_hold_seconds=1,operation_hold_seconds=2,
        thresholds='unchanged environment physical criteria; selection uses existing 75% cohort')
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    statuses={f'{task}/{seed}':'queued' for task,seed in jobs}
    def save():
        (root/'status.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),jobs=statuses),indent=2),encoding='utf-8')
    def run(job):
        task,seed=job;folder=root/'results'/task/str(seed);folder.mkdir(parents=True)
        cmd=[sys.executable,'-m','improved_tds.experiments.simple_synergy','--source-root',str(a.source_root.resolve()),
             '--task',task,'--seed',str(seed),'--output',str(folder)]
        if seed==20260911:cmd+=['--audit-policy','--render']
        with (folder/'run.log').open('w',encoding='utf-8') as out,(folder/'error.log').open('w',encoding='utf-8') as err:
            subprocess.run(cmd,cwd=project,env=env,stdout=out,stderr=err,check=True)
    save()
    report(root)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        pending={executor.submit(run,j):j for j in jobs}
        for future in as_completed(pending):
            task,seed=pending[future]
            try:future.result();statuses[f'{task}/{seed}']='completed'
            except Exception as exc:statuses[f'{task}/{seed}']='failed: '+str(exc)
            save()
            try:report(root)
            except Exception as exc:(root/'report_error.log').write_text(str(exc),encoding='utf-8')
    report(root)


if __name__=='__main__':main()
