"""Evaluate whole-span Scissors-4 fits on unseen synergy-score midpoints."""

from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from datetime import datetime,timezone
import html
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files=sorted((root/'splits').glob('*.json'))
    docs=[json.loads(path.read_text(encoding='utf-8')) for path in files]
    if not docs:
        raise ValueError('no split reports')
    summary={}
    pictures=[]
    for method in ('pca','pls'):
        trials=[t for doc in docs for t in doc['methods'][method]['trials']]
        metrics={}
        for controller in ('linear','isotonic','ramp'):
            finite=[abs(t['predicted'][controller]-t['achieved']) for t in trials
                    if t['achieved'] is not None]
            held=[abs(t['predicted'][controller]-t['achieved']) for t in trials
                  if t['achieved'] is not None and t['valid']]
            metrics[controller]=dict(trials=len(trials),observed=len(finite),held=len(held),
                observed_mae=float(np.mean(finite)) if finite else None,
                held_mae=float(np.mean(held)) if held else None)
        groups=[]
        levels=docs[0]['methods'][method]['unseen_scores']
        for j,score in enumerate(levels):
            group=[t for t in trials if t['midpoint_index']==j]
            good=[t for t in group if t['valid'] and t['achieved'] is not None]
            groups.append(dict(score=score,trials=len(group),held=len(good),
                achieved_mean=float(np.mean([t['achieved'] for t in good])) if good else None,
                predicted={name:float(np.mean([t['predicted'][name] for t in group]))
                           for name in metrics}))
        summary[method]=dict(metrics=metrics,by_unseen_score=groups,
            calibration_held=sum(doc['methods'][method]['fit']['calibration_held'] for doc in docs),
            calibration_total=sum(doc['methods'][method]['fit']['calibration_total'] for doc in docs))
        fig,axes=plt.subplots(2,1,figsize=(10,8),sharex=True,height_ratios=[3,1])
        ax,rate=axes
        for label,color in (('linear','#2674b8'),('isotonic','#15916b'),('ramp','#ce7135')):
            ax.plot([g['score'] for g in groups],
                    [g['predicted'][label] for g in groups],'-o',ms=3,color=color,
                    label=label+' fitted prediction')
        ax.scatter([g['score'] for g in groups if g['achieved_mean'] is not None],
                   [g['achieved_mean'] for g in groups if g['achieved_mean'] is not None],
                   color='black',s=40,label='unseen score: held outcome mean')
        ax.set_ylabel('Scissors angle [rad]')
        ax.grid(alpha=.3);ax.legend(fontsize=8)
        rate.bar([g['score'] for g in groups],
                 [g['held']/g['trials'] for g in groups],width=.7*float(levels[1]-levels[0]),
                 color='#758397')
        rate.set(xlabel='Physical synergy score (unseen midpoint)',ylabel='Held fraction',ylim=(0,1.05))
        rate.grid(axis='y',alpha=.3)
        fig.suptitle(f'Scissors 4 / {method.upper()}: full-score fit and held-out midpoint trials')
        fig.tight_layout()
        filename=f'{method}_fullscore_unseen.png'
        fig.savefig(root/filename,dpi=145)
        plt.close(fig)
        pictures.append(filename)
    summary['splits']=[doc['seed'] for doc in docs]
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    rows=[]
    for method in ('pca','pls'):
        result=summary[method]
        for controller,metric in result['metrics'].items():
            rows.append(f'<tr><td>{method.upper()}</td><td>{controller}</td>'
                f'<td>{metric["observed"]}/{metric["trials"]}</td>'
                f'<td>{metric["held"]}/{metric["trials"]}</td>'
                f'<td>{metric["observed_mae"]:.4f}</td>'
                f'<td>{metric["held_mae"]:.4f}</td></tr>')
    body='<h1>Scissors 4: 全スコア域フィットと未知スコア検証</h1>'
    body+='<p>各分割の17個のCalibrationスコアについて、2秒保持できた角度観測をすべて採用。'
    body+='低スコア側の再保持領域を含み、不安定な点は成功値として補わない。'
    body+='線形補間、等張回帰、ランプ関数を同じCalibrationデータにフィットした。'
    body+='未知スコアは隣接Calibration点の中点16個で、未使用の8初期姿勢から5秒の直接軌道＋2秒保持で検証した。</p>'
    body+='<p><strong>注意：</strong>保持失敗が密集する区間をまたぐ予測は補間・単調化による推定であり、'
    body+='連続した物理的な逆写像の存在を保証しない。MAEは「スコア→角度」の順写像予測誤差である。</p>'
    body+='<table><thead><tr><th>軸</th><th>フィット</th><th>角度観測</th><th>保持成立</th>'
    body+='<th>全観測MAE [rad]</th><th>保持成立時MAE [rad]</th></tr></thead><tbody>'
    body+=''.join(rows)+'</tbody></table>'
    for method,filename in zip(('PCA','PLS'),pictures):
        body+=f'<figure><img src="{filename}" style="max-width:100%"><figcaption>{method}: '
        body+='未知スコアの予測と実測、スコア別保持率。</figcaption></figure>'
    body+='<p><a href="summary.json">集計JSON</a> / '
    body+=' / '.join(f'<a href="splits/{path.name}">{html.escape(path.stem)}分割</a>' for path in files)
    body+='</p>'
    style='body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:auto;padding:25px;color:#172b42}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}th{background:#eaf0f6}figure{margin:25px 0}figcaption{color:#546778}'
    (root/'index.html').write_text('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<title>Scissors 4 full-score fit</title><style>'+style+'</style></head><body>'+
        body+'</body></html>',encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-analysis',type=Path,required=True)
    p.add_argument('--source-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    root=a.output.resolve()
    if a.workers<1:raise ValueError('workers must be positive')
    root.mkdir(parents=True,exist_ok=True)
    (root/'splits').mkdir(exist_ok=True)
    manifest=dict(protocol='scissors4_fullscore_unseen_midpoints_v1',
        source_analysis=str(a.source_analysis.resolve()),source_run=str(a.source_run.resolve()),
        seeds=list(range(20260925,20260930)),workers=a.workers,
        calibration_levels=17,unseen_midpoints=16,calibration_poses=8,test_poses=8,
        trajectory_seconds=5.,hold_seconds=2.,started_utc=datetime.now(timezone.utc).isoformat())
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    env=os.environ.copy()
    project=Path(__file__).resolve().parents[1]
    env['PYTHONPATH']=os.pathsep.join([str(project/'src'),
        str(project.parent/'tool-manip-gym/gym-scissors')])
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    seeds=manifest['seeds']
    status={str(seed):('completed' if (root/'splits'/f'{seed}.json').is_file() else 'queued')
            for seed in seeds}
    def save():
        (root/'status.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),
            jobs=status),indent=2),encoding='utf-8')
    save()
    def run(seed):
        output=root/'splits'/f'{seed}.json'
        cmd=[sys.executable,'-m','improved_tds.experiments.evaluate_scissors4_fullscore_fit',
             '--source-analysis',str(a.source_analysis.resolve()),
             '--source-run',str(a.source_run.resolve()),'--seed',str(seed),'--output',str(output)]
        with (root/'splits'/f'{seed}.log').open('w',encoding='utf-8') as out, \
                (root/'splits'/f'{seed}.err.log').open('w',encoding='utf-8') as err:
            subprocess.run(cmd,cwd=project,env=env,stdout=out,stderr=err,check=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(run,seed):seed for seed in seeds if status[str(seed)]!='completed'}
        for seed in futures.values():status[str(seed)]='running'
        save()
        for future in as_completed(futures):
            seed=futures[future]
            try:future.result();status[str(seed)]='completed'
            except Exception as exc:status[str(seed)]='failed: '+str(exc)
            save()
    if any(value!='completed' for value in status.values()):
        raise RuntimeError(f'unknown-score evaluation incomplete: {status}')
    summarize(root)


if __name__=='__main__':
    main()
