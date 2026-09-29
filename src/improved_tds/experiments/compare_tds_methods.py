"""PCA and PLS side-by-side, with honest missing/rejected status and common axes."""
import argparse
import html
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();out=root/'comparison';out.mkdir(exist_ok=True)
    manifest=json.loads((root/'manifest.json').read_text());tasks=[t['name'] for t in manifest['tasks']]
    grouped={t:[] for t in tasks}
    for path in sorted((root/'reports').glob('*_seed*.json')):
        d=json.loads(path.read_text());grouped[path.stem.split('_seed')[0]].append(d)
    bars={m:[] for m in ('pca','pls')};rows=[];details=[];metrics=[]
    for task in tasks:
        reports=grouped[task]
        for m in ('pca','pls'):
            arms=[r['configurations'][m]['arms']['joint'] for r in reports]
            cov=[np.mean([p['held_coverage'] for p in arm['poses']]) if arm['poses'] else 0. for arm in arms]
            trials=[o for arm in arms for p in arm['poses'] for o in p['target_trials']]
            value=float(np.mean(cov)) if cov else None;bars[m].append(value)
            mae=float(np.mean([o['abs_error'] for o in trials])) if trials else None
            width=np.diff(reports[0]['target_range'])[0] if reports else 1.
            cal=sum(arm['calibrated'] for arm in arms)
            status='未実施' if not reports else 'Calibration不成立' if not cal else '実行済み'
            metrics.append(dict(task=task,method=m,reports=len(reports),calibrated=cal,coverage=value,mae=mae,
                normalized_mae=mae/width if mae is not None else None,target_trials=len(trials)))
            cv='—' if value is None else f'{100*value:.1f}%';error='—' if mae is None else f'{100*mae/width:.1f}%'
            rows.append(f'<tr><td>{task}</td><td>{m.upper()}</td><td>{status}</td><td>{cal}/{len(reports)}</td><td>{cv}</td><td>{error}</td></tr>')
        if not reports:continue
        report=next((r for r in reports if r['split_seed']==20260911),reports[0])
        lo,hi=report['target_range'];width=hi-lo
        fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True,layout='constrained')
        image_columns=[]
        for ax,m in zip(axes,('pca','pls')):
            config=report['configurations'][m];arm=config['arms']['joint']
            ax.axhspan(lo,hi,color='green',alpha=.06,label='RL target range')
            if arm['calibrated']:
                s=arm['selection'];start,end=s['rho_start'],s['rho_end'];fit=arm['fit_samples']
                ax.scatter(fit['score'],fit['tool'],s=12,alpha=.4,c='gray',label=f'Success-only Fit (n={len(fit["tool"])})')
                ax.plot(-1+2*(np.asarray(s['raw_rho'])-start)/(end-start),s['raw_tool'],'--',color='#d97706',label='Calibration (held cohort)')
                ax.plot(s['map_u'],s['map_tool'],':',color='#9a3412',label='Isotonic inverse map')
                ys=np.array([[o['achieved'] for o in p['outcomes']] for p in arm['poses']]);u=np.linspace(-1,1,13)
                ax.plot(u,ys.mean(axis=0),'o-',color='#0066a8',label='Test sweep mean (all poses)')
                ax.fill_between(u,ys.mean(axis=0)-ys.std(axis=0),ys.mean(axis=0)+ys.std(axis=0),alpha=.12,color='#0066a8')
                ax.axvline(-1,color='gray',lw=.5);ax.axvline(1,color='gray',lw=.5)
                ax.set_xlabel('Normalised operation score u; Fit: unclipped projection')
            else:
                for record in config['calibration_records']:
                    ys=np.array([[o['achieved'] for o in row] for row in record['outcomes']])
                    ax.plot(record['levels'],ys.mean(axis=0),'.-',label=f'Calibration preload,grasp={record["candidate"]}')
                ax.text(.02,.96,'Calibration rejected; no Test controller',transform=ax.transAxes,va='top',color='darkred')
                ax.set_xlabel('Physical score rho (no selected normalised interval)')
            ax.set_title(f'{task} / {m.upper()} / joint / seed {report["split_seed"]}')
            ax.set_ylabel('Tool displacement (m)' if task.startswith('push') else 'Tool angle (rad)')
            ax.grid(alpha=.2);ax.legend(fontsize=7,loc='best')
            images=[]
            for img in arm['images']:
                if not Path(img).is_file():raise FileNotFoundError(img)
                rel=os.path.relpath(img,out).replace('\\','/')
                images.append(f'<a href="{html.escape(rel)}"><img loading="lazy" src="{html.escape(rel)}" alt="{m.upper()} posture/timeline"></a>')
            image_columns.append('<div><h4>'+m.upper()+'</h4>'+(''.join(images) or '<p>Joint画像なし：Calibration不成立の場合はTest画像を作成しません。</p>')+'</div>')
        # Same y and (when coordinates have the same meaning) x scale for the pair.
        limits=[ax.get_ylim() for ax in axes];axes[0].set_ylim(min(x[0] for x in limits),max(x[1] for x in limits))
        if report['configurations']['pca']['arms']['joint']['calibrated']==report['configurations']['pls']['arms']['joint']['calibrated']:
            limits=[ax.get_xlim() for ax in axes]
            for ax in axes:ax.set_xlim(min(x[0] for x in limits),max(x[1] for x in limits))
        name=f'{task}_pca_pls.png';fig.savefig(out/name,dpi=130);plt.close(fig)
        details.append(f'<section><h2>{task}</h2><img class="plot" src="{name}" alt="PCA PLS comparison {task}"><details><summary>スイープ姿勢・20Step間隔画像</summary><div class="pair">'+''.join(image_columns)+'</div></details></section>')
    x=np.arange(len(tasks));fig,ax=plt.subplots(figsize=(14,5),layout='constrained')
    for m,offset,color in (('pca',-.2,'#0066a8'),('pls',.2,'#e28724')):
        values=np.array([np.nan if v is None else v*100 for v in bars[m]])
        ax.bar(x+offset,values,.38,label=m.upper(),color=color)
        for i,value in enumerate(values):
            if np.isnan(value):ax.text(i+offset,2,'N/A',rotation=90,ha='center',fontsize=7)
    ax.set(ylim=(0,105),ylabel='Held coverage (%)',title='Joint control: PCA vs PLS (mean across available splits)')
    ax.set_xticks(x,tasks,rotation=50,ha='right');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.savefig(out/'coverage_comparison.png',dpi=130);plt.close(fig)
    completed=sum(len(v)==5 for v in grouped.values());count=sum(len(v) for v in grouped.values())
    grasp=manifest.get('adaptive_grasp_method','force_pls')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PCA・PLS TDS比較</title><style>body{font-family:system-ui,"Yu Gothic",sans-serif;max-width:1400px;margin:28px auto;padding:20px;color:#183047}p{line-height:1.8}.notice{background:#fff1d6;padding:16px}.plot{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pair img{width:100%}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ccc;padding:8px;text-align:left}th{background:#183047;color:white}section{margin-top:40px}@media(max-width:800px){.pair{grid-template-columns:1fr}}</style>
<h1>PCA・PLSによるTDS評価：直接比較</h1>'''
    page+=f'<p class="notice">評価済み：{completed}/15タスク、{count}/75分割。未実施はN/A、Calibration不成立はCoverage 0として区別します。</p>'
    page+=f'<p>操作軸PCAとPLSは同一の成功姿勢100件（角度10区間×10件）から算出。Push/Sprayの把持軸は <strong>{grasp}</strong>、Scissorsは1自由度です。左右のグラフは工具状態の縦軸を共通化し、横軸の意味が同じ場合は横軸範囲も共通化しています。</p>'
    page+='<p>灰色点＝Fitに使った成功姿勢、破線＝Calibration応答、青線＝Test操作応答。Calibration不成立時は物理スコア上の候補応答を表示し、存在しないTest曲線を補いません。Coverageは観測最小最大区間であり、中間全角度への到達を保証しません。MAEは実行できた制御器の試行のみの条件付き指標です。</p>'
    if (root/'summary/index.html').exists():page+='<p><a href="../summary/index.html">4条件すべての詳細グラフ</a></p>'
    page+='<img class="plot" src="coverage_comparison.png" alt="PCA PLS coverage comparison"><table><tr><th>タスク</th><th>操作軸</th><th>状態</th><th>Calibration成立</th><th>Coverage</th><th>正規化MAE</th></tr>'+''.join(rows)+'</table>'+''.join(details)+'</html>'
    (out/'index.html').write_text(page,encoding='utf-8')
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    print(f'Comparison HTML: {out/"index.html"}',flush=True)


if __name__=='__main__':main()
