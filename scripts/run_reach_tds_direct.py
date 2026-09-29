"""Run direct reconstructed-posture Scissors 3--5 TDS and build detailed HTML."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np


def page(body):
    return ('''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Direct PCA / PLS tool control</title><style>
body{font:16px/1.65 system-ui,sans-serif;background:#f3f5f8;color:#18283d;margin:0}
main{max-width:1280px;margin:auto;padding:28px}section{background:white;padding:20px;margin:18px 0;border-radius:10px}
table{border-collapse:collapse;width:100%}th,td{padding:9px;border-bottom:1px solid #dce2e9;text-align:left}
th{background:#eaf0f6}.scroll{overflow-x:auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}
img{max-width:100%;height:auto}figure{margin:12px 0}figcaption,.note{font-size:14px;color:#53647a}
.warn{border-left:5px solid #c77c19;padding:12px;background:#fff8eb}a{color:#1763a8}
details{margin:10px 0}summary{cursor:pointer;font-weight:600}
@media(max-width:800px){.pair{grid-template-columns:1fr}main{padding:12px}}</style></head><body><main>'''
            +body+'</main></body></html>')


def fmt(value,percentage=False):
    if value is None or not np.isfinite(value):
        return '未評価'
    return f'{100*value:.1f}%' if percentage else f'{value:.4f}'


def mean(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def curve(records,size,*,valid):
    output=[]
    for j in range(size):
        values=[o['achieved'] for record in records for o in [record['outcomes'][j]]
            if o is not None and np.isfinite(o['achieved'])
            and not o['failure_numerical'] and (o['valid'] or not valid)]
        output.append(float(np.mean(values)) if values else np.nan)
    return output


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    links=[]
    overview=[]
    labels=('pca_1d','pls_1d','pca_2d','pls_2d')
    for task in manifest['tasks']:
        folder=root/'results'/task
        files=sorted(folder.glob('*/report.json')) if folder.exists() else []
        if not files:
            links.append(f'<li>{html.escape(task)}: 実験中</li>')
            continue
        docs=[json.loads(file.read_text(encoding='utf-8')) for file in files]
        representative=docs[0]
        protocol=representative['protocol']
        if protocol!='reach_tds_direct_v1' or representative.get('fixed_preparation') is not False:
            raise ValueError('direct report received a legacy preparation experiment')
        results=[]
        comparisons=[]
        for label in labels:
            configs=[doc['configurations'][label] for doc in docs]
            summaries=[c['summary'] for c in configs]
            inverse=sum(s['inverse_available'] for s in summaries)
            independent=mean([s['independent_coverage'] for s in summaries])
            continuous=mean([s['continuous_coverage'] for s in summaries])
            held=mean([s['held_fraction'] for s in summaries])
            short=mean([s['short_window_fraction'] for s in summaries])
            results.append(f'<tr><td>{label.upper()}</td><td>{fmt(independent,True)}</td>'
                f'<td>{fmt(continuous,True)}</td><td>{fmt(held,True)}</td>'
                f'<td>{fmt(short,True)}</td><td>{inverse}/{len(configs)}</td></tr>')
            overview.append(dict(task=task,label=label,independent_coverage=independent,
                continuous_coverage=continuous,held_fraction=held,
                short_window_fraction=short,inverse_splits=inverse,splits=len(configs)))
            for controller in ('isotonic','ramp','database_linear_baseline'):
                trials=[t for c in configs for trial in c['test'] for t in trial['target_trials']
                    if t['controller']==controller]
                if not trials:
                    continue
                observed=[t['abs_error'] for t in trials if t['abs_error'] is not None]
                held_errors=[t['abs_error'] for t in trials if t['abs_error'] is not None
                    and t['outcome']['valid']]
                cal_rmse=[(c['selection']['isotonic_calibration_rmse'] if controller=='isotonic'
                    else c['selection']['ramp']['calibration_rmse']) for c in configs
                    if c['selection'] is not None and controller!='database_linear_baseline']
                comparisons.append(f'<tr><td>{label.upper()}</td><td>{controller}</td>'
                    f'<td>{len(observed)}/{len(trials)}</td><td>{len(held_errors)}</td>'
                    f'<td>{fmt(mean(cal_rmse))}</td><td>{fmt(mean(observed))}</td>'
                    f'<td>{fmt(mean(held_errors))}</td></tr>')
        images=[]
        for dof in (1,2):
            fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
            for ax,method in zip(axes,('pca','pls')):
                c=representative['configurations'][f'{method}_{dof}d']
                x=np.asarray(c['diagnostic_levels'])
                ax.scatter(c['fit_rho'],c['fit_tool'],s=13,alpha=.35,color='#67788e',label='200 success poses')
                for records,name,color in ((c['calibration'],'Calibration','#2570bd'),
                    ([p['diagnostic'] for p in c['test']],'Test','#df7624')):
                    ax.plot(x,curve(records,len(x),valid=True),'o-',color=color,ms=3,
                        label=name+' held 2 s')
                    ax.plot(x,curve(records,len(x),valid=False),'--',color=color,alpha=.55,
                        label=name+' all finite')
                if c['selection']:
                    sel=c['selection']
                    iso_x=sel['rho_start']+(np.asarray(sel['map_u'])+1)/2*(sel['rho_end']-sel['rho_start'])
                    ax.plot(iso_x,sel['map_tool'],color='#168e62',lw=2,label='isotonic')
                    ax.plot(sel['raw_rho'],sel['ramp']['fitted'],color='#c53148',lw=2,
                        linestyle=':',label='ramp')
                ax.set(title=f'{method.upper()} / {dof} DoF / g={c["grasp_score"]:+.2f}',
                    xlabel='Global centred score / nominal amplitude',ylabel='Tool angle [rad]')
                ax.grid(alpha=.2)
                ax.legend(fontsize=7)
            fig.tight_layout()
            name=f'direct_comparison_{dof}d.png'
            fig.savefig(folder/name,dpi=130)
            plt.close(fig)
            images.append(f'<figure><img src="{name}"><figcaption>{dof}自由度：'
                '成功姿勢DB、保持成立した校正・テスト応答、失敗を含む有限観測を区別して表示。</figcaption></figure>')
        fig,axes=plt.subplots(2,2,figsize=(12,10),sharex=True,sharey=True)
        for ax,label in zip(axes.flat,labels):
            c=representative['configurations'][label]
            ax.plot([0,1],[0,1],'k--',lw=1,label='ideal')
            for name,color,marker in (('isotonic','#168e62','o'),('ramp','#c53148','s'),
                ('database_linear_baseline','#67788e','x')):
                for valid,alpha in ((False,.3),(True,.85)):
                    trials=[t for p in c['test'] for t in p['target_trials']
                        if t['controller']==name and t['abs_error'] is not None
                        and bool(t['outcome']['valid'])==valid]
                    if trials:
                        ax.scatter([t['target'] for t in trials],[t['outcome']['achieved'] for t in trials],
                            color=color,marker=marker,alpha=alpha,s=18,
                            label=name+(' held' if valid else ' unheld'))
            ax.set(title=label.upper(),xlabel='Target angle [rad]',ylabel='Measured angle [rad]')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(folder/'direct_targets.png',dpi=130)
        plt.close(fig)
        gallery=[]
        render_dir=files[0].parent.name
        for label in labels:
            c=representative['configurations'][label]
            for kind,names in c['images'].items():
                if names:
                    gallery.append(f'<details><summary>{label.upper()} / {kind}</summary>'
                        +''.join(f'<figure><img loading="lazy" src="{render_dir}/{html.escape(name)}">'
                            f'<figcaption>{label.upper()} / {kind}: direct absolute-posture control</figcaption>'
                            '</figure>' for name in names)+'</details>')
        body=f'<a href="../../index.html">← 全体</a><h1>{html.escape(task)}：直接再構成姿勢への制御</h1>'
        body+=f'<p>{len(docs)}/{len(manifest["seeds"])}分割完了。'
        '成功姿勢200件の全体平均を基準にPCA/PLS軸を求め、各初期姿勢から絶対目標姿勢へ5秒で直接移動。'
        '固定準備オフセット・オンラインRL・状態復元・Test再試行はありません。</p>'
        body+='<section><h2>保持を伴う操作範囲</h2><div class="scroll"><table>'
        '<tr><th>方式</th><th>独立Coverage</th><th>連続Coverage</th><th>2秒保持点率</th>'
        '<th>最初5Step保持点率</th><th>逆写像成立</th></tr>'+''.join(results)+'</table></div>'
        '<p class="note">Coverageは保持成立点の観測角度範囲と0–1 radの重なりです。'
        '区間内の全角度へ到達可能とは限りません。</p></section>'
        body+='<section><h2>PCAとPLSの操作応答</h2>'+''.join(images)+'</section>'
        body+='<section><h2>目標角度誤差：等張回帰とランプ関数</h2>'
        '<p>両者は同じ校正応答・テスト初期姿勢・目標角度で比較します。'
        '逆写像が得られない分割のDB線形FF値を両方式の性能として扱いません。'
        '把持不成立の角度誤差は全観測MAEにのみ含みます。</p>'
        '<div class="scroll"><table><tr><th>方式</th><th>制御器</th><th>観測/試行</th>'
        '<th>保持成立数</th><th>校正RMSE [rad]</th><th>全観測MAE [rad]</th>'
        '<th>保持成立MAE [rad]</th></tr>'+''.join(comparisons)+'</table></div>'
        '<img src="direct_targets.png"></section>'
        body+='<section><h2>姿勢画像と20Stepごとの時系列</h2>'+''.join(gallery)+'</section>'
        body+='<section><h2>全分割の生データ</h2>'+''.join(
            f'<p><a href="{file.parent.name}/report.json">seed {file.parent.name} JSON</a></p>'
            for file in files)+'</section>'
        (folder/'index.html').write_text(page(body),encoding='utf-8')
        links.append(f'<li><a href="results/{task}/index.html">{html.escape(task)}：比較グラフ・目標誤差・画像</a>'
            f'（{len(docs)}/{len(manifest["seeds"])}分割）</li>')
    completed=sum(value=='completed' for value in status['jobs'].values())
    failed=[f'{key}: {value}' for key,value in status['jobs'].items() if value.startswith('failed')]
    body=f'<h1>Scissors 3–5：直接再構成姿勢によるPCA/PLS操作</h1>'
    body+=f'<p>進行状況：{completed}/{len(status["jobs"])}ジョブ完了。'
    body+='固定準備オフセットは使用しません。初期姿勢から再構成された絶対関節姿勢へ、各移動5秒の線形軌道で指令します。</p>'
    if failed:
        body+='<p class="warn">'+html.escape('; '.join(failed))+'</p>'
    body+='<section><h2>タスク別詳細</h2><ul>'+''.join(links)+'</ul></section>'
    if overview:
        rows=''.join(f'<tr><td>{r["task"]}</td><td>{r["label"].upper()}</td>'
            f'<td>{fmt(r["independent_coverage"],True)}</td>'
            f'<td>{fmt(r["held_fraction"],True)}</td>'
            f'<td>{r["inverse_splits"]}/{r["splits"]}</td></tr>' for r in overview)
        body+='<section><h2>全体概要</h2><div class="scroll"><table>'
        '<tr><th>タスク</th><th>方式</th><th>独立Coverage</th><th>2秒保持点率</th>'
        '<th>逆写像成立</th></tr>'+rows+'</table></div></section>'
    body+='<p><a href="status.json">進捗JSON</a> / <a href="summary.json">集計JSON</a>'
    ' / <a href="manifest.json">入力・条件</a></p>'
    (root/'summary.json').write_text(json.dumps(overview,indent=2),encoding='utf-8')
    (root/'index.html').write_text(page(body),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('runs/scissors_smooth_tds_200_20260925'))
    p.add_argument('--output',type=Path,default=Path('analysis/scissors_smooth_tds_direct_palm_20260926'))
    p.add_argument('--tasks',nargs='+',default=[f'scissors_lite_{i}' for i in (3,4,5)])
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(20260925,20260930)))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--poses',type=int,default=8)
    p.add_argument('--points',type=int,default=17)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    root=a.output.resolve()
    if a.report_only:
        build_report(root)
        return
    if root.exists():
        raise FileExistsError(root)
    if a.workers<1 or a.poses<1 or a.points<3:
        raise ValueError('workers/poses must be positive and points >=3')
    for task in a.tasks:
        with np.load(a.source_root/task/'success_postures.npz',allow_pickle=False) as db:
            meta=json.loads(db['metadata_json'].item())
            if not meta.get('complete') or len(db['q'])!=200 or not db['success'].all():
                raise ValueError(f'{task}: completed 200-success database required')
    project=Path(__file__).resolve().parents[1]
    root.mkdir(parents=True)
    (root/'results').mkdir()
    shutil.copytree(project/'src',root/'source_snapshot')
    shutil.copy2(__file__,root/'runner_snapshot.py')
    inputs=root/'inputs'
    (inputs/'pose_pool').mkdir(parents=True)
    hashes={}
    for task in a.tasks:
        (inputs/task).mkdir()
        for source,destination in ((a.source_root/task/'success_postures.npz',inputs/task/'success_postures.npz'),
                                   (a.source_root/'pose_pool'/f'{task}.npz',inputs/'pose_pool'/f'{task}.npz')):
            shutil.copy2(source,destination)
            hashes[str(source.resolve())]=hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest=dict(protocol='reach_tds_direct_v1',tasks=a.tasks,seeds=a.seeds,workers=a.workers,
        poses_per_split=a.poses,points=a.points,source_root=str(a.source_root.resolve()),
        input_sha256=hashes,started_utc=datetime.now(timezone.utc).isoformat(),
        trajectory_seconds=5.,fixed_preparation=False)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    env=os.environ.copy()
    env.update(PYTHONPATH=str(root/'source_snapshot'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env['PYTHONPATH']+=os.pathsep+os.pathsep.join(str(project.parent/'tool-manip-gym'/name)
        for name in ('gym-scissors','gym-push','gym-spray','gym-slider'))
    jobs=[(task,seed) for seed in a.seeds for task in a.tasks]
    status={f'{task}/{seed}':'queued' for task,seed in jobs}
    lock=threading.Lock()
    def save():
        (root/'status.json').write_text(json.dumps(dict(
            utc=datetime.now(timezone.utc).isoformat(),jobs=status),indent=2),encoding='utf-8')
    def run(job):
        task,seed=job
        folder=root/'results'/task/str(seed)
        folder.mkdir(parents=True)
        with lock:
            status[f'{task}/{seed}']='running'
            save()
        cmd=[sys.executable,'-m','improved_tds.experiments.reach_tds_direct',
            '--source-root',str(inputs),'--task',task,'--seed',str(seed),
            '--output',str(folder),'--poses',str(a.poses),'--points',str(a.points)]
        if seed==a.seeds[0]:
            cmd.append('--render')
        with (folder/'run.log').open('w',encoding='utf-8') as out,\
             (folder/'error.log').open('w',encoding='utf-8') as err:
            subprocess.run(cmd,cwd=project,env=env,stdout=out,stderr=err,check=True)
    save()
    build_report(root)
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures={executor.submit(run,job):job for job in jobs}
        for future in as_completed(futures):
            task,seed=futures[future]
            try:
                future.result()
                value='completed'
            except Exception as exc:
                value='failed: '+str(exc)
            with lock:
                status[f'{task}/{seed}']=value
                save()
                build_report(root)
    build_report(root)
    if any(value!='completed' for value in status.values()):
        raise SystemExit(1)


if __name__=='__main__':
    main()
