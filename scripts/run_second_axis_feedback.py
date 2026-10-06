"""Run the four diagnostic tasks with isolated outputs and a Japanese report.

Use the existing simulation Python; --smoke runs 2 calibration poses, 1 test
pose and 1 target per method. No training, baseline edits or hardware commands.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from improved_tds.experiments.second_axis_feedback import DEFAULT_TASKS, dump, sha256


def pose_bootstrap(trials, baseline, width):
    """Descriptive paired interval resampling authored test pose IDs."""
    import numpy as np
    keyed={(t['pose'],t['target'],t['condition']):t for t in trials}
    by_pose={}
    for (pose,target,condition),axis2 in keyed.items():
        if condition!='axis2_fb' or not axis2['valid'] or not axis2['sensitivity_enabled']:
            continue
        other=keyed.get((pose,target,baseline))
        if other is None or not other['valid'] or (baseline=='axis1_fb' and
                                                  not other['sensitivity_enabled']):
            continue
        by_pose.setdefault(pose,[]).append(100*(axis2['abs_error']-other['abs_error'])/width)
    if not by_pose:
        return None
    means=np.array([np.mean(by_pose[p]) for p in sorted(by_pose)])
    rng=np.random.default_rng(20261006)
    resampled=np.mean(rng.choice(means,size=(10000,len(means)),replace=True),axis=1)
    low,high=np.quantile(resampled,[.025,.975])
    return dict(poses=len(means),improved_poses=int(np.count_nonzero(means<0)),
                mean_percent=float(np.mean(means)),low_percent=float(low),high_percent=float(high))


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    labels={'ff':'第1軸FF＋保持','axis1_fb':'第1軸FF＋第1軸FB','axis2_fb':'第1軸FF＋第2軸FB'}
    sections=[]
    overview=[]
    def fmt(x): return '未計測' if x is None else f'{x:.2f}'
    for task in manifest['tasks']:
        path=root/task/'report.json'
        if not path.exists():
            sections.append(f'<section><h2>{task}</h2><p>{html.escape(status[task])}</p></section>')
            continue
        data=json.loads(path.read_text(encoding='utf-8'))
        width=np.diff(data['target_range'])[0]
        unit=data['tool_unit']
        body=f'<section><h2>{task}</h2><p>ツール単位：{unit}。採用基準：{html.escape(data["source_evaluation_criterion"])}。</p>'
        body+=f'<p><a href="{task}/report.json">全試行・時系列・較正プローブのJSON</a></p>'
        for method, values in data['methods'].items():
            ff_group=values['paired_comparisons']['ff'].get('by_calibration_eligibility',{}).get('axis2_enabled',{})
            axis1_group=values['paired_comparisons']['axis1_fb'].get('by_calibration_eligibility',{}).get('both_enabled',{})
            overview.append((task,method,ff_group,axis1_group,
                             pose_bootstrap(values['trials'],'ff',width),
                             pose_bootstrap(values['trials'],'axis1_fb',width)))
            body+=f'<h3>{method.upper()}（第2軸：{values["axis2_definition"]}）</h3>'
            body+='<table><tr><th>制御条件</th><th>有効/試行</th><th>全有限観測MAE [%]</th><th>有効試行MAE [%]</th><th>観測到達幅 [%]</th><th>FB停止数</th></tr>'
            for condition,s in values['summary'].items():
                body+=f'<tr><td>{labels[condition]}</td><td>{s["valid"]}/{s["trials"]}</td><td>{fmt(s["all_finite_mae_percent"])}</td><td>{fmt(s["valid_mae_percent"])}</td><td>{fmt(s["achieved_span_percent"])}</td><td>{s["stopped_feedback"]}</td></tr>'
            body+='</table>'
            body+='<p>数値異常や途中終了を「終端誤差」として扱いません。無効試行にRL目標幅の100%以上のペナルティを与えた全試行指標も併記します（通常のMAEとは別指標）。</p>'
            body+='<table><tr><th>条件</th><th>失敗ペナルティ込み [%]</th><th>終端未観測</th><th>感度有効</th><th>実際に補正</th><th>スコア飽和</th><th>FF関節クリップ</th></tr>'
            for condition,s in values['summary'].items():
                actuated=sum(t['sensitivity_enabled'] and t['score_total_variation']>1e-6
                             for t in values['trials'] if t['condition']==condition)
                body+=f'<tr><td>{labels[condition]}</td><td>{fmt(s["failure_penalized_mae_percent"])}</td><td>{s["failed_terminal_observations"]}</td><td>{s["enabled_trials"]}</td><td>{actuated}</td><td>{s["saturated_trials"]}</td><td>{s["ff_clipped_trials"]}</td></tr>'
            body+='</table><h4>同一姿勢・目標の対応比較：第2軸FB − 対照条件</h4><table><tr><th>対照</th><th>両条件有効/対応数</th><th>誤差差分 [%]</th><th>全試行ペナルティ差分 [%]</th><th>有効→無効</th></tr>'
            for condition,s in values['paired_comparisons'].items():
                body+=f'<tr><td>{labels[condition]}</td><td>{s["jointly_valid"]}/{s["pairs"]}</td><td>{fmt(s["jointly_valid_error_delta_percent"])}</td><td>{fmt(s["all_attempt_penalized_delta_percent"])}</td><td>{s["validity_losses"]}</td></tr>'
            body+='</table><p>差分が負なら第2軸FBの誤差が小さいことを示します。有効試行のみの改善と、失敗を含む改善を区別します。</p>'
            body+='<h4>較正によるFB実行可否をそろえた比較</h4><table><tr><th>対照</th><th>較正区分</th><th>対応数</th><th>両条件有効</th><th>誤差差分 [%]</th></tr>'
            for condition,s in values['paired_comparisons'].items():
                for name,part in s.get('by_calibration_eligibility',{}).items():
                    body+=f'<tr><td>{labels[condition]}</td><td>{name}</td><td>{part["pairs"]}</td><td>{part["jointly_valid"]}</td><td>{fmt(part["jointly_valid_error_delta_percent"])}</td></tr>'
            body+='</table><p>軸同士の性能比較はboth_enabled（両軸とも較正成立）を中心に解釈します。片方だけ成立した条件は、軸の優劣ではなく実行可能性の差です。補正無効のFF同士を混ぜて効果を主張しません。</p>'
            body+='<details><summary>収束・振動・接触荷重の診断</summary><table><tr><th>姿勢ID</th><th>目標</th><th>条件</th><th>収束時間 [s]</th><th>スコア総変動量 [無次元]</th><th>ツール総変動量</th><th>最大接触法線力 [N]</th></tr>'
            for t in values['trials']:
                body+=f'<tr><td>{t["pose"]}</td><td>{t["target"]:.4f} {unit}</td><td>{labels[t["condition"]]}</td><td>{fmt(t["convergence_after_ff_seconds"])}</td><td>{fmt(t["score_total_variation"])}</td><td>{fmt(t["tool_total_variation"])} {unit}</td><td>{fmt(t["peak_normal_load_n"])}</td></tr>'
            body+='</table><p>収束時間はFF終了後、誤差がRL目標幅の3%以内に入り試行終端まで維持した最初の時刻。総変動量は振動の診断量であり、振動周波数や安全性の証明ではありません。</p></details>'
            enabled=sum(c['sensitivity']['enabled'] for c in values['calibration'])
            body+=f'<p>局所感度を採用できた条件：{enabled}/{len(values["calibration"])}。不採用時はFF指令を保持します。</p>'
            fig,axes=plt.subplots(1,2,figsize=(11,4))
            for condition,label in labels.items():
                rows=[t for t in values['trials'] if t['condition']==condition]
                xs=sorted(set(t['target'] for t in rows))
                errors=[np.mean([t['abs_error']/width*100 for t in rows if t['target']==x and t['abs_error'] is not None]) for x in xs]
                axes[0].plot(xs,errors,'o-',label=label.replace('第1軸','axis1').replace('第2軸','axis2').replace('＋','+').replace('保持','hold'))
                first=rows[0]
                axes[1].plot([s['time'] for s in first['trace']], [s['tool'] for s in first['trace']],label=condition)
            axes[0].set(xlabel=f'Target [{unit}]',ylabel='All-finite MAE [% of RL target span]')
            axes[1].set(xlabel='Time [s]',ylabel=f'Tool state [{unit}]',title='First test pose / first target')
            axes[1].axvline(5,color='grey',ls='--')
            for ax in axes: ax.grid(alpha=.25); ax.legend(fontsize=8)
            fig.tight_layout()
            name=f'{method}_comparison.png'
            fig.savefig(root/task/name,dpi=130)
            plt.close(fig)
            body+=f'<img src="{task}/{name}" alt="目標別誤差および時系列比較">'
            fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
            keyed={(t['pose'],t['target'],t['condition']):t for t in values['trials']}
            for ax,reference in zip(axes,('ff','axis1_fb')):
                by_target={}
                for (pose,target,condition),axis2 in keyed.items():
                    if condition!='axis2_fb' or not axis2['sensitivity_enabled']:
                        continue
                    baseline=keyed.get((pose,target,reference))
                    if (baseline is None or not baseline['valid'] or not axis2['valid']
                            or reference=='axis1_fb' and not baseline['sensitivity_enabled']):
                        continue
                    by_target.setdefault(target,[]).append(
                        100*(axis2['abs_error']-baseline['abs_error'])/width)
                for target,delta in by_target.items():
                    ax.scatter([target]*len(delta),delta,s=20,alpha=.45,color='#2463a5')
                if by_target:
                    ax.plot(sorted(by_target),[np.mean(by_target[x]) for x in sorted(by_target)],
                            'o-',color='#de7025',label='Paired mean')
                else:
                    ax.text(.5,.5,'No jointly valid / calibrated trials',
                            transform=ax.transAxes,ha='center',va='center',fontsize=9)
                ax.axhline(0,color='#444',ls='--',lw=1)
                ax.set(title=f'axis2 FB - {reference}',xlabel=f'Target [{unit}]',
                       ylabel='Paired absolute-error difference [% of RL target span]')
                ax.grid(alpha=.25)
                if by_target: ax.legend(fontsize=8)
            fig.tight_layout()
            pair_name=f'{method}_paired_effect.png'
            fig.savefig(root/task/pair_name,dpi=130)
            plt.close(fig)
            body+=f'<img src="{task}/{pair_name}" alt="同一目標・姿勢における第2軸FBの誤差差分">'
            body+='<details><summary>較正感度と採用範囲</summary><table><tr><th>目標</th><th>軸</th><th>採用姿勢</th><th>感度</th><th>補正スコア範囲</th><th>判定</th></tr>'
            for c in values['calibration']:
                s=c['sensitivity']
                body+=f'<tr><td>{c["target"]:.4f} {unit}</td><td>{labels[c["condition"]]}</td><td>{s["usable_poses"]}/{s["total_poses"]}</td><td>{fmt(s["slope"])} {unit}/score</td><td>[{fmt(s["offset_low"])}, {fmt(s["offset_high"])}]</td><td>{s["reason"]}</td></tr>'
            body+='</table></details>'
            body+='<details><summary>初期姿勢・目標ごとの対応比較</summary><table><tr><th>姿勢ID</th><th>目標</th><th>条件</th><th>誤差 [%]</th><th>有効</th><th>停止理由</th></tr>'
            for t in values['trials']:
                error=t['abs_error']/width*100 if t['abs_error'] is not None else None
                body+=f'<tr><td>{t["pose"]}</td><td>{t["target"]:.4f} {unit}</td><td>{labels[t["condition"]]}</td><td>{fmt(error)}</td><td>{t["valid"]}</td><td>{t["stop_reason"] or "—"}</td></tr>'
            body+='</table></details>'
        sections.append(body+'</section>')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>第2軸フィードバック比較</title><style>
body{font:16px/1.7 system-ui;background:#eef2f6;color:#182a38;margin:24px auto;max-width:1200px;padding:16px}section{background:white;padding:20px;margin:20px 0}table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #ccd4dc;padding:7px;text-align:left}img{max-width:100%}details{margin:16px 0;overflow:auto}.warn{background:#fff1d8;padding:14px}</style><h1>第2軸を用いたツール状態フィードバック</h1>'''
    if manifest['smoke']:
        page+='<p class="warn">これは実装の動作確認です。姿勢数・目標数を縮小しており、性能向上の判定や論文用の結果としては使用できません。</p>'
    page+='<p>全条件で同一の初期姿勢・シード・目標を使用し、5秒の線形FF軌道＋2秒の保持またはFBを比較します。PCA/PLSも共通逆写像区間内の同じ目標で評価します。既存の学習・成功姿勢DB・逆写像は変更していません。</p>'
    page+='<p>誤差と到達幅の分母はRL目標範囲幅です。観測到達幅は今回の共通目標で得た有効終点の最大−最小であり、独立スイープCoverageや全区間の到達保証ではありません。7秒を完走した有限な失敗試行は全有限観測MAEに含め、数値異常・途中終了は終端未観測としてJSONと有効率の分母に残します。</p>'
    page+='<p>各目標で較正姿勢を用いて感度を測るため、今回は「較正済み目標・別の初期姿勢」での評価です。未知目標への汎化とは主張しません。パラメータは固定し、Test結果による調整は行いません。MuJoCo内部状態と姿勢プールが変更した取付位置・中立指令を毎試行前に初期化しています。過去の保存結果との逐点一致ではなく、今回の3条件間の公平性を検証しています。</p>'
    page+='<p>感度は較正姿勢のみで±0.1のプローブから求めます。有効な片側応答が2姿勢以上で同符号なら使用し、関節クリップなどで評価できない側は補正範囲から除きます。gの設計上限は±0.5ですが、今回の補正は較正した±0.1を超えません。接触リンク数・接触モード・法線力[N]は診断として記録します。リンク構成の変化だけではFBを無効にせず、保持喪失・関節限界・応答符号の反転では停止します。実機動作は行いません。</p>'
    page+='<p>保存する時系列は10制御ステップごとと最終時刻です。感度推定と評価指標には全ステップのデータを使用します。</p>'
    if overview:
        page+='<section><h2>第2軸FBの効果：較正できた同一姿勢・目標での対応比較</h2>'
        page+='<table><tr><th>タスク</th><th>軸</th><th>FF比 対応有効/較正成立</th><th>FF比 誤差差分 [%]</th><th>FF比 姿勢単位95%区間</th><th>第1軸FB比 対応有効/両軸成立</th><th>第1軸FB比 誤差差分 [%]</th><th>第1軸FB比 姿勢単位95%区間</th></tr>'
        for task,method,ff_group,axis1_group,ff_interval,axis1_interval in overview:
            def interval_text(value):
                if value is None: return '未評価'
                return (f'[{value["low_percent"]:.2f}, {value["high_percent"]:.2f}] '
                        f'({value["improved_poses"]}/{value["poses"]}姿勢で改善)')
            page+=(f'<tr><td>{task}</td><td>{method.upper()}</td>'
                   f'<td>{ff_group.get("jointly_valid",0)}/{ff_group.get("pairs",0)}</td>'
                   f'<td>{fmt(ff_group.get("jointly_valid_error_delta_percent"))}</td>'
                   f'<td>{interval_text(ff_interval)}</td>'
                   f'<td>{axis1_group.get("jointly_valid",0)}/{axis1_group.get("pairs",0)}</td>'
                   f'<td>{fmt(axis1_group.get("jointly_valid_error_delta_percent"))}</td>'
                   f'<td>{interval_text(axis1_interval)}</td></tr>')
        page+='</table><p>差分＝第2軸FBの絶対誤差−対照の絶対誤差。負なら改善。分母はRL目標幅。95%区間は8つの評価初期姿勢を再抽出した記述的bootstrap（1万回）で、目標9点を独立標本とは数えません。多重比較補正や未知姿勢分布への一般化は含みません。較正不成立や無効試行の数は各タスクの表に示します。</p></section>'
    page+=''.join(sections)+'<p><a href="manifest.json">実行条件</a> / <a href="status.json">実行状態</a></p></html>'
    (root/'index.html').write_text(page,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=PROJECT/'analysis/paper15_paper_bundle_20260928')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tasks',nargs='+',choices=DEFAULT_TASKS,default=list(DEFAULT_TASKS))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args()
    root=a.output.resolve()
    if a.report_only:
        shutil.copy2(__file__,root/'report_renderer_snapshot.py')
        dump(root/'report_renderer.json',dict(script_sha256=sha256(__file__),
                                              regenerated_utc=datetime.now(timezone.utc).isoformat()))
        report(root)
        return
    if a.workers<1: p.error('workers must be positive')
    root.mkdir(parents=True,exist_ok=False)
    snapshot=root/'source_snapshot'
    shutil.copytree(PROJECT/'src',snapshot,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copy2(__file__,root/'runner_snapshot.py')
    external=PROJECT.parent/'tool-manip-gym'
    source_hashes={str(f.relative_to(external)):sha256(f) for f in external.rglob('*')
                   if f.is_file() and f.suffix in ('.py','.xml','.stl','.obj')
                   and not any(part.startswith('.') or part=='__pycache__' for part in f.relative_to(external).parts)}
    manifest=dict(protocol='second_axis_feedback_campaign_v1',tasks=a.tasks,smoke=a.smoke,
        workers=a.workers,started_utc=datetime.now(timezone.utc).isoformat(),
        bundle=str(a.bundle.resolve()),baseline_manifest_sha256=sha256(a.bundle/'manifest.json'),
        external_source_sha256=source_hashes,
        source_sha256={str(f.relative_to(snapshot)):sha256(f) for f in snapshot.rglob('*.py')},
        python=sys.executable,calibration_poses=2 if a.smoke else 8,
        test_poses=1 if a.smoke else 8,targets=1 if a.smoke else 9)
    dump(root/'manifest.json',manifest)
    status={task:'queued' for task in a.tasks}
    dump(root/'status.json',status)
    env=os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(snapshot)]+[str(external/n) for n in
        ('gym-scissors','gym-spray','gym-push','gym-slider')]),OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONIOENCODING='utf-8')
    def job(task):
        cmd=[sys.executable,'-m','improved_tds.experiments.second_axis_feedback',
             '--bundle',str(a.bundle.resolve()),'--output',str(root/task),'--task',task,
             '--calibration-poses',str(manifest['calibration_poses']),
             '--test-poses',str(manifest['test_poses']),'--targets',str(manifest['targets'])]
        with (root/f'{task}.log').open('w',encoding='utf-8') as log:
            return subprocess.run(cmd,cwd=PROJECT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures={executor.submit(job,t):t for t in a.tasks}
        status={t:'running_or_queued' for t in a.tasks}
        dump(root/'status.json',status)
        for f in as_completed(futures):
            task=futures[f]
            try: status[task]='completed' if f.result()==0 else 'failed (see log)'
            except Exception as exc: status[task]='failed: '+str(exc)
            dump(root/'status.json',status)
    report(root)
    if any(s!='completed' for s in status.values()): raise SystemExit(1)


if __name__=='__main__': main()
