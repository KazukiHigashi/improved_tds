"""Frozen, four-worker Scissors reach-TDS campaign and offline HTML report."""
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
    return '''<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Scissors PCA / PLS TDS</title>
<style>body{font:16px/1.7 system-ui,sans-serif;background:#f4f6fa;color:#17283f;margin:0}
main{max-width:1250px;margin:auto;padding:32px}h1,h2,h3{line-height:1.3}a{color:#1762a4}
section,.card{background:white;padding:22px;margin:20px 0;border-radius:10px}
table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #d8e0e9;text-align:left}
th{background:#e9f0f7}img{max-width:100%;height:auto}figure{margin:15px 0}figcaption,.muted{color:#526478;font-size:14px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}.warn{border-left:5px solid #c17d12;padding:12px;background:#fff8e8}
details{background:white;margin:12px 0;padding:12px}summary{cursor:pointer;font-weight:600}
@media(max-width:800px){.pair{grid-template-columns:1fr}main{padding:14px}.scroll{overflow-x:auto}}</style>
</head><body><main>'''+body+'</main></body></html>'


def fmt(value, percent=False):
    if value is None or not np.isfinite(value): return '未評価'
    return f'{value*100:.1f}%' if percent else f'{value:.4f}'


def mean_finite(values):
    vals=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def curves(records, size, valid=True, short=False):
    values=[]
    for j in range(size):
        vv=[]
        for record in records:
            x=record['outcomes'][j]
            if x is not None and x.get('failure_numerical',False) and not short:
                continue
            if x is not None and short: x=x.get('short_window')
            if x is not None and (x['valid'] or not valid) and np.isfinite(x['achieved']):
                vv.append(x['achieved'])
        values.append(np.mean(vv) if vv else np.nan)
    return np.array(values)


def build_report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    status=json.loads((root/'status.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    tasklinks=[];allrows=[];aggregate=[]
    for task in manifest['tasks']:
        folder=root/'results'/task
        files=sorted(folder.glob('*/report.json')) if folder.exists() else []
        if not files:
            tasklinks.append(f'<li>{task}: 実験中／未完了（結果はまだありません）</li>');continue
        docs=[json.loads(f.read_text(encoding='utf-8')) for f in files]
        rep=docs[0]
        rows=[]
        for label in ('pca_1d','pls_1d','pca_2d','pls_2d'):
            configs=[d['configurations'][label] for d in docs]
            summaries=[c['summary'] for c in configs]
            targets=[t for c in configs for p in c['test'] for t in p['target_trials']]
            diagnostics=[o for c in configs for p in c['test'] for o in p['diagnostic']['outcomes'] if o is not None]
            observed=[o['achieved'] for o in diagnostics if np.isfinite(o['achieved'])]
            preparations=[p['preparation'] for c in configs for p in c['test']]
            errors=[t['abs_error'] for t in targets if t['abs_error'] is not None]
            held=[t['abs_error'] for t in targets if t['abs_error'] is not None and t['outcome']['valid']]
            inv=sum(c['selection'] is not None for c in configs)
            row=dict(task=task,label=label,continuous=mean_finite([s['continuous_coverage'] for s in summaries]),
                independent=mean_finite([s['independent_coverage'] for s in summaries]),
                short=mean_finite([s['short_window_coverage'] for s in summaries]),
                held_fraction=mean_finite([s['held_fraction'] for s in summaries]),
                mae=mean_finite(errors),held_mae=mean_finite(held),inverse_splits=inv,
                splits=len(configs),errors=len(errors),held_errors=len(held),targets=len(targets),
                preparation_held=sum(bool(p['valid']) for p in preparations),
                preparation_trials=len(preparations),
                nonzero_grasp_splits=sum(abs(c['grasp_score'])>1e-9 for c in configs),
                observed_angle_min=float(min(observed)) if observed else None,
                observed_angle_max=float(max(observed)) if observed else None,
                mean_hold_fraction=mean_finite([o['hold_fraction'] for o in diagnostics]),
                final_held_fraction=mean_finite([float(o['held']) for o in diagnostics]),
                numerical_failures=sum(bool(o['failure_numerical']) for o in diagnostics),
                diagnostic_observations=len(diagnostics))
            aggregate.append(row)
            cells=(f'<td>{label.upper()}</td><td>{fmt(row["independent"],True)}</td>'
                f'<td>{fmt(row["continuous"],True)}</td><td>{fmt(row["short"],True)}</td>'
                f'<td>{fmt(row["held_fraction"],True)}</td><td>{fmt(row["mae"])} / {fmt(row["held_mae"])}</td>'
                f'<td>{len(errors)} / {len(held)} / {len(targets)}</td><td>{inv}/{len(configs)}</td>')
            rows.append('<tr>'+cells+'</tr>')
            allrows.append(f'<tr><td><a href="results/{task}/index.html">{task}</a></td>'+cells+'</tr>')
        figures=[]
        for dof in (1,2):
            fig,axes=plt.subplots(2,2,figsize=(14,9))
            for col,method in enumerate(('pca','pls')):
                c=rep['configurations'][f'{method}_{dof}d'];x=np.array(c['diagnostic_levels'])
                ax=axes[0,col]
                ax.scatter(c['fit_rho'],c['fit_tool'],s=12,alpha=.4,c='#65758b',label='Fit200 success DB (projection)')
                for name,rr,color in [('Calibration',c['calibration'],'#2675c9'),
                                     ('Test',[p['diagnostic'] for p in c['test']],'#e27724')]:
                    ax.plot(x,curves(rr,len(x)), 'o-',ms=3,color=color,label=name+' held 2s')
                    ax.plot(x,curves(rr,len(x),valid=False), '--',lw=1,alpha=.6,color=color,label=name+' all finite observations')
                    ax.plot(x,curves(rr,len(x),short=True), ':',alpha=.7,color=color,label=name+' first 5 hold steps')
                    bad=[(x[j],t['achieved']) for r in rr for j,t in enumerate(r['outcomes'])
                         if t is not None and not t['valid'] and not t.get('failure_numerical',False)]
                    if bad: ax.scatter(*np.array(bad).T,marker='x',s=10,alpha=.2,color=color,label=name+' invalid')
                if c['selection']:
                    sel=c['selection'];ax.axvspan(min(sel['rho_start'],sel['rho_end']),max(sel['rho_start'],sel['rho_end']),alpha=.08,color='green')
                    iso_rho=sel['rho_start']+(np.asarray(sel['map_u'])+1)/2*(sel['rho_end']-sel['rho_start'])
                    ax.plot(iso_rho,sel['map_tool'],color='#198754',lw=2,label='Isotonic calibration')
                    ax.plot(sel['raw_rho'],sel['ramp']['fitted'],color='#bb2d3b',lw=2,ls='--',label='Ramp calibration')
                ax.axhline(0,color='gray',lw=.5);ax.axhline(1,color='gray',lw=.5)
                ax.set(title=f'{method.upper()} / {dof} DoF / g={c["grasp_score"]:+.2f}',
                       xlabel='Physical operation score rho',ylabel='Tool angle [rad]')
                ax.grid(alpha=.2);ax.legend(fontsize=7)
                ax=axes[1,col]
                for field,name,color in [('command_q','Requested hand posture','#2675c9'),('measured_q','Measured hand posture','#e27724')]:
                    move=[]
                    for j in range(len(x)):
                        vals=[]
                        for test in c['test']:
                            outcomes=test['diagnostic']['outcomes'];ref=outcomes[len(x)//2];now=outcomes[j]
                            if ref is not None and now is not None and not now.get('failure_numerical',False) and not ref.get('failure_numerical',False):
                                vals.append(np.sqrt(np.mean((np.array(now[field])-ref[field])**2)))
                        move.append(np.mean(vals) if vals else np.nan)
                    ax.plot(x,move,'o-',ms=3,color=color,label=name)
                ax.set(xlabel='Physical operation score rho',ylabel='RMS joint change from middle score [rad]',title='Motion audit (invalid physical trials included)')
                ax.grid(alpha=.2);ax.legend(fontsize=8)
            fig.tight_layout();name=f'comparison_{dof}d.png';fig.savefig(folder/name,dpi=130);plt.close(fig)
            figures.append(f'<h3>{dof}自由度：PCAとPLSの比較</h3><figure><img src="{name}"><figcaption>'
                '上：同じ成功姿勢100点の射影、Calibration応答、Test応答。×は把持・保持条件を満たさない観測。'
                '破線は保持不成立も含む有限観測平均、点線は保持開始後5Stepのみの補助評価。下：指令姿勢と実際のハンド姿勢がスコアに応じて変わるかの確認。'
                '欠測区間は補間せず、空白のまま表示しています。</figcaption></figure>')
        fig,axes=plt.subplots(2,2,figsize=(14,8),sharex=True,sharey=True)
        for ax,label in zip(axes.flat,('pca_1d','pls_1d','pca_2d','pls_2d')):
            c=rep['configurations'][label]
            ax.plot([0,1],[0,1],'k--',lw=1,label='Ideal')
            for controller,color,marker in [('isotonic','#198754','o'),('ramp','#bb2d3b','s'),
                                            ('database_linear_baseline','#7a8795','x')]:
                for valid,alpha in [(False,.3),(True,.85)]:
                    ts=[t for p in c['test'] for t in p['target_trials'] if t['controller']==controller
                        and t['abs_error'] is not None and bool(t['outcome']['valid'])==valid]
                    if ts:ax.scatter([t['target'] for t in ts],[t['outcome']['achieved'] for t in ts],
                                     color=color,marker=marker,s=20,alpha=alpha,
                                     label=controller+(' held 2s' if valid else ' not held'))
            ax.set(title=label.upper()+' / '+c['target_controller'],xlabel='Target angle [rad]',ylabel='Measured angle [rad]')
            ax.grid(alpha=.2);ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(folder/'targets.png',dpi=130);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(14,4.5),sharey=True)
        for ax,dof in zip(axes,(1,2)):
            for method,color in [('pca','#2675c9'),('pls','#e27724')]:
                c=rep['configurations'][f'{method}_{dof}d'];x=np.array(c['levels']);n=len(x)//2
                vals=curves(c['test'],len(x))
                ax.plot(x[:n],vals[:n],'-',color=color,label=method.upper()+' forward')
                ax.plot(x[n:],vals[n:],'--',color=color,label=method.upper()+' reverse')
            ax.set(title=f'{dof} DoF continuous sweep',xlabel='Physical score rho',ylabel='Tool angle [rad]');ax.grid(alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(folder/'continuous.png',dpi=130);plt.close(fig)
        gallery=[]
        # Render seed is fixed before experiments, not selected by performance.
        for dof in (1,2):
            columns=[]
            for method in ('pca','pls'):
                c=rep['configurations'][f'{method}_{dof}d'];items=[]
                subtitles={'operation_sweep':'操作シナジースイープ：各スコアで初期姿勢から独立に実行した最終画像',
                    'grasp_sweep':'残差PCA方向のスイープ：把持への寄与を確認する追加診断',
                    'continuous_every20':'連続往復スイープ：20Stepごとの画像（失敗終了時を含む）',
                    'target_every20':'目標角度0・0.5・1 radの操作：20Stepごとの画像（失敗終了時を含む）'}
                for kind,images in c['images'].items():
                    items.append(f'<details><summary>{subtitles[kind]}</summary>'+(''.join(
                        f'<figure><img loading="lazy" src="{files[0].parent.name}/{html.escape(im)}"><figcaption>{method.upper()}・{dof}自由度：{subtitles[kind]}</figcaption></figure>' for im in images) or '<p>画像なし（描画未実施）。</p>')+'</details>')
                columns.append(f'<div><h3>{method.upper()} / {dof}自由度</h3>'+''.join(items)+'</div>')
            gallery.append('<div class="pair">'+''.join(columns)+'</div>')
        header='<tr><th>方式</th><th>独立Coverage</th><th>連続Coverage</th><th>5Step補助Coverage</th><th>2秒保持率</th><th>MAE全観測 / 保持 [rad]</th><th>誤差件数 全 / 保持 / 予定</th><th>逆写像成立</th></tr>'
        body=f'<a href="../../index.html">← 全体比較</a><h1>{task}：PCA / PLS TDS評価</h1>'
        body+=f'<p>{len(docs)}/{len(manifest["seeds"])}分割完了。グラフ・画像は事前指定seed {rep["seed"]}、画像の初期姿勢はTest先頭の{rep["test_poses"][0]}番。性能に基づく画像選別はしていません。</p>'
        body+=f'<p>同一成功姿勢{rep["fit_rows"]}件。操作軸の内積 PCA·PLS = {rep["axis_cosine"]:.4f}（±1に近いほど同じ方向）。</p>'
        prep_nonzero=sum(any(candidate['valid_count']>0 for candidate in d['preparation_candidates']) for d in docs)
        body+=(f'<p class="warn">固定準備で1秒把持できた分割：{prep_nonzero}/{len(docs)}。'
               'この数が0の場合、Calibrationは把持保持条件を満たす応答を得られず、逆写像を構築できません。'
               '操作角度が変化していても有効Coverageには算入しません。</p>')
        body+='<div class="scroll"><table>'+header+''.join(rows)+'</table></div>'
        detail_rows=[]
        for r in aggregate[-4:]:
            angle=(f'{r["observed_angle_min"]:.3f}～{r["observed_angle_max"]:.3f}'
                   if r['observed_angle_min'] is not None else '欠測')
            detail_rows.append(f'<tr><td>{r["label"].upper()}</td>'
                f'<td>{r["preparation_held"]}/{r["preparation_trials"]}</td>'
                f'<td>{r["nonzero_grasp_splits"]}/{r["splits"]}</td>'
                f'<td>{fmt(r["mean_hold_fraction"],True)}</td>'
                f'<td>{fmt(r["final_held_fraction"],True)}</td>'
                f'<td>{angle}</td><td>{r["numerical_failures"]}/{r["diagnostic_observations"]}</td></tr>')
        body+=('<h2>把持準備・保持不成立の診断（全5分割）</h2><div class="scroll"><table>'
               '<tr><th>方式</th><th>Test準備1秒保持</th><th>第2軸 g≠0</th>'
               '<th>操作後2秒の平均保持時間率</th><th>終端だけ接触あり</th>'
               '<th>保持不成立を含む観測角度 [rad]</th><th>数値異常 / 観測</th></tr>'
               +''.join(detail_rows)+'</table></div>'
               '<p>終端だけ接触ありは2秒間の連続保持を意味しません。'
               '観測角度の範囲も、有効な把持操作範囲ではありません。'
               '第2軸 g=0 の分割では2自由度指令が対応する1自由度指令と同一です。</p>')
        diagnostics=[]
        for label,c in rep['configurations'].items():
            sel=c['selection']
            interval=f'{sel["rho_start"]:.3f} → {sel["rho_end"]:.3f}' if sel else '不成立'
            prep_valid=sum(p['preparation']['valid'] for p in c['test'])
            diagnostics.append(f'<tr><td>{label.upper()}</td><td>{c["grasp_score"]:+.2f}</td>'
                f'<td>{fmt(c["fit"]["operation_tool_correlation"])}</td>'
                f'<td>{fmt(c.get("calibration_held_correlation"))} / {fmt(c.get("test_held_correlation"))}</td>'
                f'<td>{prep_valid}/{len(c["test"])}</td><td>{interval}</td><td>{c["target_controller"]}</td></tr>')
        body+='<h2>代表分割の制御条件・相関</h2><div class="scroll"><table><tr><th>方式</th><th>第2軸g</th><th>Fit中心化相関</th><th>Calibration / Test実応答相関</th><th>Test準備保持</th><th>u=−1→+1の物理rho</th><th>目標入力方式</th></tr>'+''.join(diagnostics)+'</table></div>'
        body+='<p>実応答相関は物理rhoと保持成立平均角度の相関（3点未満・定数応答は未評価）。DB上で軸の符号を揃えても、接触・飽和・把持崩れによって実応答が負になる場合があります。Testの結果を見て符号を反転しません。2自由度でg=0が選ばれた場合は第2軸を使わず、対応する1自由度と同じ指令になります。</p>'
        body+='<p class="warn">Coverageは把持を保った実測角度の最小–最大幅と0–1 radの重なりです。区間内全角度の連続到達を保証しません。失敗初期姿勢も0として平均します。逆写像がない場合の目標誤差はDB線形FF基準の結果であり、較正済み逆写像の性能とは区別してください。</p>'
        body+=''.join(figures)+'<h2>目標角度と到達角度</h2><img src="targets.png"><p>終端の有限観測のみ。把持失敗も全観測MAEには含み、保持成立のみのMAEを別記。準備中終了・数値異常は欠測であり、誤差0にはしません。</p>'
        body+='<h2>連続往復スイープ</h2><img src="continuous.png"><p>途中リセットなし。空白は保持成立データなし。独立点スイープと混同しません。</p>'
        body+='<h2>姿勢画像・操作時系列</h2>'+''.join(gallery)
        body+='<h2>全分割の生データ</h2>'+''.join(f'<p><a href="{f.parent.name}/report.json">seed {f.parent.name} JSON</a></p>' for f in files)
        comparison=[]
        for label in ('pca_1d','pls_1d','pca_2d','pls_2d'):
            for controller in ('isotonic','ramp','database_linear_baseline'):
                trials=[t for doc in docs for pose in doc['configurations'][label]['test']
                        for t in pose['target_trials'] if t['controller']==controller]
                if not trials:continue
                observed=[t['abs_error'] for t in trials if t['abs_error'] is not None]
                held=[t['abs_error'] for t in trials if t['abs_error'] is not None and t['outcome']['valid']]
                calibration_rmse=[(doc['configurations'][label]['selection']['isotonic_calibration_rmse']
                    if controller=='isotonic' else doc['configurations'][label]['selection']['ramp']['calibration_rmse'])
                    for doc in docs if controller!='database_linear_baseline'
                    and doc['configurations'][label]['selection'] is not None]
                comparison.append(f'<tr><td>{label.upper()}</td><td>{controller}</td>'
                    f'<td>{len(observed)}/{len(trials)}</td><td>{len(held)}</td>'
                    f'<td>{fmt(mean_finite(calibration_rmse))}</td>'
                    f'<td>{fmt(mean_finite(observed))}</td><td>{fmt(mean_finite(held))}</td></tr>')
        body+=('<h2>Isotonic vs ramp: matched target trials</h2>'
               '<p>Both inverse models use the same calibration sweep, initial poses and target angles. '
               'The DB-only linear fallback is listed separately when physical calibration is unavailable. '
               'An observed error with no maintained grasp is not a valid manipulation result.</p>'
               '<div class="scroll"><table><tr><th>PCA/PLS</th><th>Inverse</th>'
               '<th>Observed / attempted</th><th>Held observations</th>'
               '<th>Calibration RMSE [rad]</th><th>All observed MAE [rad]</th><th>Held MAE [rad]</th></tr>'
               +''.join(comparison)+'</table></div>')
        (folder/'index.html').write_text(page(body),encoding='utf-8')
        tasklinks.append(f'<li><a href="results/{task}/index.html">{task}：グラフ・PCA/PLS横並び画像・全分割データ</a>（{len(docs)}/{len(manifest["seeds"])}）</li>')
    header='<tr><th>Task</th><th>方式</th><th>独立Coverage</th><th>連続Coverage</th><th>5Step補助Coverage</th><th>2秒保持率</th><th>MAE全 / 保持 [rad]</th><th>誤差件数 全 / 保持 / 予定</th><th>逆写像成立</th></tr>'
    charts=''
    if aggregate:
        fig,axes=plt.subplots(1,3,figsize=(18,4.5))
        labels=[f'{r["task"].replace("scissors_lite_","Sc")}/{r["label"]}' for r in aggregate]
        x=np.arange(len(labels));colors=['#2675c9' if 'pca' in r['label'] else '#e27724' for r in aggregate]
        axes[0].bar(x,[r['independent']*100 for r in aggregate],color=colors)
        axes[0].set(ylabel='Independent held coverage [%]',ylim=(0,100))
        axes[1].bar(x,[r['mae'] if r['mae'] is not None else np.nan for r in aggregate],color=colors)
        axes[1].set(ylabel='Observed target MAE [rad] (controller: see table)')
        axes[2].bar(x,[r['mean_hold_fraction']*100 if r['mean_hold_fraction'] is not None else np.nan for r in aggregate],color=colors)
        axes[2].set(ylabel='Mean fraction of 2 s with both handles held [%]')
        for ax in axes:ax.set_xticks(x,labels,rotation=65,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(root/'overview.png',dpi=140);plt.close(fig)
        charts='<img src="overview.png">'
    completed=sum(s=='completed' for s in status['jobs'].values())
    body=(f'<h1>Scissors 3–5：成功姿勢からのTDSツール操作</h1>'
          f'<p>PCA / PLS × 1 / 2自由度 ｜ 制御モード '
          f'{html.escape(manifest.get("scissors_control_version", "simple"))} ｜ '
          f'実験開始 {html.escape(manifest["started_utc"])}</p>')
    body+=f'<section><h2>進行状況：{completed}/{len(status["jobs"])}ジョブ完了</h2><ul>'+''.join(tasklinks)+'</ul></section>'
    failed=[f'{k}: {v}' for k,v in status['jobs'].items() if v.startswith('failed')]
    if failed:body+='<p class="warn">'+html.escape('; '.join(failed))+'</p>'
    if aggregate and all(r['independent']==0 and r['continuous']==0 for r in aggregate):
        body+=('<p class="warn">現在完了した条件では保持成立Coverageが0%です。'
               '保持不成立の角度やMAEはツール制御成功とは解釈できません。'
               '較正逆写像の成立数と把持準備の成立数を個別の表で確認してください。</p>')
    body+='<section><h2>実験条件と読み方</h2><p>各タスク200件の成功姿勢のみを使用。全成功姿勢の平均で中心化した実関節姿勢から、PCA第1成分またはPLS第1方向を求めます。操作軸はDB上のツール角度相関が正となる向きに揃えます。2自由度では操作軸に直交する最大残差分散方向を追加しますが、把持への寄与は仮説であり保証しません。</p>'
    body+='<p>制御：初期姿勢 → 共通の固定把持準備 → 操作軸スイープ。2自由度では準備時に第2方向のスコアgを適用し、操作中は固定します。オンラインRL・フィードバック・成功状態への復元・Testでの再試行はありません。固定準備自体は主成分1本だけで表す動作ではありません。</p>'
    body+='<p>Calibrationのみで準備・g・単調な物理スコア区間を選択。同一区間の応答へ等張回帰とランプ関数を適合させ、両方の逆写像を同じ目標角度で評価します。物理スコアrhoのスイープ端点はDBで観測した最小値×1.5と最大値×1.5です。準備と操作はそれぞれ5秒の線形軌道です。関節別クリップを用います。</p>'
    body+=f'<p>各分割Calibration {manifest["poses_per_split"]}姿勢 / Test {manifest["poses_per_split"]}姿勢。{len(manifest["seeds"])}つの固定seedで評価。同じ200件を使うため独立した複数学習ではありません。Fit200は全初期姿勢プールから収集済みであり、未知姿勢への交差検証ではありません。PCAとPLSは同じ分割・初期状態・目標・姿勢数を使用します。</p>'
    body+='<p>主評価は従来どおり、準備後1秒・操作指令後2秒の把持保持に基づくCoverageです。RLの角度許容誤差をFF成功指標にしません。今回のDBの成功保持は5Step（0.2秒）なので、操作保持の最初5StepのCoverageも補助指標として併記します。後半に落下した点でも短時間評価は成立し得ますが、長時間保持と混同しません。</p>'
    body+='<p>逆写像が成立しなくても独立点・連続往復スイープは実行します。目標誤差を省略しないため、その場合のみDBから学習した単純な「目標角度→物理スコア」の線形FF基準も実行し、方式名を明示します。Calibration棄却はPCA/PLS計算の失敗とは異なります。</p></section>'
    body+='<section><h2>主要指標（完了分の平均）</h2>'+charts+'<div class="scroll"><table>'+header+''.join(allrows)+'</table></div><p>MAEはrad。1 rad ≈57.3°。欠測は除外件数を明示し、0として平均しません。成功率の高い画像だけを選ばず、失敗も含めて表示します。</p></section>'
    body+='<p><a href="summary.json">集計JSON</a> / <a href="manifest.json">条件・入力ハッシュ</a> / <a href="status.json">ジョブ状況</a></p>'
    (root/'summary.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    (root/'index.html').write_text(page(body),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=Path('runs/scissors_smooth_tds_200_20260925'))
    p.add_argument('--control-version',choices=('simple','smooth'),default='smooth',
                   help='require this task version in all source databases')
    p.add_argument('--output',type=Path,default=Path('analysis/scissors_smooth_tds_200_20260925'))
    p.add_argument('--tasks',nargs='+',default=['scissors_lite_'+str(i) for i in (3,4,5)])
    p.add_argument('--seeds',nargs='+',type=int,default=list(range(20260925,20260930)))
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--poses',type=int,default=8)
    p.add_argument('--points',type=int,default=17)
    p.add_argument('--report-only',action='store_true')
    a=p.parse_args();root=a.output.resolve()
    if a.report_only:build_report(root);return
    if root.exists():raise FileExistsError(root)
    for task in a.tasks:
        source=a.source_root/task/'success_postures.npz'
        with np.load(source,allow_pickle=False) as database:
            metadata=json.loads(database['metadata_json'].item())
            if not metadata.get('complete') or len(database['q'])!=200 or not database['success'].all():
                raise ValueError(f'{task}: completed 200-row success-only DB required before starting')
    project=Path(__file__).resolve().parents[1]
    root.mkdir(parents=True);(root/'results').mkdir()
    shutil.copytree(project/'src',root/'source_snapshot')
    shutil.copy2(__file__,root/'runner_snapshot.py')
    inputs=root/'inputs';(inputs/'pose_pool').mkdir(parents=True)
    hashes={}
    for task in a.tasks:
        (inputs/task).mkdir()
        for src,dest in [(a.source_root/task/'success_postures.npz',inputs/task/'success_postures.npz'),
                         (a.source_root/'pose_pool'/f'{task}.npz',inputs/'pose_pool'/f'{task}.npz')]:
            shutil.copy2(src,dest);hashes[str(src.resolve())]=hashlib.sha256(dest.read_bytes()).hexdigest()
        with np.load(inputs/task/'success_postures.npz',allow_pickle=False) as database:
            metadata=json.loads(database['metadata_json'].item())
        if metadata.get('scissors_control_version')!=a.control_version:
            raise ValueError(f'{task}: database control version does not match {a.control_version}')
    manifest=dict(tasks=a.tasks,seeds=a.seeds,workers=a.workers,poses_per_split=a.poses,points=a.points,
        started_utc=datetime.now(timezone.utc).isoformat(),source_root=str(a.source_root.resolve()),
        input_sha256=hashes,render_seed=a.seeds[0],protocol='reach_tds_v1',
        scissors_control_version=a.control_version)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    env=os.environ.copy();env.update(PYTHONPATH=str(root/'source_snapshot'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    # Include the actual workspace packages even when source is frozen elsewhere.
    packages=[str(project.parent/'tool-manip-gym'/x) for x in ('gym-scissors','gym-push','gym-spray','gym-slider')]
    env['PYTHONPATH']+=os.pathsep+os.pathsep.join(packages)
    jobs=[(t,s) for s in a.seeds for t in a.tasks]
    statuses={f'{t}/{s}':'queued' for t,s in jobs};lock=threading.Lock()
    def save():
        (root/'status.json').write_text(json.dumps(dict(utc=datetime.now(timezone.utc).isoformat(),jobs=statuses),indent=2),encoding='utf-8')
    def run(job):
        task,seed=job;folder=root/'results'/task/str(seed);folder.mkdir(parents=True)
        with lock:statuses[f'{task}/{seed}']='running';save()
        cmd=[sys.executable,'-m','improved_tds.experiments.reach_tds','--source-root',str(inputs),
             '--task',task,'--seed',str(seed),'--output',str(folder),'--poses',str(a.poses),'--points',str(a.points)]
        if seed==a.seeds[0]:cmd.append('--render')
        with (folder/'run.log').open('w',encoding='utf-8') as out,(folder/'error.log').open('w',encoding='utf-8') as err:
            subprocess.run(cmd,cwd=project,env=env,stdout=out,stderr=err,check=True)
    save();build_report(root)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures={ex.submit(run,j):j for j in jobs}
        for future in as_completed(futures):
            task,seed=futures[future]
            try:future.result();value='completed'
            except Exception as exc:value='failed: '+str(exc)
            with lock:statuses[f'{task}/{seed}']=value;save()
            build_report(root)
    build_report(root)
    if any(v!='completed' for v in statuses.values()):raise SystemExit(1)


if __name__=='__main__':main()
