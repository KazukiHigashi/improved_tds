"""Offline, evidence-linked audit HTML for the corrected 15-task direct TDS run."""
from __future__ import annotations

from collections import Counter
import html
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from sklearn.isotonic import IsotonicRegression


PROJECT=Path(__file__).resolve().parents[1]
RUN=PROJECT/'runs/paper15_unified_3m_slew_20260927'
ROOT=PROJECT/'analysis/paper15_unified_tds_corrected_20260928'
FIG=ROOT/'figures'
KEYS=('pca_1d','pls_1d','pca_2d','pls_2d')
CATEGORIES=('scissors','push','spray')
TASKS=tuple(f'{category}_lite_{i}' for category in CATEGORIES for i in range(1,6))
LABELS={'pca_1d':'PCA 1-DoF','pls_1d':'PLS 1-DoF',
        'pca_2d':'PCA 2-DoF','pls_2d':'PLS 2-DoF'}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def esc(value):
    return html.escape(str(value),quote=True)


def rel(path):
    return esc(Path(os.path.relpath(path,ROOT)).as_posix())


def link(path,label):
    return f'<a href="{rel(path)}">{esc(label)}</a>'


def table(headings,rows):
    th=''.join(f'<th>{esc(value)}</th>' for value in headings)
    tr=''.join('<tr>'+''.join(f'<td>{value}</td>' for value in row)+'</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def percent(value,digits=1):
    return '—' if value is None else f'{100*value:.{digits}f}%'


def unit(task):
    return ('mm',1000.) if task.startswith('push') else ('rad',1.)


def audit(config,target_range):
    """Classify each proposed 3+-point interval by its first failed fit gate."""
    levels=np.asarray(config['diagnostic_levels'],dtype=float)
    outcomes=[row['outcomes'] for row in config['calibration']]
    valid=np.asarray([[bool(o and o['valid']) for o in row] for row in outcomes],dtype=bool)
    measured=np.asarray([[float(o['achieved']) if o else np.nan for o in row]
                         for row in outcomes],dtype=float)
    width=float(target_range[1]-target_range[0])
    counts=Counter()
    for i in range(len(levels)-2):
        for j in range(i+2,len(levels)):
            cohort=valid[:,i:j+1].all(axis=1)
            if cohort.mean()<.75:
                counts['cohort']+=1
                continue
            yy=measured[cohort,i:j+1]
            direction=1 if np.mean(yy[:,-1]-yy[:,0])>=0 else -1
            agreement=np.mean(direction*(yy[:,-1]-yy[:,0])>.01*width)
            if agreement<.75:
                counts['direction']+=1
                continue
            mean=yy.mean(axis=0)
            if direction*(mean[-1]-mean[0])<.05*width:
                counts['gain']+=1
                continue
            monotone=mean[::direction]
            drops=np.maximum(-np.diff(monotone),0)
            if np.max(drops)>.02*width or np.sum(drops)>.05*width:
                counts['nonmonotone']+=1
                continue
            fitted=IsotonicRegression(increasing=True).fit_transform(
                np.arange(len(monotone)),monotone)
            covered=max(0.,min(target_range[1],fitted[-1])-
                        max(target_range[0],fitted[0]))/width
            if covered<.05:
                counts['range']+=1
                continue
            counts['eligible']+=1
    chosen=next(c for c in config['grasp_candidates'] if c['g']==config['grasp_score'])
    invalid=[o for row in outcomes for o in row if o is None or not o['valid']]
    inside=outside=inside_valid=outside_valid=0
    db_low,db_high=min(config['fit_rho']),max(config['fit_rho'])
    for index,rho in enumerate(levels):
        in_db=db_low<=rho<=db_high
        for row in outcomes:
            good=bool(row[index] and row[index]['valid'])
            if in_db:
                inside+=1;inside_valid+=int(good)
            else:
                outside+=1;outside_valid+=int(good)
    return dict(preload_valid=int(chosen['valid_count']),preload_attempts=len(chosen['records']),
        calibration_valid=int(valid.sum()),calibration_attempts=int(valid.size),
        calibration_invalid=len(invalid),
        invalid_final_held=sum(bool(o and o['held']) for o in invalid),
        invalid_failed=sum(bool(o and o['failed']) for o in invalid),
        invalid_truncated=sum(bool(o and o['truncated']) for o in invalid),
        invalid_numerical=sum(bool(o and o['failure_numerical']) for o in invalid),
        inside_db=[inside_valid,inside],outside_db=[outside_valid,outside],
        proposed_intervals=sum(counts.values()),interval_rejections=dict(counts),
        selected=config['selection'] is not None,
        inverse=config['inverse_target_range'] is not None,
        valid_matrix=valid)


def draw_matrix(ax,data,tasks,title,*,vmax=1.,fmt='.0f',cmap='viridis'):
    masked=np.ma.masked_invalid(np.asarray(data,dtype=float))
    colors=plt.colormaps[cmap].copy()
    colors.set_bad('#e1e7eb')
    ax.imshow(masked,aspect='auto',vmin=0,vmax=vmax,cmap=colors)
    ax.set_title(title,fontweight='bold')
    ax.set_xticks(np.arange(4),[LABELS[k] for k in KEYS],rotation=27,ha='right')
    ax.set_yticks(np.arange(len(tasks)),tasks,fontsize=8)
    for row in range(len(tasks)):
        for col in range(4):
            value=np.asarray(data)[row,col]
            label='—' if not np.isfinite(value) else format(float(value),fmt)
            ax.text(col,row,label,ha='center',va='center',fontsize=7,
                    color='white',path_effects=[pe.withStroke(linewidth=1.5,foreground='#17212b')])
    ax.grid(False)


def overview_charts(reports,audits,replays):
    fig,axes=plt.subplots(1,3,figsize=(17,8),layout='constrained')
    for ax,metric,title in zip(axes,
            ('independent_coverage','continuous_coverage','inverse_domain_fraction'),
            ('Independent held coverage [%]','Continuous held coverage [%]',
             'Common inverse domain [% of RL target]')):
        values=[[100*reports[t]['configurations'][k]['summary'][metric] for k in KEYS]
                for t in TASKS]
        draw_matrix(ax,values,TASKS,title,vmax=100.,fmt='.0f',cmap='viridis')
    fig.savefig(FIG/'overview_coverage.png',dpi=165)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,8),layout='constrained')
    data=[[[100*audits[t][k]['calibration_valid']/audits[t][k]['calibration_attempts']
            for k in KEYS] for t in TASKS],
          [[100*audits[t][k]['interval_rejections'].get('eligible',0)/
            audits[t][k]['proposed_intervals'] for k in KEYS] for t in TASKS],
          [[100*reports[t]['configurations'][k]['summary']['held_fraction'] for k in KEYS]
            for t in TASKS]]
    for ax,values,title in zip(axes,data,
            ('Calibration valid cells [%]','Candidate intervals surviving all gates [%]',
             'Independent test endpoints held 2 s [%]')):
        draw_matrix(ax,values,TASKS,title,vmax=100.,fmt='.0f',cmap='plasma')
    fig.savefig(FIG/'overview_fit_failures.png',dpi=165)
    plt.close(fig)
    mae=[]
    for t in TASKS:
        span=np.diff(reports[t]['target_range'])[0]
        mae.append([100*reports[t]['configurations'][k]['summary']['controller_comparison']
            .get('isotonic',{}).get('mae',np.nan)/span for k in KEYS])
    fig,ax=plt.subplots(figsize=(9,8),layout='constrained')
    draw_matrix(ax,mae,TASKS,'Isotonic MAE [% of full RL target span]; grey = no inverse',
        vmax=max(15.,float(np.nanmax(mae))),fmt='.1f',cmap='magma_r')
    fig.savefig(FIG/'overview_mae.png',dpi=170)
    plt.close(fig)
    gaps=[]
    for task in TASKS:
        with np.load(RUN/task/'success_postures.npz',allow_pickle=False) as z:
            gaps.append(np.sqrt(np.mean((z['q']-z['command_q'])**2,axis=1)))
    fig,axes=plt.subplots(1,2,figsize=(17,5),layout='constrained')
    axes[0].boxplot(gaps,showfliers=False,tick_labels=TASKS,patch_artist=True,
        boxprops={'facecolor':'#a3c2e6'})
    axes[0].tick_params(axis='x',rotation=70,labelsize=8)
    axes[0].set_ylabel('RMS(actuator command - measured q) [joint units]')
    axes[0].set_title('Successful DB rows are achieved states, not actuator setpoints')
    axes[0].grid(axis='y',alpha=.25)
    replay_modes=('exact_measured','exact_command','pca_1d','pls_1d','pca_2d','pls_2d')
    for j,category in enumerate(CATEGORIES):
        rows=[row for task in TASKS if task.startswith(category)
              for row in replays[task]['rows'] if 'trials' in row]
        values=[sum(row['trials'][mode]['valid'] for row in rows)/len(rows) for mode in replay_modes]
        axes[1].bar(np.arange(len(values))+.27*j,values,width=.25,label=category.title())
    axes[1].set_xticks(np.arange(len(replay_modes))+.27,
        ('measured q','command q','PCA 1D','PLS 1D','PCA 2D','PLS 2D'),rotation=30,ha='right')
    axes[1].set_ylim(0,1.05)
    axes[1].set_ylabel('2 s hold rate in 40 matched replays / category')
    axes[1].set_title('Same DB row, initial pose, target and 5 s trajectory')
    axes[1].legend()
    axes[1].grid(axis='y',alpha=.25)
    fig.savefig(FIG/'actuator_gap_and_replay.png',dpi=165)
    plt.close(fig)


def task_charts(task,report,audit):
    u,scale=unit(task)
    lo,hi=report['target_range']
    fig,axes=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    for ax,key in zip(axes.flat,KEYS):
        c=report['configurations'][key]
        ax.scatter(c['fit_rho'],scale*np.asarray(c['fit_tool']),s=7,
            c='#667085',alpha=.20,label='DB 200 success poses',rasterized=True)
        levels=np.asarray(c['diagnostic_levels'])
        for row in c['calibration']:
            for x,o in zip(levels,row['outcomes']):
                ax.scatter(x,scale*o['achieved'],s=15,
                    c='#20a67a' if o['valid'] else '#df6b50',
                    marker='o' if o['valid'] else 'x',alpha=.42,rasterized=True)
        chosen=c['selection']
        if chosen:
            rho=np.asarray(chosen['rho_start']+(np.asarray(chosen['map_u'])+1)/2*
                (chosen['rho_end']-chosen['rho_start']))
            ax.plot(rho,scale*np.asarray(chosen['map_tool']),c='#e0a000',lw=2.3,
                label='isotonic fit')
            if chosen['ramp']:
                ugrid=np.linspace(-1,1,100)
                ramp=chosen['ramp']
                rr=chosen['rho_start']+(ugrid+1)/2*(chosen['rho_end']-chosen['rho_start'])
                yy=ramp['tool_low']+(ramp['tool_high']-ramp['tool_low'])*np.clip(
                    (ugrid-ramp['start'])/(ramp['end']-ramp['start']),0,1)
                ax.plot(rr,scale*yy,c='#e07018',lw=1.6,ls='--',label='ramp fit')
        for pose in c['test']:
            for x,o in zip(levels,pose['diagnostic']['outcomes']):
                if o and o['valid']:
                    ax.scatter(x,scale*o['achieved'],c='#174ea6',marker='.',s=13,alpha=.38)
        first=c['test'][0]
        xy=[(rho,o['achieved']) for rho,o in zip(c['levels'],first['outcomes'])
            if o and o['valid']]
        if xy:
            ax.plot([v[0] for v in xy],[scale*v[1] for v in xy],
                c='#9352b7',marker='.',ms=3,lw=.9,alpha=.7,label='test continuous, pose 1')
        ax.set(title=f'{LABELS[key]}: cal valid {audit[key]["calibration_valid"]}/136; '
                     f'inverse-ready goal range {percent(c["summary"]["inverse_domain_fraction"],0)}',
               xlabel='Synergy score ρ [dimensionless]',
               ylabel=f'{"tool depth" if task.startswith("push") else "tool angle"} [{u}]')
        ax.grid(alpha=.22)
    axes[0,0].legend(fontsize=7,loc='best')
    fig.savefig(FIG/f'{task}_response.png',dpi=155)
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(15,9),layout='constrained')
    for ax,key in zip(axes.flat,KEYS):
        c=report['configurations'][key]
        matrix=audit[key]['valid_matrix']
        ax.imshow(matrix,aspect='auto',cmap=matplotlib.colors.ListedColormap(
            ['#de735c','#38aa85']),vmin=0,vmax=1)
        sel=c['selection']
        if sel:
            scores=np.asarray(c['diagnostic_levels'])
            for bound in (sel['rho_start'],sel['rho_end']):
                ax.axvline(np.interp(bound,scores,np.arange(len(scores))),c='#111827',lw=1.5)
        ax.set(title=f'{LABELS[key]}: valid={matrix.sum()}/136; '
                     f'eligible windows={audit[key]["interval_rejections"].get("eligible",0)}/120',
               xlabel='Synergy score ρ [dimensionless]',ylabel='Calibration pose [index]')
        ax.set_xticks((0,4,8,12,16),[f'{c["diagnostic_levels"][i]:+.2f}'
            for i in (0,4,8,12,16)])
        ax.set_yticks(range(8),[str(i+1) for i in range(8)])
    fig.savefig(FIG/f'{task}_calibration_validity.png',dpi=155)
    plt.close(fig)

    fig,axes=plt.subplots(2,2,figsize=(14,10),layout='constrained')
    for ax,key in zip(axes.flat,KEYS):
        c=report['configurations'][key]
        controllers=('isotonic','ramp') if c['inverse_target_range'] else ('database_linear_baseline',)
        for controller,color in zip(controllers,('#1665ab','#e67b16')):
            trials=[trial for pose in c['test'] for trial in pose['target_trials']
                    if trial['controller']==controller]
            for held,marker in ((True,'o'),(False,'x')):
                subset=[trial for trial in trials if trial['outcome']['valid']==held]
                ax.scatter([scale*t['target'] for t in subset],
                    [scale*t['outcome']['achieved'] for t in subset],
                    marker=marker,s=16,c=color,alpha=.48 if held else .75,
                    label=(controller+(' held' if held else ' not held')))
        ax.plot([scale*lo,scale*hi],[scale*lo,scale*hi],c='#333',lw=1,ls=':')
        if c['inverse_target_range']:
            a,b=c['inverse_target_range']
            ax.axvspan(scale*a,scale*b,color='#d7eee5',alpha=.45)
        comp=c['summary']['controller_comparison']
        iso=comp.get('isotonic',{})
        held_mae=iso.get('held_mae')
        inverse_range=percent(c['summary']['inverse_domain_fraction'],0)
        title=(f'held isotonic MAE={100*held_mae/(hi-lo):.2f}% of RL goal span; '
               f'inverse-ready range={inverse_range}' if held_mae is not None else
               'no held inverse trials' if c['inverse_target_range'] else
               'no qualified inverse; DB linear fallback')
        ax.set(title=f'{LABELS[key]}: {title}',
            xlabel=f'target tool value [{u}]',ylabel=f'achieved tool value [{u}]')
        ax.grid(alpha=.22)
    axes[0,0].legend(fontsize=7,loc='best')
    fig.savefig(FIG/f'{task}_inverse_error.png',dpi=155)
    plt.close(fig)


def main():
    status=load(ROOT/'status.json')
    if any(value!='completed' for value in status['tasks'].values()):
        raise RuntimeError('all corrected TDS tasks must be completed before report generation')
    FIG.mkdir(parents=True,exist_ok=True)
    reports={task:load(ROOT/task/'report.json') for task in TASKS}
    training={task:load(RUN/task/'model_training_summary.json') for task in TASKS}
    replays={task:load(ROOT/'replay_diagnostic'/f'{task}.json') for task in TASKS}
    audits={task:{key:audit(reports[task]['configurations'][key],reports[task]['target_range'])
                  for key in KEYS} for task in TASKS}
    overview_charts(reports,audits,replays)
    for task in TASKS:
        task_charts(task,reports[task],audits[task])
    summary={}
    for category in CATEGORIES:
        tasks=[t for t in TASKS if t.startswith(category)]
        pairs=[(t,key) for t in tasks for key in KEYS]
        cal_valid=sum(audits[t][k]['calibration_valid'] for t,k in pairs)
        cal_all=sum(audits[t][k]['calibration_attempts'] for t,k in pairs)
        replay_rows=[row for t in tasks for row in replays[t]['rows'] if 'trials' in row]
        summary[category]=dict(settings=len(pairs),calibration_valid=cal_valid,
            calibration_total=cal_all,calibration_failure_rate=1-cal_valid/cal_all,
            preload_valid=sum(audits[t][k]['preload_valid'] for t,k in pairs),
            preload_attempts=sum(audits[t][k]['preload_attempts'] for t,k in pairs),
            selected=sum(audits[t][k]['selected'] for t,k in pairs),
            inverse=sum(audits[t][k]['inverse'] for t,k in pairs),
            narrow_inverse=sum(reports[t]['configurations'][k]['summary']['inverse_domain_fraction']<.2
                for t,k in pairs if audits[t][k]['inverse']),
            inside_db_valid=sum(audits[t][k]['inside_db'][0] for t,k in pairs),
            inside_db_attempts=sum(audits[t][k]['inside_db'][1] for t,k in pairs),
            outside_db_valid=sum(audits[t][k]['outside_db'][0] for t,k in pairs),
            outside_db_attempts=sum(audits[t][k]['outside_db'][1] for t,k in pairs),
            cohort_rejections=sum(audits[t][k]['interval_rejections'].get('cohort',0)
                for t,k in pairs),
            proposed_intervals=sum(audits[t][k]['proposed_intervals'] for t,k in pairs),
            replay_rows=len(replay_rows),
            exact_measured_valid=sum(row['trials']['exact_measured']['valid'] for row in replay_rows),
            exact_command_valid=sum(row['trials']['exact_command']['valid'] for row in replay_rows),
            invalid_numerical=sum(audits[t][k]['invalid_numerical'] for t,k in pairs),
            invalid_truncated=sum(audits[t][k]['invalid_truncated'] for t,k in pairs))
    iso_better=ramp_better=0
    for task in TASKS:
        for key in KEYS:
            comp=reports[task]['configurations'][key]['summary']['controller_comparison']
            if 'isotonic' in comp and 'ramp' in comp:
                iso_better+=comp['isotonic']['mae']<comp['ramp']['mae']
                ramp_better+=comp['ramp']['mae']<comp['isotonic']['mae']
    audit_json=dict(protocol='paper15_tds_diagnostic_report_v1',summary=summary,
        inverse_controller_comparison=dict(isotonic_better=iso_better,ramp_better=ramp_better),
        tasks={task:{key:{k:v for k,v in audits[task][key].items() if k!='valid_matrix'}
                     for key in KEYS} for task in TASKS})
    (ROOT/'diagnostic_metrics.json').write_text(json.dumps(audit_json,indent=2),encoding='utf-8')

    category_rows=[]
    for category in CATEGORIES:
        s=summary[category]
        category_rows.append([esc(category.title()),
            f'{s["preload_valid"]}/{s["preload_attempts"]}',
            f'{s["calibration_valid"]}/{s["calibration_total"]}',
            percent(s['calibration_failure_rate']),f'{s["selected"]}/{s["settings"]}',
            f'{s["inverse"]}/{s["settings"]} (うち狭域 {s["narrow_inverse"]})',
            f'{s["exact_measured_valid"]}/{s["replay_rows"]}',
            f'{s["exact_command_valid"]}/{s["replay_rows"]}'])
    range_rows=[]
    for category in CATEGORIES:
        s=summary[category]
        range_rows.append([esc(category.title()),
            f'{s["inside_db_valid"]}/{s["inside_db_attempts"]} '
            f'({percent(s["inside_db_valid"]/s["inside_db_attempts"])})',
            f'{s["outside_db_valid"]}/{s["outside_db_attempts"]} '
            f'({percent(s["outside_db_valid"]/s["outside_db_attempts"])})',
            f'{s["cohort_rejections"]}/{s["proposed_intervals"]} '
            f'({percent(s["cohort_rejections"]/s["proposed_intervals"])})'])
    learning_rows=[]
    for task in TASKS:
        best=training[task]['validation']['best']
        database=reports[task]['database_metadata']
        achieved=database['achieved_range']
        target=database['target_range']
        u,scale=unit(task)
        learning_rows.append([esc(task),f'{best["timesteps"]:,}',percent(best['success_rate']),
            f'{training[task]["total_timesteps"]:,}',
            f'{database["episodes"]:,}',percent(database['rows']/database['episodes']),
            f'{scale*achieved[0]:.3f}–{scale*achieved[1]:.3f} {u}',
            f'{scale*target[0]:.3f}–{scale*target[1]:.3f} {u}',
            link(RUN/task/'success_postures.npz','200 success poses')])
    analysis_rows=[]
    for task in TASKS:
        report=reports[task]
        span=np.diff(report['target_range'])[0]
        for key in KEYS:
            config=report['configurations'][key]
            a=audits[task][key]
            s=config['summary']
            comp=s['controller_comparison']
            iso=comp.get('isotonic',{})
            ramp=comp.get('ramp',{})
            fallback=comp.get('database_linear_baseline',{})
            mae=(f'{100*iso["mae"]/span:.2f}% full span'
                 if iso.get('mae') is not None else '—')
            held_mae=(f'{100*iso["held_mae"]/span:.2f}% / {iso["held"]}/{iso["trials"]} held'
                      if iso.get('held_mae') is not None else '—')
            analysis_rows.append([esc(task),esc(LABELS[key]),str(a['preload_valid'])+'/8',
                f'{a["calibration_valid"]}/136',
                str(a['interval_rejections'].get('eligible',0))+'/120',
                'yes' if a['selected'] else '<span class="fail">NO</span>',
                percent(s['independent_coverage']),percent(s['continuous_coverage']),
                percent(s['held_fraction']),percent(s['inverse_domain_fraction']),
                mae,held_mae,
                f'{100*ramp["mae"]/span:.2f}%' if ramp.get('mae') is not None else '—',
                f'{100*fallback["mae"]/span:.2f}%' if fallback.get('mae') is not None else '—'])
    sections=[]
    for task in TASKS:
        report=reports[task]
        replay=replays[task]
        u,scale=unit(task)
        rows=[]
        for key in KEYS:
            a=audits[task][key]
            c=report['configurations'][key]
            counts=a['interval_rejections']
            comp=c['summary']['controller_comparison']
            iso=comp.get('isotonic',{})
            ramp=comp.get('ramp',{})
            reasons=', '.join(f'{label} {counts.get(code,0)}' for code,label in (
                ('cohort','6/8姿勢を同一区間で保持できず'),('direction','方向不一致'),
                ('gain','応答幅不足'),('nonmonotone','非単調'),('range','Coverage<5%')))
            rows.append([esc(LABELS[key]),f'{a["calibration_valid"]}/136 '
                f'({percent(1-a["calibration_valid"]/136)}失敗)',
                f'{counts.get("eligible",0)}/120',reasons,
                '—' if c['inverse_target_range'] is None else
                f'{scale*c["inverse_target_range"][0]:.3f}–{scale*c["inverse_target_range"][1]:.3f} {u}',
                '—' if iso.get('mae') is None else
                f'{scale*iso["mae"]:.4f} {u} (held {iso["held"]}/{iso["trials"]})',
                '—' if ramp.get('mae') is None else f'{scale*ramp["mae"]:.4f} {u}'])
        matched=[row for row in replay['rows'] if 'trials' in row]
        mode_rows=[]
        for mode in ('exact_measured','exact_command')+KEYS:
            trials=[row['trials'][mode] for row in matched]
            mode_rows.append([esc(mode),f'{sum(x["valid"] for x in trials)}/{len(trials)}',
                f'{scale*np.mean([x["tool_error_from_database"] for x in trials]):.4f} {u}',
                f'{np.mean([x["reconstructed_q_rms"] for x in trials]):.4f}'])
        figures=''.join(f'<figure><img loading="lazy" src="figures/{task}_{suffix}.png">'
            f'<figcaption>{esc(title)}</figcaption></figure>' for suffix,title in (
            ('calibration_validity','各Score×各Calibration姿勢の2秒保持成否。黒線は選択されたフィット区間。'),
            ('response','DB・Calibration成功/失敗点・等張/ランプフィット・Test独立/連続スイープの角度応答。'),
            ('inverse_error','未知目標の指令角度と実現角度。緑背景はCalibrationで逆写像が構成された範囲。')))
        sections.append(f'<details id="{task}"><summary>{esc(task)} — '
            f'cal valid {sum(audits[task][k]["calibration_valid"] for k in KEYS)}/544; '
            f'inverse {sum(audits[task][k]["inverse"] for k in KEYS)}/4</summary>'
            f'<p>{link(ROOT/task/"report.json","TDS raw JSON")} / '
            f'{link(ROOT/"replay_diagnostic"/(task+".json"),"matched replay JSON")}</p>'
            f'<h3>フィッティング点の選別と逆写像</h3>'+
            table(('方式','有効Calibration点','採択候補区間','候補棄却の内訳',
                '共通逆写像区間','等張 MAE','ランプ MAE'),rows)+
            '<h3>同じ初期姿勢・目標での直接リプレイ診断</h3>'+
            '<p class="muted">各Test初期姿勢からDBの目標中央値に近い成功行を1件選択。'
            '元ポリシー全軌道は再生せず、各目標姿勢へ5秒移動＋2秒保持。'
            '角度差はDB記録角度との差であり、把持成否とは別指標。</p>'+
            table(('制御目標','終端把持成功','DB角度との差・平均','記録されたqとの差RMS'),mode_rows)+
            figures+'</details>')
    paragraphs=(
        '<p class="warn">旧TDS結果の200Step打ち切りは解消済み。本報告は修正版15タスクのJSONのみを入力とする。'
        '成功姿勢DBは各200件・10角度ビン各20件。今回の診断は学習を変更していない。</p>'
        '<p><strong>主要因：</strong>Push/SprayではDBの実現関節姿勢 q と、接触下でその姿勢を維持したアクチュエータ指令 '
        'command_q に大きな差がある。同じ初期姿勢・目標でDB行を直接リプレイすると、測定qを指令するよりcommand_qを指令した方が'
        '把持保持率が大幅に高い。ただしcommand_qもツール角度誤差を保証せず、これは「コマンド対状態」の不一致を示す診断であり、'
        '元のRL軌道の完全再現ではない。</p>'
        '<p>この同条件リプレイではPushの保持が measured q 17/40 → command_q 40/40、'
        'Sprayは3/40 → 36/40。元の成功姿勢を単一目標への直線FFで再生しても失敗するため、'
        'PCA/PLSが唯一の失敗要因ではない。逆にcommand_qの高い把持率は角度追従の成功率を意味せず、'
        '例えばPushでのDB記録深さとの差の平均は measured q 指令で約0.66 mm、command_q 指令で約1.90 mm。</p>'
        '<p>Calibration失敗は「各Score×各初期姿勢の5秒移動後、終端2秒の全フレームで把持を保持できなかった割合」。'
        '全17 Score×8姿勢=136点/設定。PCA/PLSフィット自体は成功DB 200点から常に計算できる。'
        'ここでの「フィット点構築失敗」は、角度応答を較正するための物理的に有効なスイープ点が作れないことを指す。</p>'
        '<p>フィット区間は17点から長さ3以上の120候補を調べ、同一の6/8以上の姿勢が区間全点で保持し、'
        '応答方向の一致・5%以上の角度幅・単調性を満たす場合のみ採択。逆写像可能範囲は等張回帰とランプ関数に共通の較正角度区間。'
        '範囲外はMAEに含めず、必ずCoverageと併記する。</p>'
        '<p>独立Coverageは各Scoreをリセットから試し、2秒保持に成功した角度幅/元のRL目標幅のTest8姿勢平均。'
        '連続Coverageは往復Scoreスイープを1試行内で実施した同じ指標。姿勢間の経路依存により両者は大きく乖離し得る。'
        'MAEは逆写像が構成された場合、共通区間内9未知目標×8Test姿勢の予測誤差。'
        '「観測全件MAE」は保持失敗例を含み、「held MAE」は終端保持成功例のみ。'
        '逆写像が無い行のDB線形baseline MAEは異なる評価目標であり、等張/ランプMAEと横比較しない。</p>'
        '<p>具体例としてScissors 2は独立Coverage 80–84%に対して連続Coverageが2–5%、'
        'Push 3 のPCA 2DoFは独立100%に対して連続0%。一度到達した成功姿勢を順にたどる制御では、'
        '角度応答の可逆性・把持維持は別途成立させる必要がある。これは経路依存を示すが、機構的原因を断定するものではない。</p>'
        '<p>逆写像が構成された設定でも、RL目標幅の20%未満しか逆写像できない例がPushで3/9、Sprayで8/13。'
        'MAEが小さく見えても狭い目標域だけを評価している場合があり、範囲外をゼロ誤差として扱わない。</p>'
        '<p><strong>解釈上の制限：</strong>Calibration/Testの初期姿勢は分離されるが、PCA/PLSのDBは同じ32姿勢プールから収集された。'
        'Testは真の未知初期姿勢への外挿ではない。2DoFは較正で選んだ一定の把持軸スコアを付加する方式で、'
        '各目標前に独立した把持フェーズを必ず実行する方式ではない。Scissorsでは選ばれた把持スコアが全て0で、1/2DoFは実質同じ。</p>'
        '<p>失敗は1.5倍に拡張したScore域だけの問題でもない。下表の「DB内域」は成功DBで観測した主成分得点の'
        '最小～最大内のCalibration点であり、それでもPushは約48%、Sprayは約56%が終端保持に失敗した。'
        '角度応答が不十分なだけでなく、複数Scoreにわたり同じ初期姿勢の把持を維持できる区間が作れないことが'
        'フィット棄却の主因である。</p>'
    )
    style='''
    :root{--ink:#172935;--blue:#125a91;--line:#d7e0e7;--paper:#fff;--bg:#eef3f6}
    body{font:15px/1.62 system-ui,-apple-system,"Yu Gothic",sans-serif;color:var(--ink);background:var(--bg);margin:0}
    main{max-width:1500px;margin:auto;padding:22px 26px 80px}h1{font-size:2rem;margin-bottom:.2rem}
    h2{border-bottom:2px solid #b7cbd9;padding-bottom:.3em}h3{margin-top:1.5em}
    .lead{font-size:1.1rem}.muted,figcaption{color:#526879;font-size:.91rem}
    .warn{background:#fff2e1;border-left:5px solid #d4842d;padding:12px}
    section,details{background:var(--paper);border-radius:12px;padding:18px 22px;margin:18px 0;box-shadow:0 2px 12px #102a4210}
    details>summary{font-size:1.1rem;font-weight:700;cursor:pointer;padding:4px 0}
    .scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;min-width:760px}
    th,td{padding:8px 9px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
    th{background:#edf4f8;position:sticky;top:0}tbody tr:hover{background:#f4f9fc}
    img{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}
    figure{margin:22px 0}figcaption{margin:5px 4px}a{color:var(--blue)}.fail{color:#b63d2b;font-weight:700}
    nav a{margin-right:1.1rem;display:inline-block}.key{background:#e6f2ee;padding:2px 5px;border-radius:4px}
    '''
    body=(f'<header><h1>15タスク共通条件 TDS 再評価 — 診断報告</h1>'
        '<p class="lead">PCA / PLS × 1 / 2 DoF、Coverage・MAE・Calibration点失敗率・把持保持の原因分析</p>'
        '<nav><a href="#overview">全体</a><a href="#fit">失敗分解</a>'
        '<a href="#accuracy">Coverage / MAE</a><a href="#tasks">タスク別</a></nav></header>'
        '<section id="overview"><h2>結論と評価条件</h2>'+paragraphs+
        f'<p>{link(ROOT/"manifest.json","評価条件 manifest")} / '
        f'{link(ROOT/"status.json","完了状況")} / '
        f'{link(ROOT/"diagnostic_metrics.json","集計値 JSON")}</p></section>'
        '<section id="fit"><h2>フィッティング用サンプル構築の失敗</h2>'
        '<p>分母は異なる：Calibration点は20設定×136=2,720点/カテゴリ、採択/逆写像は20設定/カテゴリ、'
        '直接リプレイは40件/カテゴリ。リプレイ行はTest姿勢につきDBから1行であり、DB全200件の成功率ではない。</p>'+
        table(('カテゴリ','Score 0 のpreload保持','有効Calibration点','無効率',
            '区間選択可能','逆写像可能',
            'DB実現qを指令','DB command_qを指令'),category_rows)+
        '<p>狭域＝逆写像が成立しても、その共通角度区間がRL全目標幅の20%未満の設定。'
        '小さなMAEでも全域制御を意味しない。</p>'+
        table(('カテゴリ','DB観測Score範囲内：有効点','1.5倍拡張域：有効点',
               '共通保持姿勢不足による候補区間棄却'),range_rows)+
        '<figure><img src="figures/overview_fit_failures.png"><figcaption>'
        'Calibration点成立率・候補区間採択率・Test独立終端保持率。灰色の未評価値はない。</figcaption></figure>'
        '<figure><img src="figures/actuator_gap_and_replay.png"><figcaption>'
        '成功DBでの実現qと指令qの差、および同条件直接リプレイの把持保持成否。'
        'command_qへの置換は今回のPCA/PLS本評価には適用していない。</figcaption></figure></section>'
        '<section id="accuracy"><h2>Coverage・逆写像可能範囲・MAE</h2>'
        '<figure><img src="figures/overview_coverage.png"><figcaption>'
        '左：独立到達Coverage、中：連続往復Coverage、右：等張/ランプ共通の逆写像可能範囲。'
        '全てRL目標角度幅を分母とする。Scissors2やPush3では独立/連続の乖離に注意。</figcaption></figure>'
        '<figure><img src="figures/overview_mae.png"><figcaption>'
        '逆写像が成立した設定のみの等張回帰MAEをRL全目標幅で正規化。'
        '灰色はフィット不可であり誤差0ではない。異なる逆写像範囲同士の単純比較は禁止。</figcaption></figure>'
        f'<p>共通区間内で比較できる42設定のうち、等張回帰MAEが低い設定 {iso_better}、'
        f'ランプ関数が低い設定 {ramp_better}。ただし差の統計的有意性は未評価。</p>'
        '<h3>全60設定の数値一覧</h3>'+table(('タスク','方式','preload成立','Calibration成立',
            '採択候補/120','Fit選択','独立Coverage','連続Coverage','独立保持率',
            '逆写像範囲','等張MAE 全観測','等張MAE 保持成功のみ','ランプMAE','DB線形baseline MAE'),
            analysis_rows)+'</section>'
        '<section><h2>RL学習と成功姿勢DB</h2>'
        '<p>Best PolicyのValidationは32初期姿勢×11目標=352試行。全タスクでDB200行、'
        '目標角度10ビンに各20行。DB採択率は収集全試行数に対する保存200行の割合。'
        '達成角度は目標端点そのものではなく許容誤差内となり得る。RL成功率とFFシナジーの成立は別。</p>'+
        table(('タスク','Best Step','Best Validation成功率','実学習Step','DB収集試行',
            'DB採択率','DB達成範囲','RL目標範囲','DB'),learning_rows)+'</section>'
        '<section id="tasks"><h2>タスク別のフィット失敗・角度応答・未知目標誤差</h2>'
        '<p>Cal点図：緑=2秒保持成功、赤=失敗、黒線=採択区間。応答図：灰=DB、緑/赤=Cal、'
        '青=Test独立成功、紫=Test姿勢1の連続成功。目標図：丸=保持成功、×=不成立。</p>'+
        ''.join(sections)+'</section>')
    page=('<!doctype html><html lang="ja"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>15-task TDS corrected experiment diagnostic report</title>'
        f'<style>{style}</style></head><body><main>{body}</main></body></html>')
    (ROOT/'index.html').write_text(page,encoding='utf-8')
    print(ROOT/'index.html')


if __name__=='__main__':
    main()
