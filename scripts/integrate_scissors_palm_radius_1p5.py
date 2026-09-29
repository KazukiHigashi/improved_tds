"""Adopt the verified 0.075 m Scissors TDS rerun in the existing paper bundle.

The old 0.05 m reports, figures and trajectory gallery are archived in place
before replacement. Push and Spray selected results are not changed.
"""
import json
from pathlib import Path
import shutil

import numpy as np

import build_paper15_tds_diagnostic_html as charts
from build_paper15_paper_bundle import TASKS, copy_required, digest, refresh_report


SCISSORS=[f'scissors_lite_{i}' for i in range(1,6)]
FIGURES=('response','calibration_validity','inverse_error')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def validate(bundle, run):
    manifest=read(bundle/'manifest.json')
    campaign=read(run/'manifest.json')
    status=read(run/'status.json')
    gallery_status=read(run/'trajectory_frames'/'status.json')
    if manifest.get('protocol')!='paper15_integrated_selected_results_v1' or set(manifest['tasks'])!=set(TASKS):
        raise ValueError('not the expected 15-task paper bundle')
    if campaign.get('scissors_evaluation_palm_radius_multiplier')!=1.5:
        raise ValueError('rerun multiplier is not 1.5')
    if set(campaign['tasks'])!=set(SCISSORS) or any(status['tasks'].get(t)!='completed' for t in SCISSORS):
        raise ValueError('Scissors 1–5 rerun is incomplete')
    if any(not gallery_status['tasks'].get(t,'').startswith('completed') for t in SCISSORS):
        raise ValueError('Scissors 0.075 m gallery is incomplete')
    for task in SCISSORS:
        old_path=bundle/task/'report.json'
        new_path=run/task/'report.json'
        old,new=read(old_path),read(new_path)
        meta=manifest['tasks'][task]
        if digest(old_path)!=meta['selected_report_sha256']:
            raise ValueError(f'paper bundle baseline changed: {task}')
        if not (np.isclose(old['palm_sphere_radius'],.05)
                and np.isclose(new['source_policy_palm_sphere_radius'],.05)
                and np.isclose(new['palm_sphere_radius'],.075)):
            raise ValueError(f'unexpected sphere radii: {task}')
        if (new['protocol']!='reach_tds_direct_v3_scaled_palm_evaluation'
                or new['evaluation_grasp_criterion']!='hinge_center_inside_scaled_palm_attached_sphere'):
            raise ValueError(f'wrong evaluation protocol: {task}')
        if (new['database_sha256']!=old['database_sha256']
                or new['database_sha256']!=meta['database_sha256']
                or new['calibration_poses']!=old['calibration_poses']
                or new['test_poses']!=old['test_poses']
                or new['target_range']!=old['target_range']
                or new['fit_rows']!=old['fit_rows']!=200
                or new['trajectory_seconds']!=old['trajectory_seconds']!=5.
                or new['operation_hold_seconds']!=old['operation_hold_seconds']!=2.):
            raise ValueError(f'non-radius experimental condition changed: {task}')
        for method in ('pca','pls'):
            for dof in (1,2):
                key=f'{method}_{dof}d'
                a,b=old['configurations'][key],new['configurations'][key]
                if (a['fit_rho']!=b['fit_rho'] or a['fit_tool']!=b['fit_tool']
                        or a['diagnostic_levels']!=b['diagnostic_levels']
                        or a['grasp_score']!=b['grasp_score'] or b['grasp_score']!=0
                        or len(b['calibration'])!=8
                        or any(len(row['outcomes'])!=17 for row in b['calibration'])
                        or len(b['test'])!=8):
                    raise ValueError(f'fit/sweep conditions changed: {task}/{key}')
        gallery=read(run/'trajectory_frames'/task/'manifest.json')
        if (not np.isclose(gallery['scissors_evaluation_palm_radius'],.075)
                or any(len(gallery['methods'][m])!=17 for m in ('pca','pls'))):
            raise ValueError(f'incomplete scaled gallery: {task}')
        if (run/task/'run.err.log').stat().st_size:
            raise ValueError(f'nonempty Scissors error log: {task}')
    return manifest


def repair_baseline_html(bundle):
    """Keep the archived 0.05 m HTML linked to its archived data and figures."""
    path=bundle/'index_palm005.html'
    page=path.read_text(encoding='utf-8')
    for task in SCISSORS:
        page=page.replace(f'{task}/report.json',f'{task}/baseline_palm005_report.json')
        page=page.replace(f'trajectory_frames/{task}/',
                          f'trajectory_frames_palm005/{task}/')
        for suffix in FIGURES:
            name=f'{task}_{suffix}.png'
            page=page.replace(f'figures/{name}',f'figures/palm005_baseline/{name}')
    for name in ('summary_inverse_domain.png','summary_normalized_mae.png'):
        page=page.replace(f'figures/{name}',f'figures/palm005_baseline/{name}')
    page=page.replace('href="manifest.json"','href="manifest_palm005.json"')
    path.write_text(page,encoding='utf-8')
    archived_manifest_path=bundle/'manifest_palm005.json'
    archived_manifest=read(archived_manifest_path)
    for task in SCISSORS:
        archived_manifest['tasks'][task]['selected_report']=f'{task}/baseline_palm005_report.json'
    archived_manifest['integrated_html_sha256']=digest(path)
    archived_manifest['archive_note']='0.05 m Scissors report and chart paths remapped after 0.075 m adoption'
    archived_manifest_path.write_text(json.dumps(archived_manifest,indent=2,ensure_ascii=False),encoding='utf-8')


def integrate(bundle, run):
    if not bundle.is_dir() or not run.is_dir():
        raise FileNotFoundError('paper bundle and completed rerun must both exist')
    if (bundle/'scissors_lite_1'/'baseline_palm005_report.json').exists():
        raise FileExistsError('0.05 m baseline already archived; refusing second integration')
    manifest=validate(bundle,run)
    old_source=manifest['source_directories']['scissors_tds']
    copy_required(bundle/'index.html',bundle/'index_palm005.html')
    copy_required(bundle/'manifest.json',bundle/'manifest_palm005.json')
    for name in ('summary_inverse_domain.png','summary_normalized_mae.png'):
        copy_required(bundle/'figures'/name,bundle/'figures'/'palm005_baseline'/name)
    for task in SCISSORS:
        copy_required(bundle/task/'report.json',bundle/task/'baseline_palm005_report.json')
        for suffix in FIGURES:
            name=f'{task}_{suffix}.png'
            copy_required(bundle/'figures'/name,bundle/'figures'/'palm005_baseline'/name)
        shutil.copytree(bundle/'trajectory_frames'/task,
            bundle/'trajectory_frames_palm005'/task)
    shutil.copytree(run/'evaluation_source_snapshot',bundle/'scissors_scaled_evaluation_source_snapshot')
    copy_required(run/'manifest.json',bundle/'source_manifests'/'scissors_palm1p5_manifest.json')
    copy_required(run/'status.json',bundle/'source_manifests'/'scissors_palm1p5_status.json')
    copy_required(run/'trajectory_frames'/'status.json',
        bundle/'source_manifests'/'scissors_palm1p5_gallery_status.json')
    copy_required(run/'launcher_snapshot.py',
        bundle/'source_manifests'/'scissors_palm1p5_launcher_snapshot.py')
    for task in SCISSORS:
        old_hash=digest(bundle/task/'baseline_palm005_report.json')
        new_report=copy_required(run/task/'report.json',bundle/task/'report.json')
        report=read(bundle/task/'report.json')
        chart_audit={key:charts.audit(report['configurations'][key],report['target_range'])
                     for key in ('pca_1d','pls_1d','pca_2d','pls_2d')}
        charts.FIG=bundle/'figures'
        charts.task_charts(task,report,chart_audit)
        for suffix in FIGURES:
            if not (bundle/'figures'/f'{task}_{suffix}.png').is_file():
                raise FileNotFoundError(f'{task}_{suffix}.png')
        source_gallery=run/'trajectory_frames'/task
        target_gallery=bundle/'trajectory_frames'/task
        for source in source_gallery.iterdir():
            if source.is_file() and (source.suffix=='.jpg' or source.name=='manifest.json'):
                copy_required(source,target_gallery/source.name)
        meta=manifest['tasks'][task]
        meta['baseline_palm005_report']=f'{task}/baseline_palm005_report.json'
        meta['baseline_palm005_report_sha256']=old_hash
        meta['selected_report_sha256']=new_report['sha256']
        meta['scissors_tds_evaluation_palm_radius_m']=.075
        meta['scissors_rl_database_palm_radius_m']=.05
    manifest['source_directories']['scissors_tds_baseline']=old_source
    manifest['source_directories']['scissors_tds']=str(run)
    manifest['selected_tds']['scissors']='Palm-attached sphere 0.075 m for TDS Calibration/Test; RL/DB 0.05 m'
    manifest['included'].extend(['old 0.05 m Scissors reports, charts and trajectory gallery',
        'new 0.075 m Scissors trajectory gallery and evaluation source snapshot'])
    (bundle/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    refresh_report(type('Args',(),dict(output=bundle))())
    repair_baseline_html(bundle)


if __name__=='__main__':
    project=Path(__file__).resolve().parents[1]
    integrate(project/'analysis'/'paper15_paper_bundle_20260928',
              project/'analysis'/'scissors_tds_palm_radius_1p5_20260928')
