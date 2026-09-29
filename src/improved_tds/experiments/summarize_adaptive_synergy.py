"""Validate adaptive FF artifacts and build a standalone review HTML report."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from improved_tds.experiments.evaluate_adaptive_synergy import ARMS


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--visual-seed', type=int, default=20260911)
    parser.add_argument('--expected-reports', type=int)
    args = parser.parse_args(argv)
    paths = sorted(args.report_dir.glob('*_seed*.json'))
    if args.expected_reports is not None and len(paths) != args.expected_reports:
        raise ValueError(f'Expected {args.expected_reports} reports, got {len(paths)}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / 'plots'
    plot_dir.mkdir(exist_ok=True)
    groups, details, paired = defaultdict(list), [], defaultdict(dict)
    production = args.expected_reports == 75
    for path in paths:
        report = json.loads(path.read_text(encoding='utf-8'))
        assert not set(report['test_poses']) & set(report['calibration_poses'])
        if production:
            assert not report['pilot']
            assert set(report['configurations']) == {'pca', 'pls'}
        task = path.stem.split('_seed')[0]
        width = report['target_range'][1] - report['target_range'][0]
        for method, configuration in report['configurations'].items():
            assert set(configuration['arms']) == set(ARMS)
            for arm, result in configuration['arms'].items():
                poses = result['poses']
                if result['calibrated']:
                    assert len(poses) == len(report['test_poses'])
                    assert all(len(p['outcomes']) == 13 for p in poses)
                    assert all(len(p['target_trials']) == (0 if result.get('inverse_map_failure') else 9) for p in poses)
                outcomes = [o for p in poses for o in p['outcomes']]
                trials = [o for p in poses for o in p['target_trials']]
                covered = [o for o in trials if o['calibration_reachable']]
                pose_correlations = []
                for pose in poses:
                    values = [o['achieved'] for o in pose['outcomes']]
                    if np.std(values) > 1e-10:
                        pose_correlations.append(float(np.corrcoef(np.linspace(-1., 1., len(values)), values)[0, 1]))
                metric = dict(task=task, split=report['split_seed'], calibrated=result['calibrated'],
                    held_coverage=float(np.mean([p['held_coverage'] for p in poses])) if poses else 0.,
                    held_rate=float(np.mean([o['held'] and not o['failed'] for o in outcomes])) if outcomes else 0.,
                    normalised_mae=float(np.mean([o['abs_error'] for o in trials])/width) if trials else None,
                    reachable_normalised_mae=float(np.mean([o['abs_error'] for o in covered])/width) if covered else None,
                    correlation=result.get('test_mean_correlation'), inverse_map_failure=result.get('inverse_map_failure', False),
                    expanded=bool(result['calibrated'] and (abs(result['selection']['rho_start']) > 1.000001 or abs(result['selection']['rho_end']) > 1.000001)),
                    asymmetric=bool(result['calibrated'] and abs(result['selection']['rho_start'] + result['selection']['rho_end']) > 1e-8),
                    excludes_grasp=bool(result['calibrated'] and not min(result['selection']['rho_start'], result['selection']['rho_end']) <= 0 <= max(result['selection']['rho_start'], result['selection']['rho_end'])),
                    reversed=bool(result['calibrated'] and result['selection']['direction'] < 0),
                    changed_candidate=bool(result['calibrated'] and result['candidate'] != configuration['arms']['legacy']['candidate']))
                metric.update(pose_correlation_count=len(pose_correlations),
                              negative_pose_correlations=sum(value < 0 for value in pose_correlations))
                groups[(task.split('_')[0], method, arm)].append(metric)
                paired[(task.split('_')[0], method, arm)][path.stem] = metric['held_coverage']
                if report['split_seed'] != args.visual_seed:
                    continue
                title = f'{task} / {method.upper()} / {arm}'
                details.append(f'<h3>{html.escape(title)}</h3>')
                if not result['calibrated']:
                    details.append('<p>校正不成立：事前に定めた把持維持・方向一致・変位幅・Coverage条件を満たす区間なし。Test未実施。</p>')
                    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
                    records = configuration['calibration_records']
                    if arm == 'direction':
                        records = [dict(candidate=result['candidate'], **configuration['fixed_calibration'])]
                    elif arm == 'range':
                        records = [r for r in records if r['candidate'] == result['candidate']]
                    for raw in records:
                        ry = np.mean([[o['achieved'] for o in row] for row in raw['outcomes']], axis=0)
                        ax.plot(raw['levels'], ry, '.-', label=f'preload,grasp={raw["candidate"]}')
                    ax.axhspan(*report['target_range'], alpha=.08, color='green', label='RL target range')
                    ax.set(title=f'{title}: rejected calibration candidates (no Test selection)',
                           xlabel='Physical score / nominal amplitude (rho)', ylabel='Raw tool response (all Calibration poses)')
                    ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout()
                    name = f'{task}__{method}__{arm}__rejected.png'
                    fig.savefig(plot_dir/name); plt.close(fig)
                    details.append(f'<img src="plots/{name}" alt="Rejected calibration responses">')
                    continue
                choice = result['selection']
                a, b = choice['rho_start'], choice['rho_end']
                fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
                fit = result['fit_samples']
                ax.scatter(fit['score'], fit['tool'], marker='x', alpha=.3, label=f'Fit samples (n={len(fit["score"])})')
                # Show unmodified calibration responses, including the unused range.
                if arm in ('legacy', 'direction'):
                    raw = configuration['fixed_calibration']
                else:
                    raw = next(c for c in configuration['calibration_records'] if c['candidate'] == result['candidate'])
                rx = np.asarray(raw['levels'])
                ry = np.mean([[o['achieved'] for o in row] for row in raw['outcomes']], axis=0)
                ax.plot(-1 + 2*(rx-a)/(b-a), ry, ':', color='gray', label='Raw calibration mean (all poses)')
                if arm != 'legacy':
                    ax.plot(-1 + 2*(np.array(choice['raw_rho'])-a)/(b-a), choice['raw_tool'], '--', color='green', label='Raw calibration (fixed held cohort)')
                ax.plot(choice['map_u'], choice['map_tool'], '--', color='orange', label='Inverse-control map')
                y = np.array([[o['achieved'] for o in p['outcomes']] for p in poses])
                u = np.linspace(-1, 1, 13)
                ax.plot(u, y.mean(axis=0), 'o-', label='Test mean (all poses)')
                ax.fill_between(u, y.mean(axis=0)-y.std(axis=0), y.mean(axis=0)+y.std(axis=0), alpha=.12)
                ax.axvspan(-1, 1, alpha=.06, color='blue', label='Selected input interval')
                ax.axhspan(*report['target_range'], alpha=.08, color='green', label='RL target range')
                ax.axvline(result['grasp_score'], color='purple', linewidth=.8, label='Grasp reference')
                ax.set(title=f'{title}\nrho [{a:.3f}, {b:.3f}]; nominal amplitude {configuration["nominal_amplitude"]:.3f}',
                       xlabel='Normalised control input (fit samples: unclipped linear projection)', ylabel='Tool angle / displacement')
                ax.legend(fontsize=7); ax.grid(alpha=.2); fig.tight_layout()
                name = f'{task}__{method}__{arm}.png'
                fig.savefig(plot_dir / name); plt.close(fig)
                details.append(f'<p>候補 [preload, grasp] = {result["candidate"]}; 物理スコア区間 [{a:.3f}, {b:.3f}]; 把持姿勢の入力座標 = {result["grasp_score"]:.3f}</p>')
                details.append(f'<img src="plots/{name}" alt="{html.escape(title)} response">')
                for image in result['images']:
                    assert Path(image).is_file(), image
                    rel = os.path.relpath(image, args.output_dir).replace('\\', '/')
                    details.append(f'<a href="{html.escape(rel)}"><img class="sheet" src="{html.escape(rel)}" alt="{html.escape(title)} 13-score posture sheet"></a>')
    rows, aggregate = [], {}
    def fmt(value):
        return '—' if value is None else f'{100*value:.1f}%'
    for key, entries in sorted(groups.items()):
        family, method, arm = key
        def mean(name):
            values = [e[name] for e in entries if e[name] is not None]
            return float(np.mean(values)) if values else None
        values = {name:mean(name) for name in ('held_coverage', 'held_rate', 'normalised_mae', 'reachable_normalised_mae')}
        values.update(n=len(entries), calibrated=sum(e['calibrated'] for e in entries),
                      error_available=sum(e['normalised_mae'] is not None for e in entries),
                      negative_correlations=sum(e['correlation'] is not None and e['correlation'] < 0 for e in entries),
                      inverse_failures=sum(e['inverse_map_failure'] for e in entries),
                      expanded=sum(e['expanded'] for e in entries), asymmetric=sum(e['asymmetric'] for e in entries),
                      excludes_grasp=sum(e['excludes_grasp'] for e in entries), reversed=sum(e['reversed'] for e in entries),
                      changed_candidate=sum(e['changed_candidate'] for e in entries),
                      pose_correlation_count=sum(e['pose_correlation_count'] for e in entries),
                      negative_pose_correlations=sum(e['negative_pose_correlations'] for e in entries), per_split=entries)
        aggregate['/'.join(key)] = values
        rows.append(f'<tr><td>{family}</td><td>{method}</td><td>{arm}</td><td>{values["calibrated"]}/{len(entries)}</td>'
                    f'<td>{fmt(values["held_coverage"])}</td><td>{fmt(values["held_rate"])}</td><td>{fmt(values["normalised_mae"])}</td>'
                    f'<td>{values["error_available"]}</td><td>{values["negative_correlations"]}</td>'
                    f'<td>{values["negative_pose_correlations"]}/{values["pose_correlation_count"]}</td></tr>')
    comparisons = {}
    for (family, method, arm), entries in paired.items():
        baseline = paired[(family, method, 'legacy')]
        differences = [v-baseline[k] for k,v in entries.items() if k in baseline]
        comparisons[f'{family}/{method}/{arm}'] = dict(mean_coverage_delta=float(np.mean(differences)), n=len(differences))
    family_rows = []
    for family in ('push', 'scissors', 'spray'):
        legacy = [aggregate[f'{family}/{method}/legacy'] for method in ('pca', 'pls')]
        legacy_coverage = float(np.mean([a['held_coverage'] for a in legacy]))
        legacy_held = float(np.mean([a['held_rate'] for a in legacy]))
        legacy_error_n = sum(a['error_available'] for a in legacy)
        legacy_mae = sum(a['normalised_mae']*a['error_available'] for a in legacy if a['normalised_mae'] is not None) / legacy_error_n if legacy_error_n else None
        for arm in ARMS:
            values = [aggregate[f'{family}/{method}/{arm}'] for method in ('pca', 'pls')]
            coverage = float(np.mean([a['held_coverage'] for a in values]))
            held = float(np.mean([a['held_rate'] for a in values]))
            error_n = sum(a['error_available'] for a in values)
            mae = sum(a['normalised_mae']*a['error_available'] for a in values if a['normalised_mae'] is not None) / error_n if error_n else None
            family_rows.append(f'<tr><td>{family}</td><td>{arm}</td><td>{sum(a["calibrated"] for a in values)}/{sum(a["n"] for a in values)}</td>'
                               f'<td>{fmt(coverage)}</td><td>{100*(coverage-legacy_coverage):+.1f} pp</td><td>{fmt(held)}</td>'
                               f'<td>{100*(held-legacy_held):+.1f} pp</td><td>{fmt(mae)}</td>'
                               f'<td>{"—" if mae is None or legacy_mae is None else f"{100*(mae-legacy_mae):+.1f} pp"}</td>'
                               f'<td>{sum(a["negative_correlations"] for a in values)}</td>'
                               f'<td>{sum(a["negative_pose_correlations"] for a in values)}/{sum(a["pose_correlation_count"] for a in values)}</td></tr>')
    summary = dict(reports=len(paths), production_complete=production, aggregates=aggregate, paired_vs_legacy=comparisons)
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    document = '''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Adaptive synergy FF evaluation</title>
<style>body{font-family:system-ui;max-width:1300px;margin:30px auto;padding:15px}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:7px}img{max-width:100%}.sheet{max-width:1000px}h3{margin-top:50px}</style>
<h1>シナジーの方向・振幅・把持選択：FF比較実験</h1>
<p>legacy: 固定振幅＋従来の累積最大値校正。direction: 実応答による方向選択＋等張回帰（校正方法も異なる）。range: 振幅・単調区間も選択。joint: 把持スコア・基準姿勢も選択。全条件で同じFit/Calibration/Test分割と対応するreset seedを使用。過去の実験とはreset seed設計・目標誤差測定方法が異なるため、今回のlegacyを比較基準とする。</p>
<p>校正基準：区間の全点で把持を維持する固定姿勢群が75%以上、その群の75%以上が同方向へ目標幅の1%以上変位、平均端点差5%以上、目標範囲内Coverage5%以上。range/jointは局所逆転2%以下・累積逆転5%以下。これらは事前の工学的基準であり統計的有意性ではない。</p>
<p>校正不成立はTestを実行せず、全条件平均のCoverage・把持維持率には0として算入。MAEは制御器が成立し実行できた条件のみの条件付き平均であり、単独で優劣を判断しない。目標誤差は全範囲9目標への実際の独立したFF試行で測定。エラーありの試行も除外しない。Coverageは把持維持した観測値の最小最大の幅で、区間内の全値への到達を保証しない。</p>
<p>各スコアは独立reset試行（把持100、操作ramp50、保持50 steps）。連続往復スイープではない。画像はあらかじめ決めたTest先頭姿勢。Fit点は各候補把持姿勢からの非クリップ線形投影で、制御経路の最近傍スコアではない。範囲外点も表示。破線と灰色の生応答を区別する。</p>'''
    document += f'<p>レポート数: {len(paths)} / production complete: {production}</p>'
    if not production:
        document += '<p style="padding:16px;background:#fff1d6;border:2px solid #c87900"><strong>部分結果の評価報告です。</strong>未完了タスクは以下の集計に含みません。Calibration不成立として評価された条件の0とは異なります。<a href="../review/index.html">全15タスクの状況・未完了理由はこちら</a>。</p>'
    document += '<h2>PCA/PLS統合：同一実験内のlegacyとの差</h2><table><tr><th>Task</th><th>Arm</th><th>校正成立</th><th>Held Coverage</th><th>Coverage差</th><th>Held rate</th><th>Held差</th><th>正規化MAE</th><th>MAE差</th><th>負相関（条件平均）</th><th>負相関（姿勢別）</th></tr>' + ''.join(family_rows) + '</table>'
    joint = [aggregate[f'{family}/{method}/joint'] for family in ('push','scissors','spray') for method in ('pca','pls')]
    document += (f'<h2>joint方式で選択された物理区間</h2><p>評価済み{sum(a["n"] for a in joint)}条件中、従来の±1を越えた条件 {sum(a["expanded"] for a in joint)}、'
                 f'非対称区間 {sum(a["asymmetric"] for a in joint)}、把持基準rho=0を含まない区間 {sum(a["excludes_grasp"] for a in joint)}、'
                 f'把持・preload候補を変更した条件 {sum(a["changed_candidate"] for a in joint)}、符号反転 {sum(a["reversed"] for a in joint)}。</p>')
    document += '<table><tr><th>Task</th><th>Method</th><th>Arm</th><th>校正成立</th><th>Held Coverage</th><th>Held rate</th><th>正規化MAE</th><th>MAE条件数</th><th>負相関条件数</th><th>負相関姿勢/相関計算可能姿勢</th></tr>' + ''.join(rows) + '</table>'
    document += ''.join(details) + '</html>'
    (args.output_dir/'index.html').write_text(document, encoding='utf-8')
    print(f'validated {len(paths)} reports; wrote {args.output_dir / "index.html"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
