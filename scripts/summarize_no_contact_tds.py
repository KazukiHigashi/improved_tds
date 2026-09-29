"""Compare the frozen contact-gated TDS run with a no-contact-gate rerun."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


TASKS = [f'{category}_lite_{variant}' for category in ('push', 'spray')
         for variant in range(1, 6)]
LABELS = ('pca_1d', 'pls_1d', 'pca_2d', 'pls_2d')
ALL_TASKS = [f'{category}_lite_{variant}' for category in ('scissors','push','spray')
             for variant in range(1,6)]


def all_outcomes(config):
    return [outcome for pose in config['calibration'] for outcome in pose['outcomes']]


def calibration_curve(config):
    x = np.asarray(config['diagnostic_levels'])
    y = np.asarray([[o['achieved'] if o['valid'] else np.nan
                     for o in pose['outcomes']] for pose in config['calibration']])
    return x, np.asarray([np.mean(y[np.isfinite(y[:, j]), j])
                          if np.isfinite(y[:, j]).any() else np.nan
                          for j in range(y.shape[1])])


def fmt(value, percent=False):
    if value is None:
        return '—'
    return f'{100*value:.1f}%' if percent else f'{value:.4f}'


def protocol_section(new_root, old_root):
    rows=[]
    for task in ALL_TASKS:
        report_root=old_root if task.startswith('scissors') else new_root
        report=json.loads((report_root/task/'report.json').read_text(encoding='utf-8'))
        meta=report['database_metadata']
        contract=meta['contract']
        target=contract['target_range']
        tolerance=float(contract['success_tolerance'])
        dt=7./report['configurations']['pca_1d']['calibration'][0]['outcomes'][0]['executed_steps']
        if not np.isclose(report['trajectory_seconds'],5.) or not np.isclose(report['operation_hold_seconds'],2.):
            raise ValueError(f'nonstandard TDS trajectory duration for {task}')
        if meta['rows']!=200 or meta['success_hold_steps']!=5 or meta['contract']['per_bin']!=20:
            raise ValueError(f'nonstandard success-posture database for {task}')
        grasp=('Palm追従球内のはさみ関節中心' if task.startswith('scissors') else
               '2指以上の対象物接触＋落下・転倒なし')
        tds=(f'Palm球内（評価半径{float(report["palm_sphere_radius"]):.3f} m）を2秒間維持。転倒は診断値'
             if task.startswith('scissors') else
             '落下・転倒なしを2秒間維持。接触数は不問')
        units='rad' if not task.startswith('push') else 'm'
        cells=(task,f'{target[0]:g}–{target[1]:g} {units}',f'±{tolerance:g} {units}',
               grasp,tds,f'{dt:.2f} s',f'{round(5/dt)}+{round(2/dt)} steps')
        rows.append('<tr>'+''.join(f'<td>{html.escape(x)}</td>' for x in cells)+'</tr>')
    return '''<section id="protocol"><h2>論文用：学習・DB収集・TDSの成功条件</h2>
<p><strong>確認結果：全15タスクのTDS軌道は各目標姿勢へ5.0秒の線形指令ランプ、その後2.0秒の終端保持です。</strong>
これはRLポリシーによる到達時間ではありません。Scissorsは制御周期0.04秒で125＋50ステップ、Push・Sprayは0.02秒で250＋100ステップです。
独立Calibrationの17スコアは各点で初期姿勢にリセットしてこの7秒軌道を実行します。連続Sweepも各区間に同じ5秒＋2秒を適用します。
線形なのは指令値の補間であり、接触や入力変化率制限を受けた実関節軌道が厳密な直線になるという意味ではありません。</p>
<table><tr><th>手続き</th><th>角度・深さの条件</th><th>把持・転倒の条件</th><th>時間条件・入力</th></tr>
<tr><td>強化学習・Validation</td><td>Scissors ±0.03 rad、Spray ±0.03 rad、Pushは可動幅の7.5%（下表）。</td>
<td>Scissors：関節中心がPalm追従球内。Push・Spray：2指以上接触かつ落下・転倒なし。</td>
<td>条件を5制御ステップ連続で達成すると stable success。SACによる逐次行動。Validationは100,000ステップごと。</td></tr>
<tr><td>成功姿勢DB収集</td><td>同じタスク成功条件。成功時の実測ツール値が要求された10分割binに入ることも必要。</td>
<td>同じ把持条件を5ステップ連続。数値異常、非有限姿勢、重複姿勢は採用しない。</td>
<td>best_success_policy.zipを固定し、決定論的／確率的行動の試行を交互に使用。各bin20姿勢、計200。成功窓の最終姿勢を保存。追加の2秒保持はない。</td></tr>
<tr><td>TDS Calibration（17スコア）</td><td>目標角度誤差による合否判定なし。PCA/PLSの再構成姿勢へ直接指令。</td>
<td>Scissors：Palm球。Push・Spray：今回の再実験では接触数を問わず落下・転倒なし。数値異常・途中終了は無効。</td>
<td>各スコア・初期姿勢ごとにリセット、5秒線形ランプ＋2秒終端保持（保持中は毎ステップ条件成立）。各方式8初期姿勢×17点。</td></tr>
<tr><td>TDS Test・逆写像目標</td><td>Calibrationから求めた逆写像域の未知目標9点の絶対誤差を測定。RLの許容値で成功判定しない。</td>
<td>Calibrationと同じ保持判定。接触不足は別の診断値として記録。</td>
<td>別の8初期姿勢。各目標で5秒線形ランプ＋2秒終端保持。RLポリシーは使わない。</td></tr></table>
<p class="note">重要な差：Push・Sprayの学習／DBでは接触数≥2を成功条件とするが、新TDS実験は接触数を合否に使わない。
Scissorsでは接触数と転倒は成功条件ではない。TDSの「有効」はタスク成功とは同義ではなく、今回のPush・Sprayでは床支持だけでも有効になり得る。
SprayのTDSでは床接触を明示的に有効化し、途中で無効になれば実験を停止する。
Sprayの学習・DB収集では床接触無効化オプションを指定しておらず、環境既定の有効状態を用いる。
5ステップの物理時間はScissorsで0.20秒、Push・Sprayで0.10秒。TDSの5秒ランプ中は把持条件の全時刻成立を要求せず、終端2秒で要求する。ただしPush・Sprayの落下・転倒はランプ中でも環境が終了し、その試行は無効となる。</p>
<details><summary>強化学習の共通設定とカテゴリ別報酬</summary>
<p>全15タスクはSAC、上限300万環境ステップ、4学習プロセス×各4環境、学習率5×10⁻⁵、固定エントロピー係数0.02、HERなし。
100,000ステップごとにValidationとチェックポイント保存を行い、Validation成功率95%以上を2回連続で満たすと早期終了する。
共通報酬係数は進捗1、把持1、成功ボーナス5、失敗ペナルティ25、制御変化ペナルティ0.1、制御量ペナルティ0.002。</p>
<p>カテゴリ固有の項はScissorsが粗・精密追従各1と相互作用0.25（接触補助報酬0）、Pushが追従2と親指接触0.25、Sprayが追従2と示指近接0.25。
Scissorsの接触数・転倒は成功判定では診断値であり、Push・Sprayの接触数不足は成功しないが、落下・転倒が物理失敗となる。
Push・Sprayの入力変化率は正規化行動1.25/s、Scissorsは0.04秒ごとに最大0.05で同じ1.25/sに相当する。</p></details>
<details><summary>全15タスクの目標範囲・RL/DB許容値・TDS制御周期</summary><div class="scroll"><table>
<tr><th>タスク</th><th>RL目標範囲</th><th>RL/DB誤差許容</th><th>RL/DB把持</th><th>TDS保持</th><th>制御周期</th><th>ランプ＋保持</th></tr>'''+''.join(rows)+'''</table></div></details>
<p>出典：各タスクの成功姿勢DB metadata、評価report.json、学習manifest、実行コード。
DBは成功姿勢だけを保存するが、TDS Calibrationは再構成姿勢の物理再生であり、RL/DB成功判定とは異なる。</p></section>'''


def gallery_section(new_root):
    blocks=[]
    replay_mismatches=[]
    for task in ALL_TASKS:
        path=new_root/'trajectory_frames'/task/'manifest.json'
        if not path.is_file():
            raise FileNotFoundError(f'17-score gallery is incomplete: {task}')
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('replay_validity_mismatches',0):
            replay_mismatches.append(f'{task}: {data["replay_validity_mismatches"]}/34点')
        if any(len(data['methods'][method])!=17 for method in ('pca','pls')):
            raise ValueError(f'gallery must contain 17 PCA and 17 PLS scores: {task}')
        methods=[]
        for method in ('pca','pls'):
            cards=[]
            for item in data['methods'][method]:
                label=(f"score {item['score_index']:02d}/17, rho={item['rho']:+.3f}, "
                       f"tool={item['achieved']:.4f}, valid={item['valid']}")
                if item.get('validity_matches_report') is False:
                    label+=f"（画像再生はvalid={item['replay_valid']}）"
                cards.append(f'<details class="score"><summary>{html.escape(label)}</summary>'
                    f'<a href="{html.escape(item["image"])}"><img loading="lazy" '
                    f'src="{html.escape(item["image"])}" alt="{html.escape(label)}"></a></details>')
            methods.append(f'<details class="method"><summary>{method.upper()}：17スコア</summary>'
                +''.join(cards)+'</details>')
        blocks.append(f'<details class="task"><summary>{task}：初期姿勢 '
            f'{data["calibration_pose"]}、PCA/PLS各17点</summary>'+''.join(methods)+'</details>')
    replay_note=('画像再生と保存済み評価の有効判定が異なった境界点：'
        + '、'.join(replay_mismatches) + '。図は画像再生軌道、表は保存済み評価を示す。'
        if replay_mismatches else '画像再生と保存済み評価の有効判定は一致した。')
    return '''<section id="trajectories"><h2>17点Calibration軌道の25ステップごとの画像</h2>
<p>全15タスクについて、Calibration初期姿勢8種類のうち先頭の1種類を固定し、PCA/PLSそれぞれの17スコアを独立に再生した。
各スコアは初期姿勢から5秒で再構成姿勢へ移動し、2秒保持する。ステップ0と、その後25ステップごとのフレームを1枚のコンタクトシートにした。
Push・Sprayでは25ステップ＝0.5秒、Scissorsでは1.0秒。画像のcontactsは診断表示であり、Push・Sprayの新TDS有効判定には使わない。
各タスクの2DoF固定把持スコアは0で対応する1DoFと同一軌道のため、同一画像を重複掲載しない。</p>
<p class="note">これは各タスクの8初期姿勢すべてを視覚検査したものではない。数値評価は8姿勢で実施済みだが、画像による把持確認は下記の代表1姿勢に限る。
'''+html.escape(replay_note)+'''</p>'''+''.join(blocks)+'</section>'


def inspect(old, new):
    before, after = old['summary'], new['summary']
    old_outcomes, new_outcomes = all_outcomes(old), all_outcomes(new)
    target_outcomes = [trial['outcome'] for pose in new['test']
                       for trial in pose['target_trials']]
    if len(old_outcomes) != len(new_outcomes):
        raise ValueError('calibration sample counts differ')
    return dict(
        old_grasp_score=old['grasp_score'],new_grasp_score=new['grasp_score'],
        old_valid=sum(o['valid'] for o in old_outcomes),
        new_valid=sum(o['valid'] for o in new_outcomes),
        total=len(new_outcomes),
        new_valid_without_endpoint_contact=sum(o['valid'] and not o['contact_held']
                                                for o in new_outcomes),
        new_target_without_endpoint_contact=sum(o['valid'] and not o['contact_held']
                                                for o in target_outcomes),
        new_target_total=len(target_outcomes),
        old_fit=bool(old['selection']), new_fit=bool(new['selection']),
        old_fit_coverage=old['selection']['coverage'] if old['selection'] else None,
        new_fit_coverage=new['selection']['coverage'] if new['selection'] else None,
        old_inverse_domain=before['inverse_domain_fraction'],
        new_inverse_domain=after['inverse_domain_fraction'],
        old_independent_coverage=before['independent_coverage'],
        new_independent_coverage=after['independent_coverage'],
        old_continuous_coverage=before['continuous_coverage'],
        new_continuous_coverage=after['continuous_coverage'],
        old_isotonic_mae=before['controller_comparison'].get('isotonic', {}).get('held_mae'),
        new_isotonic_mae=after['controller_comparison'].get('isotonic', {}).get('held_mae'),
        old_ramp_mae=before['controller_comparison'].get('ramp', {}).get('held_mae'),
        new_ramp_mae=after['controller_comparison'].get('ramp', {}).get('held_mae'),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path,
        default=Path('analysis/paper15_unified_tds_corrected_20260928'))
    parser.add_argument('--rerun', type=Path,
        default=Path('analysis/paper15_no_contact_floor_on_20260928'))
    args = parser.parse_args()
    old_root, new_root = args.baseline, args.rerun
    reports = {}
    for task in TASKS:
        old_path, new_path = old_root/task/'report.json', new_root/task/'report.json'
        if not old_path.is_file() or not new_path.is_file():
            raise FileNotFoundError(f'incomplete baseline/rerun pair: {task}')
        old = json.loads(old_path.read_text(encoding='utf-8'))
        new = json.loads(new_path.read_text(encoding='utf-8'))
        if old['database_sha256'] != new['database_sha256']:
            raise ValueError(f'database changed for {task}')
        if old['calibration_poses'] != new['calibration_poses'] or old['test_poses'] != new['test_poses']:
            raise ValueError(f'pose split changed for {task}')
        if new['evaluation_support_criterion'] != 'no_fall':
            raise ValueError(f'wrong support criterion for {task}')
        if task.startswith('spray') and not new['floor_contact_forced_enabled']:
            raise ValueError(f'floor not forced on for {task}')
        reports[task] = (old, new)

    entries = {task: {label: inspect(old['configurations'][label],new['configurations'][label])
                      for label in LABELS} for task,(old,new) in reports.items()}
    (new_root/'comparison_summary.json').write_text(json.dumps(entries,indent=2),encoding='utf-8')
    figures = new_root/'comparison_figures'
    figures.mkdir(exist_ok=True)
    sections = []
    for task,(old,new) in reports.items():
        fig, axes = plt.subplots(2,2,figsize=(11,8),sharey=True)
        rows = []
        for ax,label in zip(axes.flat,LABELS):
            before,after = old['configurations'][label],new['configurations'][label]
            x,y = calibration_curve(before)
            xn,yn = calibration_curve(after)
            ax.plot(x,y,'o--',label='contact-gated',alpha=.8,markersize=3)
            ax.plot(xn,yn,'o-',label='no contact gate',markersize=3)
            if after['selection']:
                s=after['selection']
                ax.axvspan(min(s['rho_start'],s['rho_end']),max(s['rho_start'],s['rho_end']),
                           color='#72ae7c',alpha=.14,label='new fit interval')
            ax.set_title(label.upper())
            ax.set_xlabel('physical synergy score')
            ax.set_ylabel('tool angle/depth')
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
            m=entries[task][label]
            rows.append('<tr>'+''.join(f'<td>{html.escape(str(cell))}</td>' for cell in (
                label.upper(),f"{m['old_valid']}/{m['total']}",f"{m['new_valid']}/{m['total']}",
                m['new_valid_without_endpoint_contact'],
                f"{m['old_grasp_score']:+.2f} → {m['new_grasp_score']:+.2f}",
                f"{int(m['old_fit'])} → {int(m['new_fit'])}",
                f"{fmt(m['old_inverse_domain'],True)} → {fmt(m['new_inverse_domain'],True)}",
                f"{fmt(m['old_independent_coverage'],True)} → {fmt(m['new_independent_coverage'],True)}",
                f"{fmt(m['old_continuous_coverage'],True)} → {fmt(m['new_continuous_coverage'],True)}",
                f"{fmt(m['old_isotonic_mae'])} → {fmt(m['new_isotonic_mae'])}",
                f"{fmt(m['old_ramp_mae'])} → {fmt(m['new_ramp_mae'])}"))+'</tr>')
        fig.suptitle(f'{task}: calibration response and selected no-contact fit intervals')
        fig.tight_layout()
        fig.savefig(figures/f'{task}.png',dpi=140)
        plt.close(fig)
        sections.append(f'<section><h2>{task}</h2><div class="scroll"><table><thead><tr>'
            '<th>方式</th><th>旧有効点</th><th>新有効点</th><th>新有効点のうち終端接触不足</th><th>固定第2スコア</th>'
            '<th>Fit成立</th><th>逆写像域</th><th>独立Sweep Coverage</th>'
            '<th>連続Sweep Coverage</th><th>等張MAE</th><th>Ramp MAE</th>'
            '</tr></thead><tbody>'+''.join(rows)+'</tbody></table></div>'
            f'<img src="comparison_figures/{task}.png" alt="{task} calibration curves"></section>')
    group_rows=[]
    for category in ('push','spray'):
        vals=[entries[task][label] for task in TASKS if task.startswith(category)
              for label in LABELS]
        group_rows.append('<tr>'+''.join(f'<td>{x}</td>' for x in (
            category.capitalize(),
            f"{sum(v['old_valid'] for v in vals)}/{sum(v['total'] for v in vals)}",
            f"{sum(v['new_valid'] for v in vals)}/{sum(v['total'] for v in vals)}",
            f"{sum(v['old_fit'] for v in vals)}/20 → {sum(v['new_fit'] for v in vals)}/20",
            f"{sum(v['new_valid_without_endpoint_contact'] for v in vals)}",
            f"{sum(v['new_target_without_endpoint_contact'] for v in vals)}/"
            f"{sum(v['new_target_total'] for v in vals)}"))+'</tr>')
    page='''<!doctype html><html lang="ja"><meta charset="utf-8"><title>Push/Spray 接触数判定除去 TDS 再実験</title>
<style>body{font:15px/1.55 system-ui,sans-serif;max-width:1450px;margin:auto;padding:20px;color:#213547;background:#f2f5f8}
section{background:white;padding:20px;margin:20px 0;border-radius:9px}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid #dae2e8;padding:7px;text-align:left}th{background:#e7eff4;white-space:nowrap}
.scroll{overflow-x:auto}img{max-width:100%}.note{border-left:4px solid #b87926;padding:10px;background:#fff6e7}
details{margin:9px 0;border:1px solid #dae2e8;border-radius:6px;padding:8px;background:#f9fbfd}
details.task{margin:16px 0;background:#eaf2f7}details.method{background:white}
details.score{margin-left:15px}summary{cursor:pointer;font-weight:600}details.score img{display:block;margin:12px 0}
</style>
<h1>Push・Spray 接触数ゲート除去によるTDS再評価</h1>
<p><a href="#protocol">学習・DB・TDSの条件</a> ｜ <a href="#trajectories">全15タスクの25ステップ画像</a></p>
<p>同一の学習済みモデル、成功姿勢DB、初期姿勢分割、5秒軌道＋2秒保持を使用。新条件は接触数を問わず、落下・転倒・数値異常・途中終了を除外。Sprayの床接触は常時有効。</p>
<section><h2>全体</h2><table><tr><th>群</th><th>旧有効点</th><th>新有効点</th><th>Fit成立</th><th>新Cal有効点のうち終端接触不足</th><th>新Test目標試行のうち終端接触不足</th></tr>'''+''.join(group_rows)+'''</table>
<p class="note">終端接触不足の点は、新基準では有効でも手で安定保持した証拠にはならない。床支持だけの可能性を含む。
第2スコアの選択も有効点数に依存するため、全候補が有効なら絶対値が小さい0が選ばれ、2DoFが事実上1DoFとなる点に注意する。
MAEは各条件で選ばれた逆写像範囲内の未知目標に対する値であり、範囲が異なる場合は単純な優劣比較をしない。</p></section>'''+''.join(sections)+protocol_section(new_root,old_root)+gallery_section(new_root)+'</html>'
    (new_root/'index.html').write_text(page,encoding='utf-8')
    print(new_root/'index.html')


if __name__ == '__main__':
    main()
