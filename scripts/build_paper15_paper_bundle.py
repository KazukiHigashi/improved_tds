"""Build one non-destructive, self-contained paper data directory for 15 tasks.

Selected Scissors TDS comes from the corrected palm-sphere run; selected Push
and Spray TDS comes from the later no-contact-gate/floor-enabled run. The
earlier contact-gated Push/Spray reports are retained as comparison data.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from summarize_no_contact_tds import calibration_curve, gallery_section, protocol_section

_japanese_font=Path('C:/Windows/Fonts/YuGothM.ttc')
if _japanese_font.is_file():
    font_manager.fontManager.addfont(str(_japanese_font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(_japanese_font)).get_name()


TASKS=[f'{category}_lite_{variant}' for category in ('scissors','push','spray')
       for variant in range(1,6)]
METHODS=('pca_1d','pls_1d','pca_2d','pls_2d')


def method_section():
    return '''<section id="tds_method"><h2>PCA/PLSシナジーの計算と1・2自由度TDS制御</h2>
    <p>各タスクの成功姿勢DBから、実現した最終ハンド関節姿勢を200件だけ使用する。姿勢行列を <code>Q∈ℝ<sup>200×d</sup></code>、同じ試行で実測したツール角度（Pushは押し込み深さ）を <code>y∈ℝ<sup>200</sup></code> とする。タスクごとに全200姿勢の平均 <code>q̄</code> で<strong>一括中心化</strong>し、<code>X=Q−1q̄ᵀ</code> とする。初期姿勢ごとの中心化や関節ごとの分散標準化はしない。PLSの教師信号は指令目標ではなく、DBに記録された<strong>実測ツール値</strong>である。</p>
    <div class="scroll"><table><tr><th>方式</th><th>操作方向 v<sub>op</sub></th><th>第2方向 v<sub>g</sub>（把持候補）</th></tr>
    <tr><td>PCA</td><td><code>X=UΣVᵀ</code> の第1右特異ベクトル（PC1）。姿勢分散が最大の単位方向。</td><td>第2右特異ベクトル（PC2）。第1方向に直交する最大残差分散方向。</td></tr>
    <tr><td>PLS</td><td>中心化した <code>X, y</code> に <code>PLSRegression(n_components=1, scale=False)</code> を適用した第1 <code>x_weights_</code> を単位化。姿勢と実測ツール値の共変動を利用する。</td><td><code>R=X−(Xv<sub>op</sub>)v<sub>op</sub>ᵀ</code> の第1右特異ベクトルを操作方向に直交化して単位化。PLSの第2成分ではない。</td></tr></table></div>
    <p>相関が定義できる場合、操作方向はDB上の <code>corr(Xv<sub>op</sub>, y)≥0</code> となる向きに揃える。第2方向は記録された把持指標との相関が正になる向きへ揃える。ただし<strong>方向の符号だけで物理Sweepの単調増加や安定把持は保証されない</strong>。第2方向は「把持シナジー」と確定したものではなく候補である。</p>
    <p>各単位方向 <code>v</code> に対し、使用するアクチュエータ指令範囲の下限 <code>l<sub>j</sub></code>、上限 <code>h<sub>j</sub></code>からスケール <code>A(v)=min{(h<sub>j</sub>−l<sub>j</sub>)/(2|v<sub>j</sub>|) : |v<sub>j</sub>|&gt;0}</code> を求める。DB姿勢の操作得点は <code>ρ<sub>i</sub>=[(q<sub>i</sub>−q̄)·v<sub>op</sub>]/A(v<sub>op</sub>)</code>。独立Calibrationでは、この観測得点の最小値×1.5から最大値×1.5まで17点を等間隔に試す。したがって物理得点 <code>ρ</code> は−1～1に固定されない（無次元）。</p>
    <div class="scroll"><table><tr><th>制御</th><th>再構成した絶対目標姿勢</th><th>第2得点の扱い</th></tr>
    <tr><td>1自由度</td><td><code>q<sub>cmd</sub>(ρ)=clip[q̄+A(v<sub>op</sub>)ρv<sub>op</sub>, l, h]</code></td><td><code>g=0</code></td></tr>
    <tr><td>2自由度</td><td><code>q<sub>cmd</sub>(ρ,g)=clip[q̄+A(v<sub>op</sub>)ρv<sub>op</sub>+A(v<sub>g</sub>)gv<sub>g</sub>, l, h]</code></td><td>Calibrationの8初期姿勢で <code>g∈{−1,−0.5,0,0.5,1}</code> を試し、終端2秒の有効件数が最大の一定値を選ぶ。同数なら <code>|g|</code> が小さい値を選ぶ。</td></tr></table></div>
    <p>各試行は保存された初期姿勢へリセットし、その時点の実測ハンド姿勢から上式の<strong>絶対目標姿勢</strong>へ関節指令を5秒で線形補間し、さらに2秒保持する。固定準備オフセット、RLによる前処理、状態フィードバックによる角度補正、リトライはない。2自由度の <code>g</code> は一度選ぶとSweep・逆写像試験中は固定であり、把持→操作の二段階制御ではない。現在の採用15タスクでは全て <code>g=0</code> が選択され、2自由度の実軌道は対応する1自由度と同一である。</p>
    <p>Calibrationでは17得点×8初期姿勢のツール応答を取得し、同じ6/8以上の姿勢で保持できる連続3点以上の区間を候補とする。応答方向の一致、RL目標幅の5%以上の変化量、局所的な非単調性の上限を確認して区間を選ぶ。その区間を <code>u∈[−1,1]</code> に線形正規化し、平均応答へ<strong>等張回帰</strong>と<strong>クリップ付き線形ランプ関数</strong>を別々にフィットする。両フィットとRL目標範囲の共通部分だけを逆写像試験の対象域とし、未知目標9点×別の8初期姿勢で誤差を測る。テスト応答は区間選択・フィットに使わない。</p>
    <p><strong>指標：</strong>「逆写像可能な目標範囲 [%]」=<code>100×(共通逆写像目標域の幅)/(RL目標範囲の幅)</code>。「独立/連続Coverage [%]」はそれぞれ独立17点、選択区間の往復連続Sweepで得た有効ツール値の幅をRL目標幅で割り、Test 8姿勢で平均したもの。「正規化held MAE [%]」=<code>100×mean(|実測値−未知目標値|)/RL目標幅</code>で、終端保持に成功した逆写像試行だけを対象とする。誤差0%は全域Coverageを意味せず、逆写像域外はMAEに含めない。RLの角度許容誤差をこのFF制御の成功判定には使わない。</p>
    <p>実装：<a href="scissors_scaled_evaluation_source_snapshot/improved_tds/synergy/two_axis.py">軸の算出</a> ／ <a href="scissors_scaled_evaluation_source_snapshot/improved_tds/experiments/reach_tds_direct.py">再構成・評価</a> ／ <a href="scissors_scaled_evaluation_source_snapshot/improved_tds/synergy/adaptive_range.py">Calibration・逆写像</a>。</p></section>'''


def scissors_section(root, reports):
    """Keep the selected Scissors conditions and results inside the main report."""
    condition_rows=[]
    result_rows=[]
    for variant in range(1,6):
        task=f'scissors_lite_{variant}'
        report=reports[task]
        training=json.loads((root/task/'training/model_training_summary.json').read_text(encoding='utf-8'))
        validation=json.loads((root/task/'training/validation_history.json').read_text(encoding='utf-8'))
        db=report['database_metadata']
        pca=report['configurations']['pca_1d']
        pls=report['configurations']['pls_1d']
        goal_width=float(np.diff(report['target_range'])[0])
        scores=lambda c:f"{min(c['diagnostic_levels']):.3f}–{max(c['diagnostic_levels']):.3f}"
        condition_rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (
            task, f"{report['target_range'][0]:.2f}–{report['target_range'][1]:.2f}",
            training['total_timesteps'], f(validation['best']['success_rate'],True),
            db['episodes'], db['rows'], scores(pca), scores(pls),
            f'{4 if variant<=3 else 3 if variant==4 else 2}指'))+
            f'<td><a href="scissors_evaluation_source_snapshot/improved_tds/assets/exp_scissors/exp_scissors_{variant}.xml">XML {variant}</a></td></tr>')
        for label,c in (('PCA',pca),('PLS',pls)):
            outcomes=[o for pose in c['calibration'] for o in pose['outcomes']]
            valid=sum(bool(o['valid']) for o in outcomes)
            selected=c['selection']
            iso=c['summary']['controller_comparison']['isotonic']
            ramp=c['summary']['controller_comparison']['ramp']
            result_rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (
                task,label,f'{valid}/{len(outcomes)}',f(1-valid/len(outcomes),True),
                f"{selected['rho_start']:.3f}–{selected['rho_end']:.3f}" if selected else '—',
                f(c['summary']['inverse_domain_fraction'],True),
                f(c['summary']['independent_coverage'],True),
                f(c['summary']['continuous_coverage'],True),
                f(iso['held_mae']/goal_width,True,2),
                f(ramp['held_mae']/goal_width,True,2)))+'</tr>')
    radii={float(reports[f'scissors_lite_{i}']['palm_sphere_radius']) for i in range(1,6)}
    if len(radii)!=1:
        raise ValueError('selected Scissors reports use inconsistent evaluation radii')
    radius=next(iter(radii))
    radius_note=(f'<p class="note">この表・タスク別の数値と図は、現在採用しているTDS評価のPalm球半径'
        f'{radius:.3f} mによる結果です。RL/成功姿勢DBの半径は0.050 mのままです。</p>')
    return '''<section id="scissors"><h2>Scissors 1–5：採用条件と評価結果</h2>'''+radius_note+'''
    <p>この欄と下のタスク別図は、Palm追従球を用いた<strong>修正版TDS評価</strong>を採用する。学習・成功姿勢DBは全15タスクの共通3M-step実験から取得したもの。Push・Sprayの旧接触ゲート実験や別時点のScissors試験を混在させていない。</p>
    <p>ScissorsのRL/ValidationとDB収集では、はさみ関節中心がPalm追従球（半径0.05 m）内にあり、ツール角度誤差が±0.03 radのときに成功とする。成功条件は5ステップ保持し、DBはBest Success Policyによる成功姿勢を各角度bin 20件、計200件保存する。TDS評価は角度誤差で合否判定せず、終端2秒のPalm球内保持でCalibration点を判定する。接触数と転倒はScissorsの成功判定に含めない。</p>
    <p>全タスクで200姿勢を一括中心化してPCA/PLSを計算する。表のScore範囲はDBで観測された得点の最小・最大をそれぞれ1.5倍した17点の<strong>診断Sweep全域</strong>であり、逆写像に採用された単調区間とは異なる。各点を8初期姿勢から独立に5秒線形移動＋2秒保持で再生し、別の8姿勢で未知目標9点を検証した。PCA/PLS×2自由度も評価したが、Scissorsで採択された第2スコアは全て0なので1自由度と同じ軌道である。変種ごとの初期指設定は<a href="scissors_evaluation_source_snapshot/improved_tds/environments/legacy_scissors.py">環境実装</a>、器具・接触等の物理設定は表の各XMLを参照。</p>
    <div class="scroll"><table><caption>学習・DB・Score Sweepの条件</caption><tr><th>タスク</th><th>RL目標角度 rad</th><th>実学習Step</th><th>Best Validation成功率</th><th>DB試行数</th><th>成功姿勢数</th><th>PCA 17点Score範囲</th><th>PLS 17点Score範囲</th><th>初期指設定</th><th>MuJoCoモデル</th></tr>'''+''.join(condition_rows)+'''</table></div>
    <div class="scroll"><table><caption>ScissorsのPCA/PLS 1自由度結果。MAEは保持成功した未知目標のみをRL目標幅で正規化した値。</caption><tr><th>タスク</th><th>方式</th><th>有効Calibration点</th><th>構築失敗率</th><th>採択Score区間 [無次元]</th><th>逆写像可能な目標範囲 [%]</th><th>独立Coverage [%]</th><th>連続Coverage [%]</th><th>等張held MAE [%]</th><th>Ramp held MAE [%]</th></tr>'''+''.join(result_rows)+'''</table></div>
    <p class="note">逆写像のMAEは較正で選んだ目標区間内に限るため、全目標範囲への達成度は逆写像目標域・Coverageと併読する。Calibration点の無効はPCA/PLS自体の計算失敗ではなく、再構成姿勢へ移動後のPalm球保持不成立を意味する。Scissors 4は無効点が多く、Scissors 2は独立Coverageと連続Coverageが大きく乖離する。</p>
    <p><a href="#scissors_lite_1">Scissors 1の図</a> ／ <a href="#scissors_lite_2">2の図</a> ／ <a href="#scissors_lite_3">3の図</a> ／ <a href="#scissors_lite_4">4の図</a> ／ <a href="#scissors_lite_5">5の図</a>。各タスク欄にCalibration成否図、DB・Sweep・フィット応答図、未知目標の逆写像誤差図を掲載。</p></section>'''


def scissors_radius_comparison(root, reports):
    baseline_paths=[root/f'scissors_lite_{i}'/'baseline_palm005_report.json'
                    for i in range(1,6)]
    if not any(path.is_file() for path in baseline_paths):
        return ''
    if not all(path.is_file() for path in baseline_paths):
        raise ValueError('incomplete Scissors 0.05 m baseline archive')
    baseline={f'scissors_lite_{i}':json.loads(path.read_text(encoding='utf-8'))
              for i,path in enumerate(baseline_paths,1)}
    rows=[]
    for task in baseline:
        old,new=baseline[task],reports[task]
        if not np.isclose(old['palm_sphere_radius'],.05) or not np.isclose(new['palm_sphere_radius'],.075):
            raise ValueError(f'wrong Scissors radius comparison: {task}')
        for method in ('pca_1d','pls_1d'):
            a,b=old['configurations'][method],new['configurations'][method]
            valid=lambda c:sum(bool(o['valid']) for row in c['calibration'] for o in row['outcomes'])
            mae=lambda c:c['summary']['controller_comparison']['isotonic']['held_mae']
            width=float(np.diff(new['target_range'])[0])
            rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (
                task,method.upper(),f'{valid(a)}/136 → {valid(b)}/136',
                f"{f(a['summary']['inverse_domain_fraction'],True)} → {f(b['summary']['inverse_domain_fraction'],True)}",
                f"{f(a['summary']['independent_coverage'],True)} → {f(b['summary']['independent_coverage'],True)}",
                f"{f(a['summary']['continuous_coverage'],True)} → {f(b['summary']['continuous_coverage'],True)}",
                f'{f(mae(a)/width,True,2)} → {f(mae(b)/width,True,2)}'))+'</tr>')
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    pairs=[(task,method) for task in baseline for method in ('pca_1d','pls_1d')]
    labels=[f'Sc{task[-1]} {method[:3].upper()}' for task,method in pairs]
    x=np.arange(len(pairs))
    for ax,(title,getter) in zip(axes,(
        ('有効Calibration点 [%]',lambda task,c:100*sum(o['valid'] for row in c['calibration']
            for o in row['outcomes'])/136),
        ('逆写像可能な目標範囲 [% of RL目標幅]',lambda task,c:100*c['summary']['inverse_domain_fraction']),
        ('等張held MAE [% of RL目標幅]',lambda task,c:100*c['summary']['controller_comparison']['isotonic']['held_mae']
            /float(np.diff(reports[task]['target_range'])[0])))):
        for offset,source,color,label in ((-.18,baseline,'#7997ad','0.05 m'),
                                          (.18,reports,'#d48535','0.075 m')):
            ax.bar(x+offset,[getter(task,source[task]['configurations'][method])
                for task,method in pairs],width=.36,color=color,label=label)
        ax.set(title=title,xticks=x,xticklabels=labels)
        ax.tick_params(axis='x',labelrotation=65)
        ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
    axes[0].legend();fig.savefig(root/'figures'/'scissors_palm_radius_comparison.png',dpi=155)
    plt.close(fig)
    return '''<section id="scissors_radius_comparison"><h2>Scissors：Palm球半径0.05→0.075 mの比較</h2>
    <p>学習済みBest Success Policy、成功姿勢DB 200件、PCA/PLS軸、Calibration/Test初期姿勢、17 Score、5秒移動＋2秒保持は同一。今回変更したのは<strong>Calibration/Testの終端Palm球判定半径のみ</strong>であり、RL/DBの成功条件（0.05 m、角度誤差±0.03 rad、5ステップ保持）は変えていない。接触数・転倒はScissors TDSの採否条件に加えていない。</p>
    <figure><img src="figures/scissors_palm_radius_comparison.png" alt="Scissors radius comparison"><figcaption>各Scissors変種のPCA/PLS 1自由度におけるCalibration成立、逆写像域、等張回帰MAE。</figcaption></figure>
    <div class="scroll"><table><tr><th>タスク</th><th>方式</th><th>有効Calibration点</th><th>逆写像可能な目標範囲 [%]</th><th>独立Coverage [%]</th><th>連続Coverage [%]</th><th>正規化等張held MAE [%]</th></tr>'''+''.join(rows)+'''</table></div>
    <p class="note">Scissors 4では有効点・逆写像域・独立Coverageが改善したが、正規化等張held MAEはPCA 2.68→4.31%、PLS 2.80→3.21%へ増加した（RL目標幅1 rad）。評価対象の逆写像域が広がるため、MAEの単純比較だけで精度の優劣を決めない。Scissors 2の連続Coverage低迷は半径変更でも改善していない。Palm球内という位置判定は、指接触の継続そのものを証明する指標ではない。</p>
    <p>25ステップ画像は新半径で再生成した。保存評価と画像再生でScissors 4のPCA最小Score 1点だけ有効判定が反転した（計170画像中1点、実現角度差0.00061 rad）。数値表は保存評価、画像は再生軌道を表す。旧0.05 mの報告JSONと図は各タスク配下に保存してある。</p></section>'''


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def copy_required(source, target):
    if not source.is_file():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(source,target)
    if digest(source)!=digest(target):
        raise IOError(f'copy hash mismatch: {source} -> {target}')
    return dict(path=target,sha256=digest(target),bytes=target.stat().st_size)


def f(value, percentage=False, digits=1):
    if value is None:
        return '—'
    return f'{value*100:.{digits}f}%' if percentage else f'{value:.4f}'


def summary_report(root, provenance):
    reports={task:json.loads((root/task/'report.json').read_text(encoding='utf-8'))
             for task in TASKS}
    rows=[]
    details=[]
    training_rows=[]
    for task in TASKS:
        report=reports[task]
        category=task.split('_')[0]
        target_width=float(np.diff(report['target_range'])[0])
        if len(report['configurations'])!=4 or report['fit_rows']!=200:
            raise ValueError(f'incomplete TDS result: {task}')
        configs=report['configurations']
        pca,pls=configs['pca_1d'],configs['pls_1d']
        if pca['grasp_score']!=0 or pls['grasp_score']!=0 or any(
            configs[f'{method}_2d']['grasp_score']!=0 for method in ('pca','pls')):
            raise ValueError(f'2-DoF is not a duplicate of 1-DoF: {task}')
        metric=lambda c,k:c['summary'][k]
        mae=lambda c:c['summary']['controller_comparison'].get('isotonic',{}).get('held_mae')
        row=[task,f(metric(pca,'inverse_domain_fraction'),True),
             f(metric(pls,'inverse_domain_fraction'),True),
             f(metric(pca,'independent_coverage'),True),
             f(metric(pls,'independent_coverage'),True),
             f(mae(pca)/target_width if mae(pca) is not None else None,True,2),
             f(mae(pls)/target_width if mae(pls) is not None else None,True,2)]
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in row)+'</tr>')
        config_rows=[]
        for label in METHODS:
            c=configs[label]
            s=c['summary']
            cal=[o for pose in c['calibration'] for o in pose['outcomes']]
            controller=s['controller_comparison']
            line=[label.upper(),f"{sum(o['valid'] for o in cal)}/{len(cal)}",
                'yes' if c['selection'] else 'no',f(s['inverse_domain_fraction'],True),
                f(s['independent_coverage'],True),f(s['continuous_coverage'],True),
                f(controller.get('isotonic',{}).get('held_mae')/target_width,True,2)
                    if controller.get('isotonic',{}).get('held_mae') is not None else '—',
                f(controller.get('ramp',{}).get('held_mae')/target_width,True,2)
                    if controller.get('ramp',{}).get('held_mae') is not None else '—']
            config_rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in line)+'</tr>')
        fig=(f'<figure><img loading="lazy" src="figures/{task}_calibration_validity.png" alt="{task} calibration validity"><figcaption>17 Score×8初期姿勢のCalibration成否と採択区間。</figcaption></figure>'
             f'<figure><img loading="lazy" src="figures/{task}_response.png" alt="{task} response"><figcaption>DB・Calibration・独立/連続Sweep・フィット曲線の角度応答。</figcaption></figure>'
             f'<figure><img loading="lazy" src="figures/{task}_inverse_error.png" alt="{task} inverse error"><figcaption>未知目標に対する等張回帰・Ramp逆写像の角度誤差。</figcaption></figure>'
             if category=='scissors' else
             f'<img loading="lazy" src="comparison_figures/{task}.png" alt="{task} old/new calibration">')
        extras=(f'<a href="{task}/contact_gated_report.json">旧接触ゲート条件のreport.json</a> ／ '
                if category!='scissors' else
                f'<a href="{task}/baseline_palm005_report.json">旧0.05 mのreport.json</a> ／ '
                if (root/task/'baseline_palm005_report.json').is_file() else '')
        details.append(f'<details class="task" id="{task}"><summary>{task} — PCA/PLS、1/2自由度</summary>'
            f'<p><a href="{task}/report.json">採用TDS report.json</a> ／ '
            f'<a href="{task}/database/success_postures.npz">成功姿勢DB</a> ／ '
            f'<a href="{task}/training/best_success_policy.zip">Best Success Policy</a> ／ '
            f'{extras}<a href="#trajectories">25ステップ画像一覧へ</a></p>'
            '<div class="scroll"><table><tr><th>方式</th><th>有効Calibration [件]</th><th>Fit</th>'
            '<th>逆写像可能な目標範囲 [%]</th><th>独立Coverage [%]</th><th>連続Coverage [%]</th><th>正規化等張held MAE [%]</th><th>正規化Ramp held MAE [%]</th></tr>'
            +''.join(config_rows)+'</table></div>'+fig+'</details>')
        db=report['database_metadata']
        training=json.loads((root/task/'training/model_training_summary.json').read_text(encoding='utf-8'))
        validation=json.loads((root/task/'training/validation_history.json').read_text(encoding='utf-8'))
        best=validation.get('best') or {}
        training_rows.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in
            (task,training.get('total_timesteps'),f(best.get('success_rate'),True),
             db.get('episodes'),db.get('rows'),db.get('achieved_range')) )+'</tr>')
        if db.get('rows')!=200 or not db.get('complete',False):
            raise ValueError(f'incomplete success-only DB: {task}')

    categories=[]
    for category in ('scissors','push','spray'):
        configs=[reports[task]['configurations'][method] for task in TASKS
                 if task.startswith(category) for method in METHODS]
        categories.append('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (
            category.capitalize(),f"{sum(c['selection'] is not None for c in configs)}/20",
            f(float(np.mean([c['summary']['inverse_domain_fraction'] for c in configs])),True),
            f(float(np.mean([c['summary']['independent_coverage'] for c in configs])),True),
            f(float(np.mean([c['summary']['continuous_coverage'] for c in configs])),True)))+'</tr>')
    make_summary_figures(root,reports)
    make_contact_comparison_figures(root,reports)
    make_scissors_task_figures(root,reports)
    style='''body{font:15px/1.6 system-ui,"Yu Gothic",sans-serif;max-width:1500px;margin:auto;padding:24px;color:#1b3344;background:#eff3f6}
    section{background:#fff;border-radius:10px;padding:20px 24px;margin:20px 0}table{border-collapse:collapse;width:100%}
    th,td{border-bottom:1px solid #d7e2e9;padding:8px;text-align:left;vertical-align:top}th{background:#e9f1f6;white-space:nowrap}
    .scroll{overflow-x:auto}img{max-width:100%;height:auto}.note{background:#fff4df;border-left:5px solid #c07c24;padding:12px}
    details{margin:10px 0;border:1px solid #d7e2e9;border-radius:6px;padding:8px;background:#f8fafc}
    details.task{background:#eaf2f7}details.method{background:#fff}details.score{margin-left:15px}
    summary{cursor:pointer;font-weight:600}details.score img{display:block;margin:12px 0}
    nav a{margin-right:18px}a{color:#145e8c}figure{margin:22px 0}figcaption{color:#506779}'''
    body='''<header><h1>15タスク統合：論文用TDSデータ・結果</h1>
    <p>Scissors 1–5、Push 1–5、Spray 1–5。学習済みポリシー、成功姿勢DB、PCA/PLSによる1/2自由度TDS評価、可視化をこのディレクトリに集約。</p>
    <nav><a href="#overview">全体結果</a><a href="#tds_method">計算・制御方法</a><a href="#scissors">Scissors条件・結果</a><a href="#scissors_radius_comparison">半径比較</a><a href="#task_results">タスク別</a><a href="#training">学習・DB</a>
    <a href="#protocol">実験条件</a><a href="#trajectories">25ステップ画像</a></nav></header>
    <section id="overview"><h2>採用実験と全体結果</h2>
    <p class="note">採用データはScissors＝Palm追従球条件の修正版、Push・Spray＝接触数ゲートを外しSpray床接触を有効化した最新版。
    旧Push・Spray接触ゲート条件のreport.jsonも比較用に保存した。これらは成功条件が異なるため、15タスク一律の把持成功率としては統合しない。</p>
    <table><tr><th>カテゴリ</th><th>Fit成立 [構成]</th><th>逆写像可能な目標範囲 平均 [%]</th><th>独立Sweep Coverage [%]</th><th>連続Sweep Coverage [%]</th></tr>'''+''.join(categories)+'''</table>
    <p>各カテゴリ5タスク×PCA/PLS×1/2自由度＝20構成。下表と図は対応する1自由度のPCA/PLSを比較する。
    採用実験では全構成で固定第2スコアが0となり、2自由度軌道は同じ方式の1自由度軌道と同一である。</p>
    <figure><img src="figures/summary_inverse_domain.png"><figcaption>逆写像で指令できる目標範囲を各タスクのRL目標幅に対する割合 [%] で表示。PCAとPLSの1自由度比較。</figcaption></figure>
    <figure><img src="figures/summary_normalized_mae.png"><figcaption>終端保持に成功した未知目標での等張逆写像MAEを、各タスクのRL目標幅で割った値 [%]。対象範囲は各構成の逆写像域であり、全域MAEではない。</figcaption></figure>
    <div class="scroll"><table><tr><th>タスク</th><th>PCA逆写像可能な目標範囲 [%]</th><th>PLS逆写像可能な目標範囲 [%]</th><th>PCA独立Coverage [%]</th><th>PLS独立Coverage [%]</th><th>PCA正規化held MAE [%]</th><th>PLS正規化held MAE [%]</th></tr>'''+''.join(rows)+'''</table></div>
    <p><a href="manifest.json">出典・SHA-256・収録/除外範囲</a> ／ <a href="source_manifests/training_manifest.json">学習全体設定</a></p></section>
    <section id="task_results"><h2>タスク別：PCA/PLS、Coverage、MAE、応答曲線</h2>
    <p>Scissorsは最新版の応答・逆写像誤差図を掲載。Push・Sprayは旧接触ゲート条件と新条件の校正曲線を重ねた図を掲載。</p>'''+''.join(details)+'''</section>
    <section id="training"><h2>強化学習と成功姿勢DBの要約</h2>
    <p>各DBはBest Success Policyから集めた成功姿勢200件。Validationは32初期姿勢×11目標の決定論的評価。
    DB収集では同じポリシーを固定しつつ、行動は試行ごとに決定論的／確率的を交互に使用した。</p>
    <div class="scroll"><table><tr><th>タスク</th><th>実学習Step</th><th>Best Validation成功率</th><th>DB試行数</th><th>DB姿勢</th><th>達成ツール範囲</th></tr>'''+''.join(training_rows)+'''</table></div></section>'''
    if body.count('<section id="task_results">')!=1:
        raise ValueError('task-results insertion point is ambiguous')
    body=body.replace('<section id="task_results">',method_section()+'<section id="task_results">',1)
    page='<!doctype html><html lang="ja"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page+='<title>15-task integrated TDS paper data</title><style>'+style+'</style><main>'
    page+=body+scissors_section(root,reports)+scissors_radius_comparison(root,reports)
    page+=protocol_section(root,root)+gallery_section(root)+'</main></html>'
    (root/'index.html').write_text(page,encoding='utf-8')


def make_summary_figures(root,reports):
    labels=TASKS
    x=np.arange(len(labels))
    figures=root/'figures'
    figures.mkdir(exist_ok=True)
    for measure,name,ylabel in (
        ('inverse_domain_fraction','summary_inverse_domain','逆写像可能な目標範囲 [% of RL目標幅]'),
        ('normalized_mae','summary_normalized_mae','正規化等張held MAE [% of RL目標幅]')):
        fig,ax=plt.subplots(figsize=(15,5))
        for offset,method,color in ((-.19,'pca_1d','#2475a8'),(.19,'pls_1d','#d78227')):
            values=[]
            for task in labels:
                report=reports[task]
                c=report['configurations'][method]
                if measure=='normalized_mae':
                    v=c['summary']['controller_comparison'].get('isotonic',{}).get('held_mae')
                    v=v/float(np.diff(report['target_range'])[0]) if v is not None else np.nan
                else:
                    v=c['summary'][measure]
                values.append(100*v if v is not None else np.nan)
            ax.bar(x+offset,values,width=.37,label=method.upper(),color=color)
        ax.axvline(4.5,color='#97aab6');ax.axvline(9.5,color='#97aab6')
        ax.set_xticks(x,labels,rotation=55,ha='right')
        ax.set_ylabel(ylabel);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2)
        if measure=='inverse_domain_fraction':
            ax.set_ylim(0,105)
        ax.legend();fig.tight_layout();fig.savefig(figures/f'{name}.png',dpi=150);plt.close(fig)


def make_contact_comparison_figures(root,reports):
    output=root/'comparison_figures'
    output.mkdir(exist_ok=True)
    for category in ('push','spray'):
        unit,scale=('mm',1000.) if category=='push' else ('rad',1.)
        for variant in range(1,6):
            task=f'{category}_lite_{variant}'
            previous=json.loads((root/task/'contact_gated_report.json').read_text(encoding='utf-8'))
            current=reports[task]
            fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
            for ax,label in zip(axes.flat,METHODS):
                for report,name,color,style in ((previous,'旧：接触数ゲート','#66899f','o--'),
                                                (current,'採用：接触数ゲートなし','#db7e31','o-')):
                    x,y=calibration_curve(report['configurations'][label])
                    ax.plot(x,scale*y,style,color=color,label=name,markersize=3,alpha=.85)
                selected=current['configurations'][label]['selection']
                if selected:
                    ax.axvspan(min(selected['rho_start'],selected['rho_end']),
                        max(selected['rho_start'],selected['rho_end']),
                        color='#75b687',alpha=.15,label='採用Calibration区間')
                ax.set(title=label.upper(),xlabel='シナジースコア ρ [無次元]',
                       ylabel=f'実測ツール応答 [{unit}]')
                ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=8)
            fig.suptitle(f'{task}: Calibration応答（同一のPCA/PLS得点と初期姿勢）')
            fig.savefig(output/f'{task}.png',dpi=150)
            plt.close(fig)


def make_scissors_task_figures(root,reports):
    """Render the selected Scissors reports with the current unit-aware labels."""
    import build_paper15_tds_diagnostic_html as charts

    charts.FIG=root/'figures'
    charts.FIG.mkdir(exist_ok=True)
    for variant in range(1,6):
        task=f'scissors_lite_{variant}'
        report=reports[task]
        audits={key:charts.audit(report['configurations'][key],report['target_range'])
                for key in METHODS}
        charts.task_charts(task,report,audits)


def build(args):
    project=Path(__file__).resolve().parents[1]
    training=(project/args.training).resolve()
    scissors=(project/args.scissors).resolve()
    latest=(project/args.latest).resolve()
    output=(project/args.output).resolve()
    if output.exists():
        raise FileExistsError(f'refusing to overwrite existing paper bundle: {output}')
    for p in (training,scissors,latest):
        if not p.is_dir():
            raise FileNotFoundError(p)
    if not output.is_relative_to(project/'analysis'):
        raise ValueError('output must be a new directory under project/analysis')
    shutil.copytree(latest,output)
    sources=output/'source_manifests'
    sources.mkdir()
    copy_required(latest/'manifest.json',sources/'no_contact_manifest.json')
    copy_required(latest/'status.json',sources/'no_contact_status.json')
    copy_required(training/'manifest.json',sources/'training_manifest.json')
    copy_required(scissors/'manifest.json',sources/'scissors_corrected_manifest.json')
    copy_required(scissors/'diagnostic_metrics.json',sources/'contact_gated_diagnostic_metrics.json')
    shutil.copytree(scissors/'evaluation_source_snapshot',output/'scissors_evaluation_source_snapshot')
    shutil.copytree(training/'pose_pool',output/'pose_pool')
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        protocol='paper15_integrated_selected_results_v1',
        source_directories=dict(training=str(training),scissors_tds=str(scissors),
                                push_spray_tds=str(latest)),
        selected_tds=dict(scissors='palm-sphere corrected',
                          push='no contact-count gate, drop/tip rejected',
                          spray='no contact-count gate, floor contact forced on, drop/tip rejected'),
        included=['best_success_policy.zip','final model.zip if different from best',
            'training summary and validation history','success-only 200-posture DB',
            'selected TDS report.json for all 15 tasks','old Push/Spray contact-gated reports',
            'all 15 pose pools','PCA/PLS 17-score contact sheets and summary figures',
            'evaluation source snapshots'],
        excluded=['intermediate 100k-step checkpoints','training replay buffers',
                  'raw training-candidate NPZ files','nonessential process logs'],
        tasks={})
    for task in TASKS:
        train_task=training/task
        target=output/task
        target.mkdir(exist_ok=True)
        source_report=(scissors if task.startswith('scissors') else latest)/task/'report.json'
        if task.startswith('scissors'):
            copy_required(source_report,target/'report.json')
            for suffix in ('response','calibration_validity','inverse_error'):
                name=f'{task}_{suffix}.png'
                copy_required(scissors/'figures'/name,output/'figures'/name)
        else:
            if digest(source_report)!=digest(target/'report.json'):
                raise ValueError(f'latest selected report changed during copy: {task}')
            copy_required(scissors/task/'report.json',target/'contact_gated_report.json')
        files={}
        for name in ('best_success_policy.zip','model_training_summary.json',
                     'validation_history.json','train_command.json'):
            files[name]=copy_required(train_task/name,target/'training'/name)
        for name in ('success_postures.npz','success_postures.progress.json','collect_command.json'):
            files[name]=copy_required(train_task/name,target/'database'/name)
        best_hash=files['best_success_policy.zip']['sha256']
        if digest(train_task/'best_policy.zip')!=best_hash:
            raise ValueError(f'best_policy and best_success_policy differ: {task}')
        final_hash=digest(train_task/'model.zip')
        if final_hash!=best_hash:
            files['model.zip']=copy_required(train_task/'model.zip',target/'training'/'model.zip')
        report=json.loads((target/'report.json').read_text(encoding='utf-8'))
        dbmeta=report['database_metadata']
        if dbmeta['model_sha256']!=best_hash or report['database_sha256']!=files['success_postures.npz']['sha256']:
            raise ValueError(f'model/database provenance mismatch: {task}')
        if len(report['configurations'])!=4 or report['fit_rows']!=200:
            raise ValueError(f'incomplete selected TDS report: {task}')
        manifest['tasks'][task]=dict(category=task.split('_')[0],
            selected_report=f'{task}/report.json',
            selected_report_sha256=digest(target/'report.json'),
            database_sha256=files['success_postures.npz']['sha256'],
            best_success_policy_sha256=best_hash,
            final_model_sha256=final_hash,
            best_policy_same_as_best_success=True,
            pose_pool_sha256=dbmeta['contract']['pool_sha256'],
            training_files={k:dict(path=str(v['path'].relative_to(output)),sha256=v['sha256'],
                                   bytes=v['bytes']) for k,v in files.items()})
    summary_report(output,manifest)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,default=str),encoding='utf-8')
    (output/'status.json').write_text(json.dumps(dict(tasks={task:'complete' for task in TASKS},
         html='index.html'),indent=2),encoding='utf-8')
    print(output/'index.html')


def refresh_report(args):
    project=Path(__file__).resolve().parents[1]
    output=(project/args.output).resolve()
    if not output.is_relative_to(project/'analysis') or not output.is_dir():
        raise ValueError('refresh target must be an existing paper bundle under project/analysis')
    manifest_path=output/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('protocol')!='paper15_integrated_selected_results_v1' or set(manifest['tasks'])!=set(TASKS):
        raise ValueError('unexpected paper bundle manifest')
    for task,meta in manifest['tasks'].items():
        report=output/meta['selected_report']
        if digest(report)!=meta['selected_report_sha256']:
            raise ValueError(f'selected TDS report hash changed: {task}')
    summary_report(output,manifest)
    manifest['integrated_html_revision']='PCA/PLS and 1/2-DoF method details; unit-aware axes and RL-span-normalized metrics'
    manifest['integrated_html_sha256']=digest(output/'index.html')
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    print(output/'index.html')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,default=Path('runs/paper15_unified_3m_slew_20260927'))
    parser.add_argument('--scissors',type=Path,default=Path('analysis/paper15_unified_tds_corrected_20260928'))
    parser.add_argument('--latest',type=Path,default=Path('analysis/paper15_no_contact_floor_on_20260928'))
    parser.add_argument('--output',type=Path,default=Path('analysis/paper15_paper_bundle_20260928'))
    parser.add_argument('--refresh-report',action='store_true',help='update only HTML and derived summary figures in an existing verified bundle')
    args=parser.parse_args()
    refresh_report(args) if args.refresh_report else build(args)


if __name__=='__main__':
    main()
