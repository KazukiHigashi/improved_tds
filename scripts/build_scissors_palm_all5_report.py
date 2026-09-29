"""Readable five-task HTML summary of the direct PCA/PLS experiment."""

import argparse
import html
import json
from pathlib import Path
from statistics import mean

import numpy as np


STYLE="""body{font:16px/1.55 system-ui,sans-serif;background:#f2f5f8;color:#183047;margin:0}
main{max-width:1260px;margin:auto;padding:24px}section{background:white;border-radius:9px;padding:20px;margin:18px 0}
table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #dbe2e9;text-align:left}
th{background:#edf2f7}.scroll{overflow-x:auto}img{max-width:100%;height:auto}figure{margin:14px 0}
figcaption,.note{color:#506579;font-size:.9em}a{color:#1768aa}details{margin:12px 0}
summary{cursor:pointer;font-weight:600}.warning{border-left:4px solid #bd7521;padding:10px;background:#fff7e9}"""


def page(title,body):
    return (f'<!doctype html><html lang="ja"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>'
            f'<style>{STYLE}</style></head><body><main>{body}</main></body></html>')


def avg(items,digits=3,percent=False):
    values=[float(x) for x in items if x is not None]
    if not values:return '—'
    value=mean(values)*(100 if percent else 1)
    return f'{value:.{digits}f}{"%" if percent else ""}'


def collect(root,task):
    files=sorted((root/'results'/task).glob('*/report.json'))
    return [(p,json.loads(p.read_text(encoding='utf-8'))) for p in files]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root',type=Path,required=True)
    p.add_argument('--run-root',type=Path,required=True)
    a=p.parse_args()
    root=a.analysis_root.resolve()
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    task_links=[]
    learning=[]
    overall=[]
    for task in manifest['tasks']:
        history=json.loads((a.run_root/task/'validation_history.json').read_text(encoding='utf-8'))
        best=history['best']
        with np.load(a.run_root/task/'success_postures.npz',allow_pickle=False) as z:
            db=json.loads(z['metadata_json'].item())
        learning.append(f'<tr><td>{task}</td><td>{best["timesteps"]:,}</td>'
            f'<td>{100*best["success_rate"]:.1f}%</td>'
            f'<td>{100*best["final_success_rate"]:.1f}%</td>'
            f'<td>{best["mean_abs_error"]:.3f}</td><td>{db["rows"]}</td></tr>')
        docs=collect(root,task)
        if not docs:
            task_links.append(f'<li>{task}: no completed TDS split</li>')
            continue
        first=docs[0][1]
        rows=[]
        targets=[]
        for label in ('pca_1d','pls_1d','pca_2d','pls_2d'):
            configs=[doc['configurations'][label] for _,doc in docs]
            inv=[c.get('inverse_target_range') for c in configs]
            iso=[c['summary']['controller_comparison'].get('isotonic',{}) for c in configs]
            ramp=[c['summary']['controller_comparison'].get('ramp',{}) for c in configs]
            rows.append(f'<tr><td>{label.upper()}</td>'
                f'<td>{avg((c["summary"]["independent_coverage"] for c in configs),1,True)}</td>'
                f'<td>{avg((c["summary"]["continuous_coverage"] for c in configs),1,True)}</td>'
                f'<td>{avg((c["summary"]["held_fraction"] for c in configs),1,True)}</td>'
                f'<td>{sum(x is not None for x in inv)}/{len(inv)}</td>'
                f'<td>{avg((x[0] for x in inv if x is not None))}–'
                f'{avg((x[1] for x in inv if x is not None))}</td>'
                f'<td>{avg((c["summary"].get("inverse_domain_fraction") for c in configs),1,True)}</td>'
                f'<td>{avg((x.get("mae") for x in iso))}</td>'
                f'<td>{avg((x.get("mae") for x in ramp))}</td></tr>')
            overall.append((task,label,configs))
            for path,doc in docs:
                c=doc['configurations'][label]
                domain=c.get('inverse_target_range')
                targets.append(f'<tr><td>{path.parent.name}</td><td>{label.upper()}</td>'
                    f'<td>{"—" if domain is None else f"{domain[0]:.3f}–{domain[1]:.3f}"}</td>'
                    f'<td>{avg([c["summary"]["controller_comparison"].get("isotonic",{}).get("mae")])}</td>'
                    f'<td>{avg([c["summary"]["controller_comparison"].get("ramp",{}).get("mae")])}</td>'
                    f'<td><a href="{path.parent.name}/report.json">JSON</a></td></tr>')
        folder=root/'results'/task
        gallery=[]
        representative=docs[0][1]
        seed=docs[0][0].parent.name
        for label in ('pca_1d','pls_1d','pca_2d','pls_2d'):
            for kind,names in representative['configurations'][label]['images'].items():
                if names:
                    gallery.append(f'<details><summary>{label.upper()} / {html.escape(kind)} '
                        f'({len(names)} sheets)</summary>'+''.join(
                        f'<figure><img loading="lazy" src="{seed}/{html.escape(name)}">'
                        f'<figcaption>{label.upper()} / {html.escape(kind)} / {html.escape(name)}</figcaption></figure>'
                        for name in names)+'</details>')
        body=f'<p><a href="../../index.html">全タスク一覧へ戻る</a></p><h1>{task}: PCA / PLS直接制御</h1>'
        body+='<p>新しいPalm追従球の成功条件で学習し、各角度ビン20件の成功姿勢、計200件を使用。'
        body+='各初期姿勢から再構成姿勢まで5秒で移動し、2秒保持するフィードフォワード制御。'
        body+='CalibrationとTestは異なる初期姿勢を用いる。</p>'
        body+='<section><h2>5分割の比較</h2><div class="scroll"><table><tr><th>方式</th>'
        body+='<th>独立試行Coverage</th><th>連続スイープCoverage</th><th>保持成立点</th>'
        body+='<th>逆写像成立分割</th><th>逆写像角度域 [rad]</th><th>RL全域比</th>'
        body+='<th>等張回帰MAE [rad]</th><th>ランプMAE [rad]</th></tr>'+''.join(rows)+'</table></div>'
        body+='<p class="note">独立試行Coverageは各スコアを初期姿勢から別々に試した到達幅。'
        body+='連続スイープCoverageは同一試行で往復した際の幅。'
        body+='逆写像角度域はCalibrationだけから決めた等張回帰・ランプ関数共通の非飽和範囲。'
        body+='範囲外は逆写像評価から除外し、RL全域を達成したとは主張しない。</p></section>'
        if task=='scissors_lite_4':
            body+='<section class="warning"><h2>Scissors 4の解釈</h2>'
            body+='<p>低角度側で有効な単調・保持区間が得られていないため、Calibrationで確認できた角度域のみを逆写像対象とする。'
            body+='範囲を狭めても物理的な到達幅や操作誤差そのものは改善しない。</p></section>'
        body+='<section><h2>PCA / PLSと1 / 2自由度の応答</h2>'
        for dof in (1,2):
            body+=f'<figure><img src="direct_comparison_{dof}d.png"><figcaption>{dof}自由度：成功姿勢DB、'
            body+='Calibration、未使用初期姿勢でのTest、等張回帰とランプ関数。</figcaption></figure>'
        body+='<figure><img src="direct_targets.png"><figcaption>目標角度と実測角度。薄い点は保持失敗。</figcaption></figure></section>'
        body+='<section><h2>分割別の逆写像評価</h2><div class="scroll"><table><tr>'
        body+='<th>分割</th><th>方式</th><th>逆写像角度域 [rad]</th><th>等張回帰MAE</th>'
        body+='<th>ランプMAE</th><th>全データ</th></tr>'+''.join(targets)+'</table></div></section>'
        body+='<section><h2>スイープ姿勢・20ステップごとの画像</h2>'+''.join(gallery)+'</section>'
        (folder/'index.html').write_text(page(f'{task} PCA / PLS',body),encoding='utf-8')
        task_links.append(f'<li><a href="results/{task}/index.html">{task}</a> ({len(docs)}/5分割)</li>')
    body='<h1>Scissors 1–5: Palm追従球条件での学習と直接TDS評価</h1>'
    body+='<p>新たに5タスクを学習し、各Best Policyから成功姿勢200件を収集。'
    body+='PCA/PLS、1/2自由度、等張回帰/ランプ関数を5分割で比較する。</p>'
    body+='<section><h2>学習・データベース</h2><div class="scroll"><table><tr><th>タスク</th>'
    body+='<th>Best step</th><th>Validation成功率</th><th>最終時点成功率</th>'
    body+='<th>角度MAE [rad]</th><th>成功姿勢</th></tr>'+''.join(learning)+'</table></div></section>'
    body+='<section><h2>タスク別TDS結果</h2><ul>'+''.join(task_links)+'</ul></section>'
    body+='<p><a href="manifest.json">実験条件</a> / <a href="status.json">評価状態</a> / '
    body+='<a href="summary.json">数値サマリー</a></p>'
    (root/'index.html').write_text(page('Scissors 1–5 TDS',body),encoding='utf-8')


if __name__=='__main__':
    main()
