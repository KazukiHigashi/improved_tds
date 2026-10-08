"""Export the exact simplified Push 3 MuJoCo tool envelope as CAD references.

Only the three cylinder geoms and the slide range are reproduced. The source
model deliberately excludes body/plunger collision and specifies no bore,
guide, spring seat or retainer; these exports are not fabrication-ready parts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_XML = (PROJECT_ROOT.parent/'tool-manip-gym/gym-push/gym_push/assets/'
              'shadow_hand/shadow_hand_push3.xml')
DEFAULT_OUTPUT = PROJECT_ROOT/'cad/push3_reference_20261002'


def _numbers(value: str, expected: int) -> tuple[float, ...]:
    numbers = tuple(float(item) for item in value.split())
    if len(numbers) != expected:
        raise ValueError(f'expected {expected} numbers: {value!r}')
    return numbers


def read_dimensions(source: Path) -> dict:
    root = ET.parse(source).getroot()
    body = root.find("./worldbody/body[@name='push_object']")
    if body is None:
        raise ValueError('Push 3 body not found')
    plunger = body.find("./body[@name='push_plunger']")
    if plunger is None:
        raise ValueError('Push 3 plunger not found')
    def cylinder(parent: ET.Element, name: str) -> dict:
        geom = parent.find(f"./geom[@name='{name}']")
        if geom is None or geom.get('type') != 'cylinder':
            raise ValueError(f'missing cylinder geom: {name}')
        radius, half_height = _numbers(geom.attrib['size'], 2)
        center = _numbers(geom.get('pos', '0 0 0'), 3)
        if radius <= 0 or half_height <= 0 or center[:2] != (0., 0.):
            raise ValueError(f'invalid coaxial cylinder: {name}')
        return dict(radius_mm=1000*radius, diameter_mm=2000*radius,
                    half_height_mm=1000*half_height,
                    height_mm=2000*half_height,
                    center_z_mm=1000*center[2])
    main = cylinder(body, 'push_body_geom')
    neck = cylinder(plunger, 'push_neck_geom')
    cap = cylinder(plunger, 'push_plunger_geom')
    slide = plunger.find("./joint[@name='push_plunger_joint']")
    if slide is None or slide.get('type') != 'slide':
        raise ValueError('plunger slide joint not found')
    slide_min, slide_max = _numbers(slide.attrib['range'], 2)
    axis = _numbers(slide.attrib['axis'], 3)
    if axis != (0., 0., -1.) or slide_min != 0. or slide_max <= 0.:
        raise ValueError('unexpected plunger travel direction or range')
    body_position = _numbers(body.attrib['pos'], 3)
    plunger_position = _numbers(plunger.attrib['pos'], 3)
    body_bottom = main['center_z_mm'] - main['half_height_mm']
    body_top = main['center_z_mm'] + main['half_height_mm']
    neck_bottom = 1000*plunger_position[2] + neck['center_z_mm'] - neck['half_height_mm']
    neck_top = 1000*plunger_position[2] + neck['center_z_mm'] + neck['half_height_mm']
    cap_bottom = 1000*plunger_position[2] + cap['center_z_mm'] - cap['half_height_mm']
    cap_top = 1000*plunger_position[2] + cap['center_z_mm'] + cap['half_height_mm']
    if (abs(body_top - neck_bottom) > 1e-9 or
            abs(neck_top - cap_bottom) > 1e-9):
        raise ValueError('Push 3 cylinders do not form the expected stacked envelope')
    body_inertia = body.find('./inertial')
    plunger_inertia = plunger.find('./inertial')
    if body_inertia is None or plunger_inertia is None:
        raise ValueError('source inertial values missing')
    return dict(protocol='push3_cad_reference_v1',
        source_xml=str(source.resolve()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_environment='ShadowHandLitePush3-v0',
        cad_unit='mm', mujoco_unit='m',
        body=main, neck=neck, cap=cap,
        body_initial_center_world_mm=[1000*value for value in body_position],
        plunger_origin_relative_to_body_center_mm=[1000*value for value in plunger_position],
        body_bottom_from_body_center_mm=body_bottom,
        neck_bottom_from_body_base_mm=neck_bottom-body_bottom,
        neck_top_from_body_base_mm=neck_top-body_bottom,
        cap_bottom_from_body_base_mm=cap_bottom-body_bottom,
        cap_top_from_body_base_mm=cap_top-body_bottom,
        body_height_mm=main['height_mm'],
        plunger_height_mm=cap_top-body_top,
        rest_total_height_mm=cap_top-body_bottom,
        full_press_total_height_mm=cap_top-body_bottom-1000*slide_max,
        stroke_mm=1000*slide_max,
        stroke_axis=axis,
        simulation_body_mass_g=1000*float(body_inertia.attrib['mass']),
        simulation_plunger_mass_g=1000*float(plunger_inertia.attrib['mass']),
        simulation_slide_stiffness_n_per_m=float(slide.attrib['stiffness']),
        simulation_slide_damping_n_s_per_m=float(slide.attrib['damping']),
        simulation_slide_frictionloss_n=float(slide.attrib['frictionloss']),
        simulation_slide_springref_mm=1000*float(slide.attrib['springref']))


def scad_text(dim: dict) -> str:
    body, neck, cap = dim['body'], dim['neck'], dim['cap']
    return f'''// Push 3: exact simplified MuJoCo exterior reference, units: mm.
// Source: {Path(dim['source_xml']).name}
// SHA-256: {dim['source_sha256']}
// Not a working push-button mechanism: no bore, guide, stop, or spring seat.
// In the simulator the body/plunger collision pair is excluded; positive
// press_mm intentionally makes the solid reference shapes interpenetrate.

$fn = 128;
view = "assembly"; // "assembly", "body", "plunger", "exploded"
press_mm = 0;    // 0 .. {dim['stroke_mm']:g}; simulator travel toward -Z
exploded_gap_mm = 12;

body_d_mm = {body['diameter_mm']:g};
body_h_mm = {body['height_mm']:g};
neck_d_mm = {neck['diameter_mm']:g};
neck_h_mm = {neck['height_mm']:g};
button_d_mm = {cap['diameter_mm']:g};
button_h_mm = {cap['height_mm']:g};
stroke_mm = {dim['stroke_mm']:g};

module push3_body() {{
    cylinder(d=body_d_mm, h=body_h_mm);
}}

module push3_plunger() {{
    union() {{
        cylinder(d=neck_d_mm, h=neck_h_mm);
        translate([0, 0, neck_h_mm])
            cylinder(d=button_d_mm, h=button_h_mm);
    }}
}}

module push3_assembly(press=0, gap=0) {{
    color([0.90, 0.93, 0.97]) push3_body();
    color([0.20, 0.36, 0.62])
        translate([0, 0, body_h_mm - press + gap]) push3_plunger();
}}

assert(press_mm >= 0 && press_mm <= stroke_mm, "press_mm outside MuJoCo range");
if (view == "body") push3_body();
else if (view == "plunger") push3_plunger();
else if (view == "exploded") push3_assembly(0, exploded_gap_mm);
else push3_assembly(press_mm);
'''


def _triangle(a, b, c):
    ab = tuple(b[i]-a[i] for i in range(3))
    ac = tuple(c[i]-a[i] for i in range(3))
    normal = (ab[1]*ac[2]-ab[2]*ac[1],
              ab[2]*ac[0]-ab[0]*ac[2],
              ab[0]*ac[1]-ab[1]*ac[0])
    length = sum(value*value for value in normal)**.5
    if length <= 1e-12:
        raise ValueError('degenerate CAD triangle')
    return (tuple(value/length for value in normal), a, b, c)


def cylinder_triangles(radius: float, low: float, high: float,
                       segments: int = 128) -> list:
    from math import cos, sin, tau
    triangles = []
    for i in range(segments):
        a, b = tau*i/segments, tau*((i+1) % segments)/segments
        x0, y0, x1, y1 = radius*cos(a), radius*sin(a), radius*cos(b), radius*sin(b)
        bottom_a, bottom_b = (x0,y0,low), (x1,y1,low)
        top_a, top_b = (x0,y0,high), (x1,y1,high)
        triangles.extend((
            _triangle(bottom_a, bottom_b, top_b),
            _triangle(bottom_a, top_b, top_a),
            _triangle((0.,0.,low), bottom_b, bottom_a),
            _triangle((0.,0.,high), top_a, top_b)))
    return triangles


def plunger_triangles(neck_radius: float, neck_height: float,
                      cap_radius: float, cap_height: float,
                      segments: int = 128) -> list:
    from math import cos, sin, tau
    triangles = []
    top = neck_height + cap_height
    for i in range(segments):
        a, b = tau*i/segments, tau*((i+1) % segments)/segments
        x0,y0,x1,y1 = cos(a),sin(a),cos(b),sin(b)
        n0,n1 = (neck_radius*x0,neck_radius*y0),(neck_radius*x1,neck_radius*y1)
        c0,c1 = (cap_radius*x0,cap_radius*y0),(cap_radius*x1,cap_radius*y1)
        triangles.extend((
            _triangle((n0[0],n0[1],0.), (n1[0],n1[1],0.),
                      (n1[0],n1[1],neck_height)),
            _triangle((n0[0],n0[1],0.), (n1[0],n1[1],neck_height),
                      (n0[0],n0[1],neck_height)),
            _triangle((n0[0],n0[1],neck_height),
                      (n1[0],n1[1],neck_height), (c1[0],c1[1],neck_height)),
            _triangle((n0[0],n0[1],neck_height),
                      (c1[0],c1[1],neck_height), (c0[0],c0[1],neck_height)),
            _triangle((c0[0],c0[1],neck_height),
                      (c1[0],c1[1],neck_height), (c1[0],c1[1],top)),
            _triangle((c0[0],c0[1],neck_height),
                      (c1[0],c1[1],top), (c0[0],c0[1],top)),
            _triangle((0.,0.,0.), (n1[0],n1[1],0.), (n0[0],n0[1],0.)),
            _triangle((0.,0.,top), (c0[0],c0[1],top), (c1[0],c1[1],top))))
    return triangles


def write_stl(path: Path, triangles: list, label: str) -> None:
    header = label.encode('ascii')[:80].ljust(80, b' ')
    with path.open('wb') as stream:
        stream.write(header)
        stream.write(struct.pack('<I', len(triangles)))
        for normal, a, b, c in triangles:
            stream.write(struct.pack('<12fH', *(normal+a+b+c), 0))


def write_3mf(path: Path, parts: list[tuple[str, list, float]]) -> None:
    """Write a minimal 3MF package with an explicit millimeter model unit."""
    core = 'http://schemas.microsoft.com/3dmanufacturing/core/2015/02'
    package = 'http://schemas.openxmlformats.org/package/2006/content-types'
    relationships = 'http://schemas.openxmlformats.org/package/2006/relationships'
    ET.register_namespace('', core)
    model = ET.Element(f'{{{core}}}model', {'unit': 'millimeter',
                                         '{http://www.w3.org/XML/1998/namespace}lang': 'en-US'})
    resources = ET.SubElement(model, f'{{{core}}}resources')
    build = ET.SubElement(model, f'{{{core}}}build')
    for object_id, (name, facets, z_offset) in enumerate(parts, start=1):
        obj = ET.SubElement(resources, f'{{{core}}}object',
                            {'id': str(object_id), 'name': name, 'type': 'model'})
        mesh = ET.SubElement(obj, f'{{{core}}}mesh')
        vertices = ET.SubElement(mesh, f'{{{core}}}vertices')
        triangles = ET.SubElement(mesh, f'{{{core}}}triangles')
        index: dict[tuple[float, float, float], int] = {}
        for _, a, b, c in facets:
            ids = []
            for point in (a, b, c):
                if point not in index:
                    index[point] = len(index)
                    ET.SubElement(vertices, f'{{{core}}}vertex',
                                  dict(zip(('x', 'y', 'z'), (f'{v:.9g}' for v in point))))
                ids.append(index[point])
            ET.SubElement(triangles, f'{{{core}}}triangle',
                          {'v1': str(ids[0]), 'v2': str(ids[1]), 'v3': str(ids[2])})
        attrs = {'objectid': str(object_id)}
        if z_offset:
            attrs['transform'] = f'1 0 0 0 1 0 0 0 1 0 0 {z_offset:g}'
        ET.SubElement(build, f'{{{core}}}item', attrs)
    model_xml = ET.tostring(model, encoding='utf-8', xml_declaration=True)

    ET.register_namespace('', package)
    types = ET.Element(f'{{{package}}}Types')
    ET.SubElement(types, f'{{{package}}}Default',
                  {'Extension': 'rels', 'ContentType':
                   'application/vnd.openxmlformats-package.relationships+xml'})
    ET.SubElement(types, f'{{{package}}}Default',
                  {'Extension': 'model', 'ContentType':
                   'application/vnd.ms-package.3dmanufacturing-3dmodel+xml'})
    types_xml = ET.tostring(types, encoding='utf-8', xml_declaration=True)

    ET.register_namespace('', relationships)
    rels = ET.Element(f'{{{relationships}}}Relationships')
    ET.SubElement(rels, f'{{{relationships}}}Relationship',
                  {'Target': '/3D/3dmodel.model', 'Id': 'rel0',
                   'Type': 'http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel'})
    rels_xml = ET.tostring(rels, encoding='utf-8', xml_declaration=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('[Content_Types].xml', types_xml)
        archive.writestr('_rels/.rels', rels_xml)
        archive.writestr('3D/3dmodel.model', model_xml)


def dimensions_markdown(dim: dict) -> str:
    b, n, c = dim['body'], dim['neck'], dim['cap']
    return f'''# Push 3 寸法・CAD参照

出典：`ShadowHandLitePush3-v0` が読み込む [`{Path(dim['source_xml']).name}`]({dim['source_xml'].replace(chr(92), '/')})。XML SHA-256：`{dim['source_sha256']}`。MuJoCoのmをmmに換算した値です。以下の形状は**シミュレーション上の簡略形状**で、実用品の製作図ではありません。

| 部位 | 外径 | 軸方向長さ | 基準位置（本体底面 z=0） |
|---|---:|---:|---:|
| 円筒本体 | {b['diameter_mm']:g} mm | {b['height_mm']:g} mm | z=0〜{b['height_mm']:g} mm |
| 可動軸（neck） | {n['diameter_mm']:g} mm | {n['height_mm']:g} mm | 休止時 z={b['height_mm']:g}〜{b['height_mm']+n['height_mm']:g} mm |
| 押しボタン（cap） | {c['diameter_mm']:g} mm | {c['height_mm']:g} mm | 休止時 z={b['height_mm']+n['height_mm']:g}〜{dim['rest_total_height_mm']:g} mm |

- 押し込み方向：本体の −Z。可動範囲：0〜{dim['stroke_mm']:g} mm。
- 全高：休止時 {dim['rest_total_height_mm']:g} mm、最大押し込み時 {dim['full_press_total_height_mm']:g} mm。
- MuJoCo初期配置の本体中心：({', '.join(f'{v:g}' for v in dim['body_initial_center_world_mm'])}) mm。これはシーン内の配置であり、部品寸法ではありません。
- シミュレーション上の本体質量 {dim['simulation_body_mass_g']:g} g、可動部質量 {dim['simulation_plunger_mass_g']:g} g、スライドばね定数 {dim['simulation_slide_stiffness_n_per_m']:g} N/m。これらは製作材料・ばね仕様の指定ではありません。

## ファイル

- `push3_assembly_rest_mm.3mf`：Fusion 360向けの静止時組立形状。3MF内部で単位を **millimeter** と明示し、本体底面 z=0、可動部底面 z=120 mmに配置しています。
- `push3_body_mm.3mf`、`push3_plunger_mm.3mf`：部品別の3MF。いずれも単位はmm、各部品の底面 z=0です。
- `push3_reference.scad`：寸法を編集できるOpenSCAD原稿。`view` で本体・可動部・組立・分解表示を切替可能です。
- `push3_body.stl`：本体外形。底面 z=0、単位はmmとしてインポートしてください。
- `push3_plunger.stl`：可動軸とボタンが一体の参照形状。局所底面 z=0。
- `push3_assembly_rest.stl`：休止位置に両部品を配置した参照メッシュ。単位はmm。
- `push3_dimensions.svg`：寸法の概略図。
- `dimensions.json`：機械処理向けの値と出典情報。

## Fusion 360への読み込みとスケール確認

推奨：新規デザインの「デザイン」→「メッシュ」→「作成」→「メッシュを挿入」で `push3_assembly_rest_mm.3mf` を選択してください。3MFにはmm単位が記録されています。読み込み後、本体直径 **52 mm**、底面からボタン頂面まで **148 mm** であることを「計測」で確認してください。ファイルを単に開く／アップロードする方法より、既存デザインへの「メッシュを挿入」が配置確認に適しています。

STLを使う場合は同じ「メッシュを挿入」で単位タイプを **ミリメートル** に明示してください。STL自体には単位情報がなく、Fusion 360の「開く」やアップロード経由では10倍の大きさとして扱われることがあります。既に10倍になったメッシュだけは倍率0.1で修正できますが、最初から3MFか「メッシュを挿入」を使うことを推奨します。メッシュ形式は編集可能なソリッドCADではありません。CADソリッドを作る場合は上表の円柱寸法を用いてFusion 360上で再モデリングしてください。

## 製作前に必要な追加設計

MuJoCoでは本体と可動部の衝突が除外され、本体は**中実円柱**として定義されています。押し込むと両者は幾何学的に重なります。従って、CADの外形をそのまま印刷してもスライド機構にはなりません。実機用にはガイド穴・軸の差し込み長・摺動クリアランス・止め輪またはストッパ・ばね座・ばね選定・材料強度を別途設計し、試作と安全確認が必要です。これらの寸法は元のシミュレーションには含まれないため、本資料では恣意的に確定していません。
'''


def dimension_svg(dim: dict) -> str:
    b, n, c = dim['body'], dim['neck'], dim['cap']
    scale = 2.5
    x = 230
    bottom = 475
    body_top = bottom-scale*b['height_mm']
    neck_top = body_top-scale*n['height_mm']
    cap_top = neck_top-scale*c['height_mm']
    body_left, body_width = x-scale*b['radius_mm'], scale*b['diameter_mm']
    neck_left, neck_width = x-scale*n['radius_mm'], scale*n['diameter_mm']
    cap_left, cap_width = x-scale*c['radius_mm'], scale*c['diameter_mm']
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 550" width="760" height="550">
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="4" refY="4" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 Z" fill="#334155"/></marker></defs>
<rect width="760" height="550" fill="#f8fafc"/>
<style>text{{font:16px sans-serif;fill:#172033}} .small{{font-size:12px;fill:#475569}} .dim{{stroke:#334155;stroke-width:1.5;marker-start:url(#arrow);marker-end:url(#arrow)}} .guide{{stroke:#94a3b8;stroke-width:1}}</style>
<text x="35" y="35" font-size="23">Push 3 — MuJoCo外形寸法（mm）</text>
<rect x="{body_left:g}" y="{body_top:g}" width="{body_width:g}" height="{scale*b['height_mm']:g}" fill="#dbeafe" stroke="#334155" stroke-width="2"/>
<rect x="{neck_left:g}" y="{neck_top:g}" width="{neck_width:g}" height="{scale*n['height_mm']:g}" fill="#94a3b8" stroke="#334155" stroke-width="2"/>
<rect x="{cap_left:g}" y="{cap_top:g}" width="{cap_width:g}" height="{scale*c['height_mm']:g}" fill="#4169a7" stroke="#334155" stroke-width="2"/>
<line x1="105" y1="{body_top:g}" x2="{body_left-8:g}" y2="{body_top:g}" class="guide"/><line x1="105" y1="{bottom:g}" x2="{body_left-8:g}" y2="{bottom:g}" class="guide"/>
<line x1="105" y1="{body_top:g}" x2="105" y2="{bottom:g}" class="dim"/><text x="40" y="{(body_top+bottom)/2:g}">本体 {b['height_mm']:g}</text>
<line x1="{body_left:g}" y1="500" x2="{body_left+body_width:g}" y2="500" class="dim"/><text x="{x-38:g}" y="527">Ø{b['diameter_mm']:g}</text>
<line x1="{cap_left:g}" y1="{cap_top-14:g}" x2="{cap_left+cap_width:g}" y2="{cap_top-14:g}" class="dim"/><text x="{x-18:g}" y="{cap_top-20:g}" class="small">Ø{c['diameter_mm']:g}</text>
<line x1="{neck_left:g}" y1="{neck_top+scale*n['height_mm']/2:g}" x2="{neck_left+neck_width:g}" y2="{neck_top+scale*n['height_mm']/2:g}" class="dim"/><text x="{x+45:g}" y="{neck_top+scale*n['height_mm']/2+5:g}" class="small">Ø{n['diameter_mm']:g}</text>
<text x="315" y="{(cap_top+neck_top)/2+5:g}">ボタン 厚さ {c['height_mm']:g}</text>
<text x="315" y="{(neck_top+body_top)/2+5:g}">可動軸 高さ {n['height_mm']:g}</text>
<text x="385" y="215">押し込み量 {dim['stroke_mm']:g} ↓</text>
<text x="385" y="245">休止時全高 {dim['rest_total_height_mm']:g}</text>
<text x="385" y="275">最大押し込み時全高 {dim['full_press_total_height_mm']:g}</text>
<text x="385" y="325" class="small">元XMLは本体・可動部の衝突を除外。</text>
<text x="385" y="347" class="small">内部ガイド穴やばね座は未定義。</text>
<text x="35" y="537" class="small">出典: {Path(dim['source_xml']).name} / 外形参照のみ・製作図ではありません</text>
</svg>'''


def export(source: Path, output: Path) -> dict:
    dim = read_dimensions(source)
    output.mkdir(parents=True, exist_ok=True)
    (output/'dimensions.json').write_text(json.dumps(dim, ensure_ascii=False, indent=2),
                                          encoding='utf-8')
    (output/'push3_reference.scad').write_text(scad_text(dim), encoding='utf-8')
    (output/'README.md').write_text(dimensions_markdown(dim), encoding='utf-8')
    (output/'push3_dimensions.svg').write_text(dimension_svg(dim), encoding='utf-8')
    body_mesh = cylinder_triangles(dim['body']['radius_mm'], 0., dim['body_height_mm'])
    plunger_mesh = plunger_triangles(dim['neck']['radius_mm'],
                                    dim['neck']['height_mm'],
                                    dim['cap']['radius_mm'],
                                    dim['cap']['height_mm'])
    def shifted(triangle, amount):
        normal, *points = triangle
        return normal, *(tuple((x, y, z+amount)) for x, y, z in points)
    write_stl(output/'push3_body.stl', body_mesh, 'Push3 body; units mm')
    write_stl(output/'push3_plunger.stl', plunger_mesh, 'Push3 plunger; units mm')
    write_stl(output/'push3_assembly_rest.stl',
              body_mesh+[shifted(t, dim['body_height_mm']) for t in plunger_mesh],
              'Push3 rest assembly; units mm')
    write_3mf(output/'push3_body_mm.3mf', [('Push3 body', body_mesh, 0.)])
    write_3mf(output/'push3_plunger_mm.3mf', [('Push3 plunger', plunger_mesh, 0.)])
    write_3mf(output/'push3_assembly_rest_mm.3mf',
              [('Push3 body', body_mesh, 0.),
               ('Push3 plunger', plunger_mesh, dim['body_height_mm'])])
    return dim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE_XML)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    dimensions = export(args.source, args.output)
    print(args.output)
    print(f"body diameter {dimensions['body']['diameter_mm']:g} x {dimensions['body_height_mm']:g} mm; "
          f"stroke {dimensions['stroke_mm']:g} mm")


if __name__ == '__main__':
    main()
