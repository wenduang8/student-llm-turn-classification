"""Lay two stored records side by side, axis by axis, for discussion.

The two records need not be of the same kind: in the 20-target calibration the second coder's file
is an independent first round, while the researcher's file is prepared from the development review
and confirmed by him. Say which is which with --describe-a / --describe-b; the text is written into
the workbook so that the comparison is not read as two independent first rounds.

Accepts only files stored by receive_return.py whose receipt says the checks passed in
complete-return mode and whose current SHA-256 still matches the receipt, so the comparison is
always of the preserved pre-discussion records. One exception, for record B only (D-150): with
--reference-b <registry README> it may be a fixed reference record that has no receipt, such as the
researcher's labels in data/labels/. It is then accepted only when its SHA-256 equals the full value
the registry lists for its file name, every record passes the record validator, it carries one
coder id and its keys are exactly the allocation targets. A reference is never stored as a return.

Optional --groups <json> arranges the discussion sheet by discussion group: a "groups" list (id,
axis, title, experience, refs) and an "assign" list giving each discussion row (conversation_id,
segment_order, axis; axis "（整条）" for a whole-interaction flag) its group and whether it is a
representative case. Every discussion row must be assigned exactly once and nothing else may be.
Grouping only orders the rows and adds a 讨论组 sheet; it changes no comparison. It reads the records and writes one new workbook:
a per-axis comparison, a per-target sheet of flags and notes, a count of rows needing discussion
per axis, and a blank discussion record. It computes no agreement statistic; pre-discussion
agreement is computed separately by coder_agreement.py (D-148, D-150). It never changes either record; discussion outcomes are recorded in
the discussion sheet, and any revised label goes into a separate stage='adjudicated' record.

    python coding/compare_coders.py --allocation <dir> <A-stored.jsonl> <B-stored.jsonl> --out <new.xlsx>
        --describe-a "..." --describe-b "..."  [--reference-b data/labels/README.md] [--groups <groups.json>]
"""
from __future__ import annotations
import argparse, hashlib, json, re, sys
from pathlib import Path
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill
from openpyxl.worksheet.datavalidation import DataValidation

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'coding'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from direct_coding import DIRECT_CODES, DIRECT_FIELDS, validate

DISAGREEMENT_TYPES = ['个案判断', '手册边界', '材料缺失', '其他']
def sha256(p: Path) -> str: return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_checked(path: Path, manifest_sha: str) -> tuple[list, dict]:
    path = Path(path)
    rp = path.with_name(path.name + '.receipt.json')
    if not rp.is_file():
        raise ValueError(f'{path.name}: no receipt; store it with receive_return.py first')
    r = json.loads(rp.read_text(encoding='utf-8'))
    if r.get('status') != 'passes' or r.get('mode') != 'complete return':
        raise ValueError(f'{path.name}: receipt does not show a passing complete return')
    if sha256(path) != r.get('sha256'):
        raise ValueError(f'{path.name}: file has changed since it was received')
    if r.get('allocation_manifest_sha256') != manifest_sha:
        raise ValueError(f'{path.name}: received against a different allocation manifest')
    rows = [json.loads(l) for l in path.read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    return rows, r

def registry_sha(path: Path, registry: Path) -> str:
    """The full SHA-256 that a label registry README lists for this file name in its table."""
    name = Path(path).name
    found = []
    for line in Path(registry).read_text(encoding='utf-8').splitlines():
        cells = line.split('|')
        if line.lstrip().startswith('|') and len(cells) > 2 and cells[1].strip() == name:
            found.append(re.findall(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])', line))
    if len(found) != 1 or len(found[0]) != 1:
        raise ValueError(f'{name}: {Path(registry).name} must list this file once with one full 64-character SHA-256')
    return found[0][0]

def load_reference(path: Path, registry: Path, targets: list) -> tuple[list, dict]:
    """A fixed reference record without a receipt (D-150), checked against its registry entry."""
    path = Path(path)
    got = sha256(path)
    if got != registry_sha(path, registry):
        raise ValueError(f'{path.name}: SHA-256 differs from the value listed in {Path(registry).name}')
    rows = []
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f'{path.name}: line {n} is not valid JSON')
        errs = validate(row)
        if errs:
            raise ValueError(f'{path.name}: line {n} fails the record validator: ' + '; '.join(errs))
        rows.append(row)
    coders = {r['coder_id'] for r in rows}
    if len(coders) != 1:
        raise ValueError(f'{path.name}: coder ids {sorted(coders)}; expected exactly one')
    keys = [(r['conversation_id'], r['segment_order']) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != set(targets):
        raise ValueError(f'{path.name}: record keys do not match the allocation targets')
    return rows, {'coder_ids': sorted(coders), 'sha256': got, 'reference_registry': Path(registry).as_posix()}

def load_groups(path: Path) -> tuple[list, dict]:
    g = json.loads(Path(path).read_text(encoding='utf-8'))
    groups = g['groups']; ids = [x['id'] for x in groups]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate group id')
    assign = {}
    for a in g['assign']:
        k = (a['conversation_id'], int(a['segment_order']), a['axis'])
        if k in assign:
            raise ValueError(f'discussion row assigned twice: {k}')
        if a['group'] not in ids:
            raise ValueError(f"unknown group {a['group']!r}")
        assign[k] = (a['group'], bool(a.get('representative')))
    return groups, assign

def status(axis, a, b):
    """Describe how two values for one axis relate. None means the coder left it empty."""
    if a is None and b is None: return '双方都留空'
    if a is None or b is None: return '仅一方留空'
    if DIRECT_FIELDS[axis]['selection'] == 'single':
        return '相同' if a == b else '不同'
    A, B = set(a), set(b)
    if A == B: return '相同'
    return '部分重叠' if A & B else '无重叠'

def empty_reason(r, axis):
    if r['labels'][axis] is not None: return ''
    if axis in r.get('unavailable', {}): return '不可用：' + r['unavailable'][axis]
    if axis.startswith('AI ') and 'truncated capture' in r.get('flags', []): return '回复截断'
    return '未填'

def fmt(v): return '' if v is None else (v if isinstance(v, str) else '; '.join(v))

def compare(allocation: Path, file_a: Path, file_b: Path, out: Path, describe_a: str = '', describe_b: str = '',
            reference_b: Path | None = None, groups: Path | None = None) -> dict:
    allocation, out = Path(allocation), Path(out)
    if out.exists():
        raise FileExistsError(f'{out} exists; choose a new output name')
    mp = allocation / 'manifest.json'
    manifest = json.loads(mp.read_text(encoding='utf-8')); msha = sha256(mp)
    targets = [(t['conversation_id'], int(t['segment_order'])) for t in manifest['targets']]
    ra, rca = load_checked(file_a, msha)
    rb, rcb = load_checked(file_b, msha) if reference_b is None else load_reference(file_b, reference_b, targets)
    ca, cb = rca['coder_ids'][0], rcb['coder_ids'][0]
    if ca == cb:
        raise ValueError(f'both files carry coder id {ca!r}; a comparison needs two coders')
    A = {(r['conversation_id'], r['segment_order']): r for r in ra}
    B = {(r['conversation_id'], r['segment_order']): r for r in rb}
    if set(A) != set(targets) or set(B) != set(targets):
        raise ValueError('record keys do not match the allocation targets')
    group_list, assign = load_groups(groups) if groups is not None else (None, None)
    before = (sha256(file_a), sha256(file_b))
    rows, counts, disc = [], {ax: {} for ax in DIRECT_CODES}, []
    for k in targets:
        for ax in DIRECT_CODES:
            va, vb = A[k]['labels'][ax], B[k]['labels'][ax]
            st = status(ax, va, vb)
            only_a = only_b = ''
            if DIRECT_FIELDS[ax]['selection'] == 'multiple' and va is not None and vb is not None:
                only_a = '; '.join(x for x in va if x not in vb); only_b = '; '.join(x for x in vb if x not in va)
            rows.append([k[0], k[1], ax, fmt(va), fmt(vb), st, only_a, only_b, empty_reason(A[k], ax), empty_reason(B[k], ax)])
            counts[ax][st] = counts[ax].get(st, 0) + 1
            if st != '相同' and st != '双方都留空':
                disc.append([k[0], k[1], ax, fmt(va) or empty_reason(A[k], ax), fmt(vb) or empty_reason(B[k], ax), st, only_a, only_b])
        for who, R in ((ca, A[k]), (cb, B[k])):
            if 'needs discussion' in R.get('flags', []):
                disc.append([k[0], k[1], '（整条）', '', '', f'{who} 标了 needs discussion', '', ''])
    disc_header = ['对话编号', '轮次', '维度', f'A：{ca}', f'B：{cb}', '状态', '仅 A 选', '仅 B 选', '分歧类型', '讨论要点',
                   '讨论后结论（取值或“未解决”）', '是否需要修订手册', '修订内容', '记录人', '日期']
    disc_widths = [14, 6, 18, 26, 26, 16, 18, 18, 12, 36, 24, 10, 30, 10, 12]
    if assign is not None:
        keys = [(d[0], d[1], d[2]) for d in disc]
        if set(keys) != set(assign):
            raise ValueError(f'groups file does not match the discussion rows: {len(set(keys) - set(assign))} unassigned, '
                             f'{len(set(assign) - set(keys))} assigned rows that are not discussion rows')
        order = {x['id']: i for i, x in enumerate(group_list)}
        disc = [[assign[k][0], '是' if assign[k][1] else ''] + d
                for _, k, d in sorted(zip(range(len(disc)), keys, disc), key=lambda t: (order[assign[t[1]][0]], t[0]))]
        disc_header = ['讨论组', '代表'] + disc_header; disc_widths = [8, 6] + disc_widths
    wb = openpyxl.Workbook()
    hdr_fill = PatternFill('solid', fgColor='DDE4EE')
    def keep_text(ws):
        # openpyxl stores any string starting with '=' as a formula; coder notes must stay literal text.
        for row in ws.iter_rows():
            for c in row:
                if isinstance(c.value, str): c.data_type = 's'
    def put(ws, header, data, widths):
        ws.append(header)
        for c in ws[1]: c.font = Font(bold=True); c.fill = hdr_fill; c.alignment = Alignment(wrap_text=True, vertical='top')
        for d in data: ws.append(d)
        for i, w in enumerate(widths, 1): ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w
        for row in ws.iter_rows(min_row=2):
            for c in row: c.alignment = Alignment(wrap_text=True, vertical='top')
        ws.freeze_panes = 'D2'; ws.auto_filter.ref = ws.dimensions
        keep_text(ws)
    ws = wb.active; ws.title = '说明'
    info = [('用途', '两份已存档记录的逐轴对照，供校准讨论使用。两份记录的性质可能不同，见“记录 A 的性质”“记录 B 的性质”。'),
            ('不是什么', '不是一致性指标，没有计算任何统计量；讨论前一致率由 coder_agreement.py 另行计算（D-148、D-150）。两份存档记录都没有被修改。若其中一份不是独立首轮，这张表也不能当作两份独立判断的比较。'),
            ('记录 A', f"{ca}：{Path(file_a).name}，SHA-256 {rca['sha256']}"),
            ('记录 A 的性质', describe_a or '未注明（请用 --describe-a 说明，例如“第二编码员独立首轮”）'),
            ('记录 B', f"{cb}：{Path(file_b).name}，SHA-256 {rcb['sha256']}"
                      + (f"；参照文件，没有回执，按 {rcb['reference_registry']} 所列 SHA-256 核对（D-150）" if reference_b is not None else '')),
            ('记录 B 的性质', describe_b or '未注明（请用 --describe-b 说明，例如“研究者版本：由 67 条开发复核整理，经 DW 确认，不是独立首轮”）'),
            ('分配', f"{allocation.as_posix()}，manifest SHA-256 {msha}，{len(targets)} 个目标，阶段 {manifest.get('phase')}"),
            ('状态含义', '相同；不同（单选）；部分重叠、无重叠（多选）；仅一方留空；双方都留空。留空原因见各自“留空原因”列。'),
            ('讨论后', '讨论结论写在“讨论记录”页。若要形成讨论后的标签，另存一份 stage 为 adjudicated 的记录文件，用 direct_coding.py 校验；不改已存档的文件。')]
    ws.append(['项目', '内容']); [setattr(c, 'font', Font(bold=True)) for c in ws[1]]
    for r in info: ws.append(list(r))
    ws.column_dimensions['A'].width = 14; ws.column_dimensions['B'].width = 110
    keep_text(ws)
    for row in ws.iter_rows(min_row=2):
        for c in row: c.alignment = Alignment(wrap_text=True, vertical='top')
    put(wb.create_sheet('逐轴对照'), ['对话编号', '轮次', '维度', f'A：{ca}', f'B：{cb}', '状态', '仅 A 选', '仅 B 选', 'A 留空原因', 'B 留空原因'],
        rows, [14, 6, 22, 30, 30, 12, 20, 20, 18, 18])
    notes = []
    for k in targets:
        notes.append([k[0], k[1], '; '.join(A[k].get('flags', [])), '; '.join(B[k].get('flags', [])), A[k].get('notes', ''), B[k].get('notes', '')])
    put(wb.create_sheet('标记与备注'), ['对话编号', '轮次', 'A 标记', 'B 标记', 'A 备注', 'B 备注'], notes, [14, 6, 24, 24, 40, 40])
    states = ['相同', '不同', '部分重叠', '无重叠', '仅一方留空', '双方都留空']
    cnt = [[ax] + [counts[ax].get(s, 0) for s in states] for ax in DIRECT_CODES]
    put(wb.create_sheet('讨论清单计数'), ['维度（计数只用于安排讨论，不是一致性指标）'] + states, cnt, [40] + [12] * len(states))
    wsd = wb.create_sheet('讨论记录')
    put(wsd, disc_header, disc, disc_widths)
    if disc:
        dv = DataValidation(type='list', formula1='"' + ','.join(DISAGREEMENT_TYPES) + '"', allow_blank=True)
        dv2 = DataValidation(type='list', formula1='"是,否"', allow_blank=True)
        wsd.add_data_validation(dv); wsd.add_data_validation(dv2)
        col = lambda h: openpyxl.utils.get_column_letter(disc_header.index(h) + 1)
        dv.add(f"{col('分歧类型')}2:{col('分歧类型')}{len(disc) + 1}"); dv2.add(f"{col('是否需要修订手册')}2:{col('是否需要修订手册')}{len(disc) + 1}")
    if group_list is not None:
        n = {}; reps = {}
        for d in disc:
            n[d[0]] = n.get(d[0], 0) + 1
            if d[1]: reps.setdefault(d[0], []).append(f'{d[2]} #{d[3]}')
        put(wb.create_sheet('讨论组', 1), ['讨论组', '维度', '分歧', '行数', '代表条目', '研究者的编码经验（复核笔记，待与第二编码员讨论）', '决定编号'],
            [[x['id'], x['axis'], x['title'], n.get(x['id'], 0), '; '.join(reps.get(x['id'], [])), x.get('experience', ''), x.get('refs', '')]
             for x in group_list], [8, 18, 36, 6, 30, 90, 22])
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    if (sha256(file_a), sha256(file_b)) != before:
        raise RuntimeError('an input record changed during comparison')
    return {'out': str(out), 'targets': len(targets), 'discussion_rows': len(disc), 'counts': counts, 'coders': [ca, cb],
            'groups': None if group_list is None else {x['id']: sum(1 for d in disc if d[0] == x['id']) for x in group_list}}

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--allocation', type=Path, required=True)
    p.add_argument('file_a', type=Path); p.add_argument('file_b', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--describe-a', default='', help='what record A is, e.g. an independent first round')
    p.add_argument('--describe-b', default='', help='what record B is, e.g. the researcher version from the development review')
    p.add_argument('--reference-b', type=Path, default=None,
                   help='record B is a fixed reference without a receipt, checked against the full SHA-256 this registry README lists (D-150)')
    p.add_argument('--groups', type=Path, default=None, help='JSON assigning each discussion row to a discussion group')
    a = p.parse_args()
    try:
        r = compare(a.allocation, a.file_a, a.file_b, a.out, a.describe_a, a.describe_b, a.reference_b, a.groups)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        print(f'REFUSED: {e}', file=sys.stderr); sys.exit(2)
    print(json.dumps(r, ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
