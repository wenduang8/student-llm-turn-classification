"""Agreement between two coders' records (D-148, D-150; measure D-179, D-201).

The reported measure is simple agreement per code, C / max(A, B): A and B are the interactions each
record codes with the code, C those both do, over the interactions both records code on that
dimension. Solve-Provide and Explain are also given on the absent side (the same formula on the
interactions without the code). The figures below this paragraph are kept as records only: the
thesis reports neither exact-set agreement nor F1, and whether Cohen's kappa is reported is open
(D-201).

For each dimension: percentage agreement over the interactions that both records code. A
multi-select dimension agrees only when the two label sets are identical (exact set match); order
does not matter. Cohen's kappa is added for the single-select student interaction type. A flagged
Task automation default counts as Task automation. Interactions that one or both records leave
empty on a dimension (for example a truncated assistant response) are counted and reported but not
scored. No figure across dimensions is computed: two coders can agree on three dimensions and
differ on the fourth, and that must stay visible.

F1 is reported beside these (D-158): per code, F1 = 2 x both / (selections by A + selections by B),
which is symmetric between the coders (positive specific agreement). The single-select dimension
gets per-class F1 and macro-F1; a multi-select dimension gets per-code F1, micro-F1 pooled over all
codes of the schema vocabulary (negative codes included, as in the multi-hot encoding) and macro-F1
over the codes at least one coder used. F1 gives partial credit and is not exact set match.

Inputs are read with the guards of compare_coders.py: record A is a return stored by
receive_return.py with a passing receipt; record B is either such a return or, with --reference-b,
a fixed reference checked against the full SHA-256 its registry README lists (D-150). Neither input
is changed. The output is one new JSON file and a Markdown summary beside it; existing files are
never overwritten.

    python coding/coder_agreement.py --allocation <dir> <A-stored.jsonl> <B.jsonl> --out <new.json>
        [--reference-b data/labels/README.md] [--describe-a "..."] [--describe-b "..."]
"""
from __future__ import annotations
import argparse, datetime, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'coding'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from direct_coding import DIRECT_CODES, DIRECT_FIELDS
from compare_coders import load_checked, load_reference, sha256

def cohen_kappa(pairs: list, categories: list) -> float | None:
    """Unweighted Cohen's kappa for paired single-choice labels."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(a == b for a, b in pairs) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in categories) / (n * n)
    return None if pe == 1 else (po - pe) / (1 - pe)

def f1(both: int, a: int, b: int) -> float | None:
    return None if a + b == 0 else 2 * both / (a + b)

def r3(x): return None if x is None else round(x, 3)

ABSENT_SIDE = {'AI response type': ('Solve-Provide', 'Explain')}

def simple_agreement(scored: list, axis: str, single: bool) -> dict:
    """Per code A, B, C and C / max(A, B) on the scored pairs (D-179); absent side where D-201 asks for it."""
    has = (lambda v, c: v == c) if single else (lambda v, c: c in v)
    sides = [(c, False) for c in DIRECT_CODES[axis]] + [(c, True) for c in ABSENT_SIDE.get(axis, ())]
    out = {}
    for c, absent in sides:
        a = sum(has(x, c) != absent for x, _ in scored)
        b = sum(has(y, c) != absent for _, y in scored)
        both = sum((has(x, c) != absent) and (has(y, c) != absent) for x, y in scored)
        out[c + (' (absent)' if absent else '')] = {'A': a, 'B': b, 'C': both,
                                                    'simple_agreement': r3(both / max(a, b)) if max(a, b) else None}
    return out

def axis_agreement(axis: str, A: dict, B: dict, keys: list) -> dict:
    single = DIRECT_FIELDS[axis]['selection'] == 'single'
    codes = list(DIRECT_CODES[axis])
    scored, one_empty, both_empty, states = [], 0, 0, Counter()
    for k in keys:
        a, b = A[k]['labels'][axis], B[k]['labels'][axis]
        if a is None and b is None: both_empty += 1; continue
        if a is None or b is None: one_empty += 1; continue
        scored.append((a, b))
        if single:
            states['相同' if a == b else '不同'] += 1
        else:
            sa, sb = set(a), set(b)
            states['相同' if sa == sb else ('部分重叠' if sa & sb else '无重叠')] += 1
    n, agree = len(scored), states['相同']
    out = {'per_code_simple_agreement': simple_agreement(scored, axis, single),
           'selection': 'single' if single else 'multiple (exact set match)', 'scored': n, 'agree': agree,
           'percent': None if n == 0 else round(100 * agree / n, 1),
           'one_side_empty_not_scored': one_empty, 'both_empty_not_scored': both_empty, 'status_counts': dict(states)}
    if single:
        k = cohen_kappa(scored, codes)
        out['cohen_kappa'] = None if k is None else round(k, 3)
        cm = {ca: {cb: sum(1 for a, b in scored if a == ca and b == cb) for cb in codes} for ca in codes}
        out['confusion_rows_A_cols_B'] = cm
        per = {c: f1(cm[c][c], sum(cm[c].values()), sum(cm[x][c] for x in codes)) for c in codes}
        used = [v for v in per.values() if v is not None]
        out['f1'] = {'macro_f1': r3(sum(used) / len(used)) if used else None, 'per_class_f1': {c: r3(v) for c, v in per.items()}}
    else:
        out['per_code_descriptive'] = {c: {'A': sum(c in a for a, _ in scored), 'B': sum(c in b for _, b in scored),
                                           'both': sum(c in a and c in b for a, b in scored)} for c in codes}
        pc = out['per_code_descriptive']
        per = {c: f1(pc[c]['both'], pc[c]['A'], pc[c]['B']) for c in codes}
        used = [v for v in per.values() if v is not None]
        out['f1'] = {'micro_f1': r3(f1(sum(x['both'] for x in pc.values()), sum(x['A'] for x in pc.values()), sum(x['B'] for x in pc.values()))),
                     'macro_f1': r3(sum(used) / len(used)) if used else None, 'per_code_f1': {c: r3(v) for c, v in per.items()}}
    return out

def agreement(allocation: Path, file_a: Path, file_b: Path, out: Path, describe_a: str = '', describe_b: str = '',
              reference_b: Path | None = None) -> dict:
    allocation, out = Path(allocation), Path(out)
    if out.suffix != '.json':
        raise ValueError('--out must end in .json (a Markdown summary is written beside it)')
    md = out.with_suffix('.md')
    if out.exists() or md.exists():
        raise FileExistsError(f'{out.name} or {md.name} exists; choose a new output name')
    mp = allocation / 'manifest.json'
    manifest = json.loads(mp.read_text(encoding='utf-8')); msha = sha256(mp)
    targets = [(t['conversation_id'], int(t['segment_order'])) for t in manifest['targets']]
    ra, rca = load_checked(file_a, msha)
    rb, rcb = load_checked(file_b, msha) if reference_b is None else load_reference(file_b, reference_b, targets)
    ca, cb = rca['coder_ids'][0], rcb['coder_ids'][0]
    if ca == cb:
        raise ValueError(f'both files carry coder id {ca!r}; agreement needs two coders')
    A = {(r['conversation_id'], r['segment_order']): r for r in ra}
    B = {(r['conversation_id'], r['segment_order']): r for r in rb}
    if set(A) != set(targets) or set(B) != set(targets):
        raise ValueError('record keys do not match the allocation targets')
    before = (sha256(file_a), sha256(file_b))
    result = {
        'what': 'agreement between two records: per-code simple agreement C / max(A, B), the reported measure (D-179, D-201); '
                'exact set match per dimension, Cohen kappa and F1 kept as records only',
        'decisions': 'D-148, D-150, D-158, D-179, D-201',
        'record_A': {'coder': ca, 'file': Path(file_a).as_posix(), 'sha256': rca['sha256'], 'describe': describe_a},
        'record_B': {'coder': cb, 'file': Path(file_b).as_posix(), 'sha256': rcb['sha256'], 'describe': describe_b,
                     'reference_registry': rcb.get('reference_registry')},
        'allocation': allocation.as_posix(), 'manifest_sha256': msha, 'interactions': len(targets),
        'computed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        'axes': {ax: axis_agreement(ax, A, B, targets) for ax in DIRECT_CODES},
        'not_computed': 'no figure across dimensions (CLAUDE.md, Evaluation)'}
    if (sha256(file_a), sha256(file_b)) != before:
        raise RuntimeError('an input record changed during computation')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    lines = ['# 两份记录的一致率', '',
             f"A：{ca}（{describe_a or '未注明'}），SHA-256 {rca['sha256']}",
             f"B：{cb}（{describe_b or '未注明'}），SHA-256 {rcb['sha256']}", '',
             '## 逐代码简单一致率 C / max(A, B)（D-179、D-201，论文报告的度量）', '', 'A、B为两份记录各自选该代码的互动数，C为两份都选的数，只算两份记录在该维度都有值的互动。', '', '| 维度 | 代码 | A | B | C | C/max |', '|---|---|---|---|---|---|'] + [f"| {ax} | {c} | {d['A']} | {d['B']} | {d['C']} | {'' if d['simple_agreement'] is None else format(100 * d['simple_agreement'], '.1f') + '%'} |" for ax, r in result['axes'].items() for c, d in r['per_code_simple_agreement'].items()] + ['', '## 以下仅作记录（D-201：论文不报整组一致率和F1；κ是否报告待定）', '', 
             f"{len(targets)}条互动，按维度计算；多选维度整组相同才算一致；学生互动类型另报Cohen's κ。一方或双方留空的互动不计入分母，单列。不算跨维度的总一致率。由 `coding/coder_agreement.py` 生成，数字以同名JSON为准。", '',
             '| 维度 | 一致 | 计入 | 一致率 | κ | micro-F1 | macro-F1 | 部分重叠 | 无重叠 | 不同（单选） | 未计入 |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for ax, r in result['axes'].items():
        s = r['status_counts']
        lines.append(f"| {ax} | {r['agree']} | {r['scored']} | {r['percent']}% | {r.get('cohen_kappa', '')} | {r['f1'].get('micro_f1', '')} | {r['f1']['macro_f1']} | {s.get('部分重叠', '')} | {s.get('无重叠', '')} | {s.get('不同', '')} | {r['one_side_empty_not_scored'] + r['both_empty_not_scored']} |")
    lines += ['', 'F1按代码算：2×两人都选 /（A选的次数 + B选的次数），两位编码员对调结果相同。单选维度给每类F1和macro-F1；多选维度给每个代码的F1、按全部代码合并的micro-F1（含ND和none of the four applies）和macro-F1（只算至少一人用过的代码）。F1给部分重合计分，不是整组一致。', '',
              '| 维度 | 代码 | A选 | B选 | 都选 | F1 |', '|---|---|---|---|---|---|']
    for ax, r in result['axes'].items():
        if 'per_code_descriptive' in r:
            for c, d in r['per_code_descriptive'].items():
                lines.append(f"| {ax} | {c} | {d['A']} | {d['B']} | {d['both']} | {r['f1']['per_code_f1'][c]} |")
        else:
            cm = r['confusion_rows_A_cols_B']
            for c in cm:
                lines.append(f"| {ax} | {c} | {sum(cm[c].values())} | {sum(cm[x][c] for x in cm)} | {cm[c][c]} | {r['f1']['per_class_f1'][c]} |")
    md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--allocation', type=Path, required=True)
    p.add_argument('file_a', type=Path); p.add_argument('file_b', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--describe-a', default=''); p.add_argument('--describe-b', default='')
    p.add_argument('--reference-b', type=Path, default=None)
    a = p.parse_args()
    try:
        r = agreement(a.allocation, a.file_a, a.file_b, a.out, a.describe_a, a.describe_b, a.reference_b)
    except (ValueError, FileExistsError, FileNotFoundError) as e:
        print(f'REFUSED: {e}', file=sys.stderr); sys.exit(2)
    print(json.dumps({ax: {c: f"{d['C']}/{max(d['A'], d['B'])}" + ('' if d['simple_agreement'] is None else f" {100 * d['simple_agreement']:.1f}%")
                           for c, d in v['per_code_simple_agreement'].items()} for ax, v in r['axes'].items()},
                     ensure_ascii=False, indent=1))

if __name__ == '__main__':
    main()
