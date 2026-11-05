"""Receive a coder's returned record file: store it unchanged first, then check it.

The returned file is copied byte for byte into <allocation>/responses/received/ under a name that
is never reused, and a receipt is written beside it. Checks follow: the existing record validator
(direct_coding.validate) on every line, plus allocation-level checks the validator does not make
(the exact target set in manifest.json, one coder, the allocation's stage and version). A failed
check never blocks storage and never edits the stored copy. Nothing here changes a judgement.

    python coding/receive_return.py <returned.jsonl> --allocation <allocation dir> --coder <name>
    options: --label first (default; use v2, v3 ... for later versions)  --draft (allow blanks)
Exit status: 0 checks pass, 1 checks fail (file still stored), 2 refused (nothing stored).
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, re, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / 'coding'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from direct_coding import validate, DIRECT_CODES, DIRECT_VERSION

def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def slug(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9_-]+', '_', s.strip())
    if not s:
        raise ValueError('coder name is empty after removing unsupported characters')
    return s

TEXT_FIELDS = ('schema_version', 'coder_id', 'stage', 'conversation_id')   # notes too, when present

def as_text(v) -> str:
    """Hashable, printable form of a field value of any JSON type."""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)

def check(path: Path, manifest: dict, draft: bool = False) -> dict:
    """Return a check report for a record file. Reads only."""
    stage = manifest.get('phase')
    version = manifest.get('schema_version', DIRECT_VERSION)
    targets = [(t['conversation_id'], int(t['segment_order'])) for t in manifest['targets']]
    source_truncated = {(t['conversation_id'], int(t['segment_order']))
                        for t in manifest.get('source_truncated_targets', [])}
    if not source_truncated.issubset(set(targets)):
        raise ValueError('Source-truncated manifest keys must be allocated targets')
    line_errors, keys, coders, stages, versions, rows = [], [], set(), set(), set(), []
    text = path.read_text(encoding='utf-8-sig')
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            line_errors.append({'line': n, 'errors': [f'not valid JSON: {e.msg}']})
            continue
        try:
            errs = validate(row, allow_blank=draft)
        except Exception as e:   # the validator must not stop the report
            errs = [f'validator could not read this record: {type(e).__name__}']
        if isinstance(row, dict):
            wrong = [f for f in TEXT_FIELDS if not isinstance(row.get(f), str)]
            if 'notes' in row and not isinstance(row['notes'], str): wrong.append('notes')
            if wrong:
                errs = list(errs) + ['must be text: ' + ', '.join(wrong)]
            cid, seg = row.get('conversation_id'), row.get('segment_order')
            if isinstance(cid, str) and type(seg) is int:
                keys.append((cid, seg))
                if (cid, seg) in source_truncated:
                    flags = row.get('flags')
                    labels = row.get('labels')
                    if not isinstance(flags, list) or 'truncated capture' not in flags:
                        errs = list(errs) + ['source check requires the truncated capture flag to stay set']
                    if (not isinstance(labels, dict)
                            or labels.get('AI delegation') is not None
                            or labels.get('AI response type') is not None):
                        errs = list(errs) + ['source check requires both AI fields to stay blank']
            else:
                errs = list(errs) + ['conversation_id must be text and segment_order an integer']
            coders.add(as_text(row.get('coder_id'))); stages.add(as_text(row.get('stage'))); versions.add(as_text(row.get('schema_version')))
            rows.append(row)
        else:
            errs = list(errs) + ['record is not a JSON object']
        if errs:
            line_errors.append({'line': n, 'errors': errs})
    seen, dup = set(), []
    for k in keys:
        if k in seen:
            dup.append(k)
        seen.add(k)
    missing = [k for k in targets if k not in seen]
    extra = sorted({k for k in keys if k not in set(targets)}, key=str)
    problems = []
    if line_errors: problems.append(f'{len(line_errors)} line(s) fail the record validator')
    if missing: problems.append(f'{len(missing)} target(s) missing')
    if extra: problems.append(f'{len(extra)} key(s) not in the allocation')
    if dup: problems.append(f'{len(dup)} duplicate key(s)')
    if len(coders) != 1: problems.append(f'coder ids in file: {sorted(map(str, coders))}; expected exactly one')
    if stages != {stage}: problems.append(f'stages in file: {sorted(map(str, stages))}; allocation stage is {stage}')
    if versions != {version}: problems.append(f'schema versions in file: {sorted(map(str, versions))}; allocation version is {version}')
    completeness = {}
    for axis in DIRECT_CODES:
        c = {'filled': 0, 'blank': 0, 'unavailable': 0, 'null_truncated_capture': 0}
        for r in rows:
            labels = r.get('labels') if isinstance(r.get('labels'), dict) else {}
            unavailable = r.get('unavailable') if isinstance(r.get('unavailable'), dict) else {}
            flags = r.get('flags') if isinstance(r.get('flags'), list) else []
            v = labels.get(axis)
            if v is not None: c['filled'] += 1
            elif axis in unavailable: c['unavailable'] += 1
            elif axis.startswith('AI ') and 'truncated capture' in flags: c['null_truncated_capture'] += 1
            else: c['blank'] += 1
        completeness[axis] = c
    return {'status': 'passes' if not problems else 'fails', 'problems': problems, 'mode': 'draft (blanks allowed)' if draft else 'complete return',
            'records': len(rows), 'targets_expected': len(targets), 'missing_targets': [list(k) for k in missing],
            'extra_keys': [list(k) for k in extra], 'duplicate_keys': [list(k) for k in dup], 'line_errors': line_errors,
            'coder_ids': sorted(map(str, coders)), 'completeness_by_axis': completeness,
            'scope': 'format, target keys and completeness only; no judgement is checked, changed or compared'}

def receive(src: Path, allocation: Path, coder: str, label: str = 'first', draft: bool = False) -> tuple[Path, dict]:
    src, allocation = Path(src), Path(allocation)
    manifest_path = allocation / 'manifest.json'
    if not src.is_file():
        raise FileNotFoundError(f'returned file not found: {src}')
    if not manifest_path.is_file():
        raise FileNotFoundError(f'no manifest.json in {allocation}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    dest_dir = allocation / 'responses' / 'received'
    dest = dest_dir / f"{slug(coder)}-{manifest.get('phase', 'unknown')}-{slug(label)}.jsonl"
    receipt_path = dest.with_name(dest.name + '.receipt.json')
    if dest.exists() or receipt_path.exists():
        raise FileExistsError(f'{dest.name} already stored; a first return is never replaced. Use --label v2 (or later) for a new version.')
    dest_dir.mkdir(parents=True, exist_ok=True)
    before = sha256(src)
    shutil.copyfile(src, dest)
    if sha256(dest) != before:
        raise IOError('stored copy differs from the returned file')
    try:
        report = check(dest, manifest, draft)
    except Exception as e:   # the stored copy must always get a receipt
        report = {'status': 'fails', 'problems': [f'checker stopped with {type(e).__name__}: {e}; the stored file is unchanged'],
                  'mode': 'draft (blanks allowed)' if draft else 'complete return', 'records': None, 'coder_ids': [],
                  'completeness_by_axis': {}, 'scope': 'checks did not complete'}
    receipt = {'stored_file': dest.name, 'sha256': before, 'bytes': dest.stat().st_size, 'returned_filename': src.name,
               'coder_label_given': coder, 'version_label': label,
               'received_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
               'allocation_manifest_sha256': sha256(manifest_path), **report}
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=1), encoding='utf-8')
    return dest, receipt

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('returned', type=Path)
    p.add_argument('--allocation', type=Path, required=True)
    p.add_argument('--coder', required=True)
    p.add_argument('--label', default='first')
    p.add_argument('--draft', action='store_true')
    a = p.parse_args()
    try:
        dest, r = receive(a.returned, a.allocation, a.coder, a.label, a.draft)
    except (FileExistsError, FileNotFoundError, ValueError) as e:
        print(f'REFUSED: {e}', file=sys.stderr); sys.exit(2)
    print(json.dumps({'stored': str(dest), 'sha256': r['sha256'], 'status': r['status'], 'problems': r['problems'],
                      'completeness_by_axis': r['completeness_by_axis']}, ensure_ascii=False, indent=1))
    sys.exit(0 if r['status'] == 'passes' else 1)

if __name__ == '__main__':
    main()
