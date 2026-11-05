"""Read direct human records without deriving or changing any judgement."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from codebook_schema import DIRECT_CODES, DIRECT_FIELDS, DIRECT_VERSION, DIRECT_INSTRUMENT, DIRECT_DEFAULT

NEGATIVES = DIRECT_INSTRUMENT['negative_exclusive']

def validate(record, allow_blank=False):
    if not isinstance(record, dict):
        return ['record must be an object']
    errors = []
    if record.get('schema_version') != DIRECT_VERSION:
        errors.append('incorrect schema version')
    if not record.get('conversation_id') or type(record.get('segment_order')) is not int or record['segment_order'] < 1:
        errors.append('missing or invalid interaction key')
    if not record.get('coder_id'):
        errors.append('missing coder identity')
    if record.get('stage') not in ('calibration', 'independent', 'adjudicated'):
        errors.append('invalid stage')
    labels = record.get('labels', {})
    if not isinstance(labels, dict):
        return errors + ['labels must be an object']
    if set(labels) != set(DIRECT_CODES):
        errors.append('exactly four label fields are required')
    flags = record.get('flags', [])
    if not isinstance(flags, list) or any(x not in DIRECT_INSTRUMENT['metadata_flags'] for x in flags):
        errors.append('invalid flags')
        flags = []
    unavailable = record.get('unavailable', {})
    if not isinstance(unavailable, dict) or any(k not in DIRECT_CODES or not isinstance(v, str) or not v.strip() for k,v in unavailable.items()):
        errors.append('invalid unavailable reasons')
        unavailable = {}
    for axis, codes in DIRECT_CODES.items():
        v = labels.get(axis)
        if axis.startswith('AI ') and 'truncated capture' in flags:
            if v is not None:
                errors.append(f'{axis}: truncated response must remain null')
            continue
        if v is None:
            if not allow_blank and axis not in unavailable:
                errors.append(f'{axis}: unfilled judgement')
            continue
        if axis in unavailable:
            errors.append(f'{axis}: label conflicts with unavailable reason')
        if DIRECT_FIELDS[axis]['selection'] == 'single':
            if not isinstance(v, str) or v not in codes:
                errors.append(f'{axis}: invalid single choice')
        else:
            if not isinstance(v, list) or not v or any(not isinstance(x, str) or x not in codes for x in v):
                errors.append(f'{axis}: invalid nonempty label set')
            elif len(v) != len(set(v)) or (NEGATIVES.get(axis) in v and len(v) > 1):
                errors.append(f'{axis}: duplicate or incompatible selections')
    if DIRECT_DEFAULT['flag'] in flags:
        if labels.get(DIRECT_DEFAULT['axis']) != DIRECT_DEFAULT['value'] or not str(record.get('notes', '')).strip():
            errors.append('Task automation default requires that human choice and a reason')
    return errors

def read_records(path, allow_blank=False):
    rows, seen = [], set()
    # Split at LF (read_text has turned CRLF and a lone CR into LF): splitlines() also breaks at
    # U+0085/U+2028/U+2029, which a JSON string may contain unescaped.
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').split('\n'), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            detail = f'{exc.msg} at column {exc.colno}' if isinstance(exc, json.JSONDecodeError) else str(exc)
            raise ValueError(f'line {line_no}: not valid JSON ({detail})') from exc
        errors = validate(row, allow_blank)
        key = ((row.get('coder_id'), row.get('stage'), row.get('conversation_id'), row.get('segment_order'))
               if isinstance(row, dict) else None)
        if key in seen:
            errors.append('duplicate coder/stage/interaction')
        if errors:
            raise ValueError(f'line {line_no}: ' + '; '.join(errors))
        seen.add(key); rows.append(row)
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('records', type=Path)
    p.add_argument('--allow-blank', action='store_true', help='Check a blank allocation, not completed coding')
    args = p.parse_args()
    rows = read_records(args.records, args.allow_blank)
    filled = {a: sum(r['labels'][a] is not None for r in rows) for a in DIRECT_CODES}
    print(json.dumps({'records_checked': len(rows), 'filled_fields': filled,
        'scope': 'structure only; no label correctness, independence or reliability assertion'}, indent=2))

if __name__ == '__main__':
    main()
