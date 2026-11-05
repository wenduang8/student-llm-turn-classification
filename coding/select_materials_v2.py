"""D-87 allocation. New state/output only; no labels, training, or external send.

The original-298 lock is required by the production entry point and exporter.
Engine is an in-memory state machine, also used with synthetic pools in tests.
Ranking, first-batch optimization and E0--E7b pool construction are imported
unchanged from v1. Review outcomes change conversation eligibility, not ranking.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import html
import json
import re
from pathlib import Path

import select_materials as v1

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / 'data/maintenance/20261006-source-recheck'
OUT = ROOT / 'data/coding/direct-20260924/allocation-20261006'
VERSION = 'source-check-v2-20261006'
STATE_VERSION = 'allocation-state-v2-20261006'
QUEUE_POLICY = 'confirm-source-before-continuing-queue-v1'
PHASES = ('evaluation', 'development', 'training_initial', 'training_long')
ACCEPTED = {'pass', 'pass_with_annotations'}
STATUSES = ACCEPTED | {'hold'}
ISSUES = {f'H{i}' for i in range(1, 8)} | {'M1', 'M2', 'D32', 'F1'}
FIELDS = ('Student Prompt', 'LLM Response')
STRATA = v1.STRATA
rank, band, first_batch = v1.rank, v1.band, v1.first_batch
sha, read_json, write_json, csv_rows = v1.sha, v1.read_json, v1.write_json, v1.csv_rows


def canonical_sha(value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def text_sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def local_path(root, relative):
    """Receipt paths must remain repository-relative; never resolve outside it."""
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts or ':' in str(relative):
        raise ValueError(f'Not a repository-relative path: {relative}')
    path = (root / rel).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Path leaves repository: {relative}')
    return path


def load_rows(root=ROOT):
    grouped = collections.defaultdict(list)
    for row in csv_rows(root / 'data/processed/transcripts_analysis.csv'):
        grouped[row['Conversation ID']].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row['Segment Order']))
    return grouped


def build_pool():
    # Do not release E6 entries or reinterpret E4/E5 here. D-87 keeps this gate.
    return v1.build_pool()


def validate_reviews(reviews, pool, rows_by_c, config, root=ROOT):
    """Validate provenance and precise annotations, not the reviewer's judgement."""
    if not isinstance(reviews, dict):
        raise ValueError('Reviews must be a CID-to-receipt object')
    lookup = {p['cid']: p for p in pool}
    rules_path = local_path(root, config['rules_path'])
    if config['review_version'] != VERSION or sha(rules_path) != config['rules_sha256']:
        raise ValueError('Adopted source rules changed')
    for cid, receipt in reviews.items():
        if cid not in lookup or not isinstance(receipt, dict):
            raise ValueError(f'Review outside pool or malformed: {cid}')
        p = lookup[cid]
        status = receipt.get('status')
        if status not in STATUSES:
            raise ValueError(f'Invalid status: {cid}')
        if (receipt.get('reviewer_kind') != 'AI_source_review_v2'
                or receipt.get('rules_version') != VERSION
                or receipt.get('rules_sha256') != config['rules_sha256']):
            raise ValueError(f'Review rule/provenance mismatch: {cid}')
        for key in ('reason', 'scope', 'normalized_sha256'):
            if not isinstance(receipt.get(key), str) or not receipt[key].strip():
                raise ValueError(f'Missing text field {cid}: {key}')
        count = receipt.get('analysis_rows_checked')
        if type(count) is not int or not 0 <= count <= p['n']:
            raise ValueError(f'Invalid checked row count: {cid}')
        if status in ACCEPTED and count != p['n']:
            raise ValueError(f'Accepted review must cover all rows: {cid}')
        if type(receipt.get('sensitive_content_present')) is not bool:
            raise ValueError(f'Missing sensitivity status: {cid}')
        codes = receipt.get('issue_codes')
        if (not isinstance(codes, list) or any(not isinstance(x, str) for x in codes)
                or len(codes) != len(set(codes)) or not set(codes) <= ISSUES):
            raise ValueError(f'Invalid issue codes: {cid}')
        if any(c.startswith('H') for c in codes) != (status == 'hold'):
            raise ValueError(f'Hold/status disagreement: {cid}')
        sources = receipt.get('source_paths')
        if not isinstance(sources, list) or len(sources) < 2:
            raise ValueError(f'Need original and normalized receipts: {cid}')
        paths = set()
        for source in sources:
            if not isinstance(source, dict) or not isinstance(source.get('path'), str):
                raise ValueError(f'Invalid source receipt: {cid}')
            path = local_path(root, source['path'])
            if path in paths or not path.is_file() or sha(path) != source.get('sha256'):
                raise ValueError(f'Duplicate, missing or changed source: {cid}: {path}')
            paths.add(path)
        normalized = local_path(root, p['normalized_path'])
        if normalized not in paths or sha(normalized) != receipt['normalized_sha256']:
            raise ValueError(f'Normalized receipt mismatch: {cid}')
        rows = rows_by_c.get(cid, [])
        by_order = {int(row['Segment Order']): row for row in rows}
        if len(rows) != p['n'] or len(by_order) != len(rows):
            raise ValueError(f'Inventory/analysis count or key mismatch: {cid}')
        annotations = receipt.get('annotations')
        truncated = receipt.get('truncated_orders')
        if not isinstance(annotations, list) or not isinstance(truncated, list):
            raise ValueError(f'Annotations and truncated_orders must be lists: {cid}')
        if (any(type(order) is not int or order not in by_order for order in truncated)
                or len(truncated) != len(set(truncated))):
            raise ValueError(f'Invalid truncated order: {cid}')
        intervals = collections.defaultdict(list)
        for ann in annotations:
            if not isinstance(ann, dict):
                raise ValueError(f'Invalid annotation: {cid}')
            order, field = ann.get('segment_order'), ann.get('field')
            start, end = ann.get('start'), ann.get('end')
            if type(order) is not int or order not in by_order or field not in FIELDS:
                raise ValueError(f'Annotation row/field mismatch: {cid}')
            text = by_order[order][field]
            if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text)
                    or ann.get('text') != text[start:end] or ann.get('field_sha256') != text_sha(text)):
                raise ValueError(f'Annotation span/text/hash mismatch: {cid} #{order}')
            if ann.get('kind') not in {'M1', 'M2'} or ann['kind'] not in codes:
                raise ValueError(f'Annotation issue code mismatch: {cid}')
            if not isinstance(ann.get('basis'), str) or not ann['basis'].strip():
                raise ValueError(f'Missing annotation source basis: {cid}')
            intervals[(order, field)].append((start, end))
        for spans in intervals.values():
            spans.sort()
            if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
                raise ValueError(f'Overlapping annotations: {cid}')
        if {a['kind'] for a in annotations} != set(codes) & {'M1', 'M2'}:
            raise ValueError(f'M issue without precise span: {cid}')
        if bool(truncated) != ('D32' in codes):
            raise ValueError(f'D32 issue/order disagreement: {cid}')
        if status in ACCEPTED and (status == 'pass_with_annotations') != bool(annotations or truncated):
            raise ValueError(f'Annotation/status disagreement: {cid}')


def validate_original_lock(config, reviews, root=ROOT, task=TASK, expected_count=298):
    """No old verdict is loaded. Freeze and verify the complete new review set."""
    lock_path = task / 'original-298-lock.json'
    if not lock_path.is_file():
        raise ValueError('All original-298 reviews must be locked before allocation')
    lock = read_json(lock_path)
    if (lock.get('lock_version') != 'original-298-lock-v1'
            or lock.get('rules_version') != VERSION
            or lock.get('rules_sha256') != config['rules_sha256']
            or lock.get('config_sha256') != sha(task / 'selection-config.json')):
        raise ValueError('Original review lock contract mismatch')
    targets_path = local_path(root, lock['original_targets_path'])
    locked_path = local_path(root, lock['locked_reviews_path'])
    if (sha(targets_path) != lock.get('original_targets_sha256')
            or sha(locked_path) != lock.get('locked_reviews_sha256')):
        raise ValueError('Original targets or locked review file changed')
    targets = read_json(targets_path)
    locked = read_json(locked_path)
    hashes = lock.get('receipt_sha256', {})
    if (not isinstance(targets, list) or any(not isinstance(x, str) for x in targets)
            or len(targets) != expected_count or len(set(targets)) != expected_count
            or lock.get('target_count') != expected_count
            or set(locked) != set(targets) or set(hashes) != set(targets)):
        raise ValueError('Original review lock is not the complete target set')
    for cid in targets:
        if (locked[cid].get('status') not in STATUSES
                or canonical_sha(locked[cid]) != hashes[cid]
                or cid not in reviews or canonical_sha(reviews[cid]) != hashes[cid]):
            raise ValueError(f'Locked original review changed/missing: {cid}')
    return sha(lock_path)


class Engine:
    """Fixed v1 ranking/budgets; confirm sources before scanning past occupancy."""
    def __init__(self, config, pool, state=None):
        self.c = config
        self.pool = {p['cid']: p for p in pool}
        if state is not None and (state.get('state_version') != STATE_VERSION
                                  or state.get('queue_policy') != QUEUE_POLICY or 'held_students' in state):
            raise ValueError('Old allocation state/cursors cannot be used by v2')
        self.state = state if state is not None else {
            'state_version': STATE_VERSION, 'queue_policy': QUEUE_POLICY, 'phase': 'evaluation',
            'selected': {k: [] for k in PHASES}, 'cursors': {},
            'held_conversations': [], 'events': [], 'locked': [],
            'locked_review_sha256': {}, 'completed': False,
        }

    def event(self, action, **kw):
        self.state['events'].append({'seq': len(self.state['events']) + 1, 'action': action, **kw})

    def used(self):
        return {self.pool[cid]['student'] for ids in self.state['selected'].values() for cid in ids}

    def total(self, phase, s=None):
        return sum(self.pool[c]['n'] for c in self.state['selected'][phase]
                   if s is None or self.pool[c]['stratum'] == s)

    def next_candidate(self, phase, s, min_n, max_n, cap, stratum_cap=None):
        qkey = f'{phase}|{s or "all"}'
        queue = sorted([p for p in self.pool.values() if (s is None or p['stratum'] == s)
                        and min_n <= p['n'] <= max_n],
                       key=lambda p: rank(self.c['seed'], phase, p['cid']))
        cursor = self.state['cursors'].get(qkey, 0)
        while cursor < len(queue):
            p = queue[cursor]
            cursor += 1
            self.state['cursors'][qkey] = cursor
            reason = None
            if p['cid'] in self.state['held_conversations']:
                reason = 'conversation held'
            elif p['student'] in self.used():
                reason = 'student already selected'
            elif self.total(phase) + p['n'] > cap:
                reason = 'overall cap'
            elif stratum_cap is not None and self.total(phase, s) + p['n'] > stratum_cap:
                reason = 'stratum cap'
            if reason:
                self.event('skip', phase=phase, cid=p['cid'], reason=reason)
                continue
            self.state['selected'][phase].append(p['cid'])
            self.event('propose', phase=phase, cid=p['cid'], queue=qkey,
                       rank=rank(self.c['seed'], phase, p['cid']))
            return True
        return False

    def advance(self, reviews):
        for phase in self.state['locked']:
            for cid in self.state['selected'][phase]:
                receipt = reviews.get(cid, {})
                if (receipt.get('status') not in ACCEPTED
                        or canonical_sha(receipt) != self.state['locked_review_sha256'].get(cid)):
                    raise ValueError(f'Locked source review changed: {cid}; explicit correction required')
        if self.state['completed']:
            return []

        def propose(phase, s, min_n, max_n, cap, stratum_cap=None):
            """Resolve known holds before scanning past their temporary occupancy.

            A new unknown source pauses the entire allocation immediately. Thus
            it cannot make another queue permanently skip the same student's
            conversation, or make an otherwise eligible item fail a budget cap.
            Actual budget skips keep their original cursor/ranking behaviour.
            """
            while self.next_candidate(phase, s, min_n, max_n, cap, stratum_cap):
                cid = self.state['selected'][phase][-1]
                status = reviews.get(cid, {}).get('status')
                if status == 'hold':
                    self.state['selected'][phase].remove(cid)
                    self.state['held_conversations'] = sorted(set(self.state['held_conversations']) | {cid})
                    self.event('source_hold', phase=phase, cid=cid, reason=reviews[cid].get('reason', ''))
                    continue
                if status not in ACCEPTED:
                    return True, [self.pool[cid]]
                return True, []
            return False, []

        for phase in PHASES:
            if phase in self.state['locked']:
                continue
            self.state['phase'] = phase
            unresolved = [self.pool[cid] for cid in self.state['selected'][phase]
                          if reviews.get(cid, {}).get('status') not in STATUSES]
            if unresolved:
                return unresolved
            for cid in list(self.state['selected'][phase]):
                rev = reviews.get(cid)
                if rev and rev.get('status') == 'hold':
                    self.state['selected'][phase].remove(cid)
                    self.state['held_conversations'] = sorted(set(self.state['held_conversations']) | {cid})
                    self.event('source_hold', phase=phase, cid=cid, reason=rev.get('reason', ''))
            if phase == 'evaluation':
                cfg = self.c[phase]
                for s in STRATA:
                    while self.total(phase, s) < cfg['lower_per_stratum']:
                        found, pending = propose(phase, s, 1, 10**9, cfg['maximum'], cfg['upper_per_stratum'])
                        if pending:
                            return pending
                        if not found:
                            break
                while self.total(phase) < cfg['minimum']:
                    found, pending = propose(phase, '1-4', 1, 4, cfg['maximum'], cfg['upper_per_stratum'])
                    if pending:
                        return pending
                    if not found:
                        break
            elif phase == 'development':
                cfg = self.c[phase]
                while self.total(phase) < cfg['minimum'] or len(self.state['selected'][phase]) < cfg['minimum_students']:
                    found, pending = propose(phase, None, 1, cfg['conversation_maximum'], cfg['maximum'])
                    if pending:
                        return pending
                    if not found:
                        break
            elif phase == 'training_initial':
                cfg = self.c[phase]
                while self.total(phase) < cfg['minimum']:
                    found, pending = propose(phase, None, 1, cfg['conversation_maximum'], cfg['maximum'])
                    if pending:
                        return pending
                    if not found:
                        break
            else:
                cfg = self.c[phase]
                remaining = (self.c['training_development_combined_maximum']
                             - self.total('development') - self.total('training_initial'))
                if not self.state['selected'][phase]:
                    _, pending = propose(phase, None, cfg['conversation_minimum'], cfg['conversation_maximum'], remaining)
                    if pending:
                        return pending
            pending = [self.pool[cid] for cid in self.state['selected'][phase]
                       if reviews.get(cid, {}).get('status') not in ACCEPTED]
            if pending:
                return pending
            if not self.state['selected'][phase]:
                raise ValueError(f'No source-reviewed material can be selected for {phase}')
            if phase == 'evaluation':
                self.state['first_batch'] = sorted(first_batch(
                    [self.pool[c] for c in self.state['selected'][phase]], self.c['seed'],
                    self.c['first_batch']['maximum']))
            for cid in self.state['selected'][phase]:
                self.state['locked_review_sha256'][cid] = canonical_sha(reviews[cid])
            self.state['locked'].append(phase)
            self.event('lock', phase=phase, interactions=self.total(phase),
                       conversations=len(self.state['selected'][phase]))
        self.state['completed'] = True
        return []


def advance_known(engine, reviews):
    """Consume already-reviewed holds without asking for a second review of them.

Stop at the first unknown source, before its tentative student or interaction
count can exclude a later candidate. Partial saved rounds are never applied early.
"""
    while True:
        before = canonical_sha(engine.state)
        pending = engine.advance(reviews)
        if not pending or any(reviews.get(p['cid'], {}).get('status') not in STATUSES for p in pending):
            return pending
        if before == canonical_sha(engine.state):
            raise ValueError('Allocation did not progress through known reviews')


def split_annotations(text, annotations):
    pieces, end = [], 0
    for ann in sorted(annotations, key=lambda a: a['start']):
        pieces.append(text[end:ann['start']])
        end = ann['end']
    pieces.append(text[end:])
    return ''.join(pieces)


SHARE_URL = re.compile(r'https?://(?:chatgpt\.com/share/|chat\.openai\.com/share/|'
                       r'claude\.ai/share/|gemini\.google\.com/share/|g\.co/gemini/share/)'
                       r'[^\s<>"\']+', re.IGNORECASE)


def default_deidentify(text, context):
    """Minimal shared-link removal only. This does NOT certify PII clearance."""
    replacements = [{'start': m.start(), 'end': m.end(), 'text': m.group(),
                     'replacement': '[共享链接已隐藏]', 'kind': 'shared_url',
                     'basis': 'Candidate reading materials do not expose shared chat links'}
                    for m in SHARE_URL.finditer(text)]
    return {'text': apply_replacements(text, replacements), 'replacements': replacements}


def apply_replacements(text, replacements):
    result, end = [], 0
    for item in sorted(replacements, key=lambda r: r['start']):
        start, stop = item.get('start'), item.get('end')
        if (type(start) is not int or type(stop) is not int or not end <= start < stop <= len(text)
                or item.get('text') != text[start:stop]
                or not isinstance(item.get('replacement'), str)
                or not isinstance(item.get('kind'), str) or not item['kind']
                or not isinstance(item.get('basis'), str) or not item['basis']):
            raise ValueError('Invalid or overlapping deidentification replacement')
        result.extend([text[end:start], item['replacement']])
        end = stop
    result.append(text[end:])
    return ''.join(result)


def derive_record(row, receipt, deidentify=None):
    """A deidentifier returns {text, replacements}; offsets refer to cleaned input.

Replacements are exact [start,end) Python Unicode character spans with text,
replacement, kind and basis. The trace stays private. The callback's presence
alone never certifies a packet for external release.
"""
    order = int(row['Segment Order'])
    cid = row['Conversation ID']
    output = {'conversation_id': cid, 'segment_order': order}
    traces, notes = [], []
    callback = deidentify or default_deidentify
    for field, name in [('Student Prompt', 'student_prompt'), ('LLM Response', 'ai_response')]:
        annotations = [a for a in receipt['annotations'] if a['segment_order'] == order and a['field'] == field]
        clean = split_annotations(row[field], annotations)
        context = {'conversation_id': cid, 'segment_order': order, 'field': field, 'kind': 'dialogue'}
        result = callback(clean, context)
        if (not isinstance(result, dict) or not isinstance(result.get('replacements'), list)
                or result.get('text') != apply_replacements(clean, result['replacements'])):
            raise ValueError('Deidentifier text does not match its trace')
        if SHARE_URL.search(result['text']):
            raise ValueError('Deidentifier left a shared-chat link in candidate material')
        output[name] = result['text']
        traces.append({**context, 'raw_field_sha256': text_sha(row[field]),
                       'clean_field_sha256': text_sha(clean), 'derived_field_sha256': text_sha(result['text']),
                       'annotations': annotations, 'replacements': result['replacements']})
        for ann in annotations:
            note_context = {**context, 'kind': 'non_dialogue_note', 'annotation_start': ann['start']}
            note = callback(ann['text'], note_context)
            if (not isinstance(note, dict) or not isinstance(note.get('replacements'), list)
                    or note.get('text') != apply_replacements(ann['text'], note['replacements'])
                    or SHARE_URL.search(note['text'])):
                raise ValueError('Invalid note deidentification')
            notes.append({'field': field, 'kind': ann['kind'], 'text': note['text'],
                          'instruction': '不属于实际问答，不计入任何四轴，也不作为对话上下文。'})
            traces.append({**note_context, 'raw_field_sha256': text_sha(ann['text']),
                           'replacements': note['replacements']})
    output['non_dialogue_notes'] = notes
    output['availability'] = {'truncated_capture': order in receipt['truncated_orders'],
                              'ai_fields_unavailable': order in receipt['truncated_orders']}
    output['source_review_status'] = receipt['status']
    return output, traces


def export_materials(engine, reviews, *, root=ROOT, task=TASK, out=OUT, deidentify=None):
    # Perform every gate and derive everything in memory before creating files.
    state = engine.state
    if not state['completed'] or set(state['locked']) != set(PHASES):
        raise ValueError('Cannot export an incomplete/unlocked allocation')
    engine.advance(reviews)  # Also catches changed annotations on a locked pass.
    lock_hash = validate_original_lock(engine.c, reviews, root, task)
    rows_by_c = load_rows(root)
    validate_reviews(reviews, list(engine.pool.values()), rows_by_c, engine.c, root)
    if state.get('original_review_lock_sha256') != lock_hash:
        raise ValueError('State does not belong to the locked original review set')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output is not empty; preserve existing reading/coding work')
    first = set(state['first_batch'])
    groups = {'evaluation_first': [c for c in state['selected']['evaluation'] if c in first],
              'evaluation_optional': [c for c in state['selected']['evaluation'] if c not in first],
              **{phase: state['selected'][phase] for phase in PHASES[1:]}}
    packed, raws, traces, summaries, roster = {}, {}, [], [], []
    all_students = set()
    for group, cids in groups.items():
        packed[group], raws[group] = [], []
        for cid in cids:
            p = engine.pool[cid]
            if p['student'] in all_students:
                raise ValueError('Student occurs in multiple selected conversations')
            all_students.add(p['student'])
            roster.append({'group': group, **{k: p[k] for k in ('cid', 'student', 'n', 'stratum', 'assignment')},
                           'source_review_status': reviews[cid]['status']})
            for row in rows_by_c[cid]:
                record, trace = derive_record(row, reviews[cid], deidentify)
                packed[group].append({'group': group, 'student_id': p['student'], **record})
                raws[group].append({'group': group, 'student_id': p['student'],
                                    'conversation_id': cid, 'segment_order': int(row['Segment Order']),
                                    'student_prompt': row['Student Prompt'], 'ai_response': row['LLM Response'],
                                    'source_file': row.get('Source File', '')})
                traces.extend(trace)
        summaries.append({'group': group, 'interactions': len(packed[group]), 'conversations': len(cids),
                          'students': len(cids), 'strata': {s: sum(engine.pool[c]['n'] for c in cids
                                                                 if engine.pool[c]['stratum'] == s) for s in STRATA},
                          'assignments': {a: {'conversations': sum(engine.pool[c]['assignment'] == a for c in cids),
                                              'interactions': sum(engine.pool[c]['n'] for c in cids if engine.pool[c]['assignment'] == a)}
                                          for a in sorted({engine.pool[c]['assignment'] for c in cids})}})
    counts = {g: len(rows) for g, rows in packed.items()}
    ev = counts['evaluation_first'] + counts['evaluation_optional']
    train = sum(counts[g] for g in PHASES[1:])
    if (ev > engine.c['evaluation']['maximum'] or counts['evaluation_first'] > engine.c['first_batch']['maximum']
            or train > engine.c['training_development_combined_maximum']):
        raise ValueError('Export exceeds registered budget')
    result = {'protocol': engine.c['protocol'], 'status': 'source_reviewed_selection_unlabelled',
              'seed': engine.c['seed'], 'rules_version': VERSION, 'rules_sha256': engine.c['rules_sha256'],
              'original_review_lock_sha256': lock_hash, 'groups': summaries, 'evaluation_total': ev,
              'development_training_total': train,
              'remaining_training_budget': engine.c['training_development_combined_maximum'] - train,
              'held_conversations': state['held_conversations'], 'human_labels_created': False,
              'coding_packet_released': False, 'deidentification_callback_used': deidentify is not None,
              'deidentification_clearance': 'Separate verification required; this export does not certify external release',
              'shortfalls': {'evaluation': max(0, engine.c['evaluation']['minimum'] - ev),
                            'first_batch': max(0, engine.c['first_batch']['minimum_goal'] - counts['evaluation_first']),
                            'development_students': max(0, engine.c['development']['minimum_students'] - len(groups['development'])),
                            'development_interactions': max(0, engine.c['development']['minimum'] - counts['development']),
                            'training_initial_interactions': max(0, engine.c['training_initial']['minimum'] - counts['training_initial']),
                            **{f'evaluation_stratum_{s}': max(0, engine.c['evaluation']['lower_per_stratum'] - engine.total('evaluation', s)) for s in STRATA}}}
    out.mkdir(parents=True, exist_ok=True)
    for group, records in packed.items():
        folder = out / group
        folder.mkdir()
        write_json(folder / 'manifest.json', {'protocol': engine.c['protocol'], 'schema_version': engine.c['schema_version'],
                   'phase': 'selected_unlabelled', 'group': group, 'target_count': len(records),
                   'conversation_count': len(groups[group]), 'student_count': len(groups[group]),
                   'targets': [{k: r[k] for k in ('conversation_id', 'segment_order')} for r in records],
                   'students': sorted({r['student_id'] for r in records}), 'conversation_ids': groups[group],
                   'coding_packet_released': False})
        (folder / 'interactions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        private = out / 'private' / group
        private.mkdir(parents=True)
        (private / 'raw-interactions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in raws[group]), encoding='utf-8')
    write_json(out / 'private/source-checks.json', {c: reviews[c] for ids in groups.values() for c in ids})
    write_json(out / 'private/text-derivation.json', traces)
    write_json(out / 'selection-summary.json', result)
    write_json(out / 'conversation-roster.json', roster)
    with (out / 'conversation-roster.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['group', 'cid', 'student', 'n', 'stratum', 'assignment', 'source_review_status']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(roster)
    render_readers(out, packed, summaries, result)
    return result


def render_readers(out, packed, summaries, result):
    titles = {'evaluation_first': '评估首批｜第二编码员 最多150条', 'evaluation_optional': '评估后续｜第二编码员 可选补标',
              'development': '开发材料｜DW', 'training_initial': '训练首批｜DW', 'training_long': '训练长对话｜DW'}
    style = ('body{font:16px/1.65 system-ui;max-width:1100px;margin:40px auto;padding:0 24px;color:#203048}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7f8;padding:16px}'
             '.note,aside{background:#fff4cf;padding:14px;margin:15px 0}td,th{padding:10px;border-bottom:1px solid #ddd}')
    disclaimer = ('<p class="note">未编码的候选阅读材料，尚未发布为正式编码包。评估编码须等待校准后稳定手册，'
                  '对外发送前仍须完成去标识核验。下方单列的非问答说明不计入任何四轴或上下文。'
                  '来源核查和截断提示均不是人工标签。</p>')
    def save(path, title, body):
        path.write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" '
                        'content="width=device-width,initial-scale=1"><title>' + html.escape(title)
                        + '</title><style>' + style + '</style><body>' + body + '</body></html>', encoding='utf-8')
    (out / 'conversations').mkdir()
    for group, records in packed.items():
        by_c = collections.defaultdict(list)
        for row in records:
            by_c[row['conversation_id']].append(row)
        links = []
        for cid, rows in by_c.items():
            links.append(f'<li><a href="../conversations/{html.escape(cid)}.html">{html.escape(cid)}</a> · {len(rows)}条</li>')
            body = f'<p><a href="../{group}/index.html">返回清单</a></p><h1>{html.escape(cid)}</h1>' + disclaimer
            for row in rows:
                body += f'<article><h2>#{row["segment_order"]}</h2><h3>Student</h3><pre>{html.escape(row["student_prompt"])}</pre>'
                body += '<h3>AI response</h3><pre>' + html.escape(row['ai_response']) + '</pre>'
                if row['availability']['truncated_capture']:
                    body += '<p class="note">D32：回复不完整。后续人工记录的两个AI维度留空并标记 truncated capture；学生维度依证据判断。本提示未生成标签。</p>'
                for note in row['non_dialogue_notes']:
                    body += '<aside><strong>' + html.escape(note['kind']) + ' 非问答说明，不计入四轴或上下文</strong><pre>' + html.escape(note['text']) + '</pre></aside>'
                body += '</article>'
            save(out / 'conversations' / f'{cid}.html', cid, body)
        save(out / group / 'index.html', titles[group], '<a href="../START-HERE.html">返回总览</a><h1>' + titles[group] + '</h1>' + disclaimer + '<ul>' + ''.join(links) + '</ul>')
    table = ''.join(f'<tr><td><a href="{r["group"]}/index.html">{titles[r["group"]]}</a></td><td>{r["interactions"]}</td><td>{r["students"]}</td></tr>' for r in summaries)
    body = '<h1>10月6日材料分配</h1>' + disclaimer + '<table><tr><th>材料</th><th>互动</th><th>学生／对话</th></tr>' + table + '</table>'
    body += f'<p>评估共{result["evaluation_total"]}条；训练与开发共{result["development_training_total"]}条，预算余{result["remaining_training_budget"]}条。首批之外是否双人编码不作完成承诺。</p>'
    if any(result['shortfalls'].values()):
        body += '<p class="note">数量目标存在缺口，见 selection-summary.json；没有更换种子或放宽配额。</p>'
    save(out / 'START-HERE.html', '10月6日材料分配', body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--advance', action='store_true', help='Continue fixed queue after complete review rounds')
    parser.add_argument('--export', action='store_true', help='Export completed allocation to the new empty output directory')
    args = parser.parse_args()
    if not (args.advance or args.export):
        parser.error('Specify --advance and/or --export')
    config = read_json(TASK / 'selection-config.json')
    reviews = read_json(TASK / 'source-reviews.json')
    lock_hash = validate_original_lock(config, reviews)
    pool, excluded, hashes = build_pool()
    hashes[config['rules_path']] = sha(ROOT / config['rules_path'])
    if sha(ROOT / config['decision_path']) != config['decision_sha256']:
        raise ValueError('Registered decision changed; resolve explicitly before allocation')
    validate_reviews(reviews, pool, load_rows(), config)
    state_path = TASK / 'selection-state.json'
    state = read_json(state_path) if state_path.exists() else None
    contract_sha = sha(TASK / 'selection-config.json')
    if state is not None:
        if (state.get('config_sha256') != contract_sha or state.get('input_hashes') != hashes
                or state.get('original_review_lock_sha256') != lock_hash):
            raise ValueError('Allocation inputs/contract/lock changed; do not continue silently')
    elif not args.advance:
        raise ValueError('No v2 allocation state exists')
    engine = Engine(config, pool, state)
    engine.state.update(config_sha256=contract_sha, input_hashes=hashes,
                        original_review_lock_sha256=lock_hash)
    if args.advance:
        if state is None:
            write_json(TASK / 'pool.json', pool)
            write_json(TASK / 'pool-exclusions.json', excluded)
            write_json(TASK / 'evaluation-order.json', {s: sorted([p['cid'] for p in pool if p['stratum'] == s],
                        key=lambda c: rank(config['seed'], 'evaluation', c)) for s in STRATA})
        pending = advance_known(engine, reviews)
        write_json(state_path, engine.state)
        write_json(TASK / 'pending-source-review.json', pending)
        print(json.dumps({'phase': engine.state['phase'], 'completed': engine.state['completed'],
                          'pending_conversations': len(pending), 'pending_interactions': sum(p['n'] for p in pending),
                          'totals': {p: engine.total(p) for p in PHASES}}, ensure_ascii=False))
    if args.export:
        result = export_materials(engine, reviews)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
