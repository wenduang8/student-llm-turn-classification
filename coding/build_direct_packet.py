"""Build a blank direct-coding packet for one group of the D-100 allocation (allocation-20261010).

    python coding/build_direct_packet.py --group evaluation_first --output <new dir> --dry-run [--deidentify <file>]
    python coding/build_direct_packet.py --group evaluation_first --output data/coding/<dir> \
        --guide docs/codebook/<stable guide>.md --guide-version <version> --decision D-1xx --deidentify <file>

Every gate runs in memory before any file is written:
  1. the original 298-review lock (D-87) validates unchanged;
  2. the D-100 ledger changes exactly its four listed receipts, with matching before/after hashes;
  3. a fresh replay of the D-87 allocation from the effective receipts reproduces the saved D-100
     state and the allocation-20261010 manifests;
  4. the analysis corpus and codebook schema match the D-100 baseline hashes, and every listed
     conversation is complete, accepted and from a distinct student.
The text shown is derived, not edited: separable non-dialogue spans (M1/M2) are moved out of the
dialogue and listed apart, and shared-chat links are hidden. No labels are prefilled. A D-32
truncated response carries a locked 'truncated capture' flag with both AI fields blank.
Identifiers are replaced only by the explicit, conversation-scoped rules of a --deidentify file kept
under data/. A pattern screen for contact details and user names is written privately; a formal build
refuses while any candidate is neither replaced nor listed as reviewed, or while the file is not marked
adopted. This does not certify de-identification. Output directories are never reused. Nothing is sent.
"""
from __future__ import annotations
import argparse, datetime, hashlib, html, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / 'coding'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import select_materials_v2 as v
from codebook_schema import DIRECT_CODES, DIRECT_FIELDS, DIRECT_VERSION, DIRECT_DEFAULT
from direct_coding import validate

OLD = ROOT / 'data/maintenance/20261006-source-recheck'
D100 = ROOT / 'data/maintenance/20261010-source-adoptions'
ALLOC = ROOT / 'data/coding/direct-20260924/allocation-20261010'
GROUPS = ('evaluation_first', 'evaluation_optional', 'development', 'training_initial', 'training_long')
D100_CIDS = {'P141_P2_04', 'P280_P2_01', 'P330_P2_04', 'P035_P2_06'}
STAGE = 'independent'
PROVISIONAL_GUIDE = ('docs/codebook/guide-direct-20260924.md', 'docs/codebook/calibration-clarifications-20261009.md')
TITLES = {'evaluation_first': '150-interaction independent coding batch', 'evaluation_optional': 'optional evaluation batch',
          'development': 'development material', 'training_initial': 'initial training material',
          'training_long': 'long-conversation training material'}
DEFAULT_CODERS = {'evaluation_first': ['DW', 'second-coder'], 'evaluation_optional': ['DW', 'second-coder'],
                  'development': ['DW'], 'training_initial': ['DW'], 'training_long': ['DW']}
AI_AXES = [a for a in DIRECT_CODES if a.startswith('AI ')]
TRUNCATED_FLAG = 'truncated capture'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()


# ---------------------------------------------------------------- gates

def verified_allocation(root=ROOT):
    """Return effective receipts, rows, groups, pool and evidence; raise on any mismatch."""
    config = v.read_json(OLD / 'selection-config.json')
    base = v.read_json(OLD / 'source-reviews.json')
    lock_hash = v.validate_original_lock(config, base, root, OLD)
    pool, exclusions, _ = v.build_pool()
    if (v.canonical_sha(pool) != v.canonical_sha(v.read_json(OLD / 'pool.json'))
            or v.canonical_sha(exclusions) != v.canonical_sha(v.read_json(OLD / 'pool-exclusions.json'))):
        raise ValueError('Candidate pool or exclusions differ from the D-87 record')
    rows = v.load_rows(root)
    v.validate_reviews(base, pool, rows, config, root)
    ledger = v.read_json(D100 / 'source-adjudications.json')
    effective = v.read_json(D100 / 'effective-source-reviews.json')
    if (ledger.get('decision') != 'D-100' or ledger.get('baseline_lock_sha256') != lock_hash
            or ledger.get('baseline_reviews_sha256') != sha(OLD / 'source-reviews.json')
            or set(ledger.get('overrides', {})) != D100_CIDS):
        raise ValueError('D-100 ledger does not match the original lock and its four cases')
    if set(effective) != set(base):
        raise ValueError('Effective receipts cover a different conversation set')
    changed = {c for c in base if v.canonical_sha(base[c]) != v.canonical_sha(effective[c])}
    if changed != D100_CIDS:
        raise ValueError(f'Effective receipts differ outside the D-100 cases: {sorted(changed ^ D100_CIDS)}')
    for cid, o in ledger['overrides'].items():
        if (o.get('base_receipt_sha256') != v.canonical_sha(base[cid])
                or o.get('effective_receipt_sha256') != v.canonical_sha(effective[cid])
                or o.get('effective_status') != effective[cid]['status']):
            raise ValueError(f'D-100 ledger hash or status mismatch: {cid}')
    v.validate_reviews(effective, pool, rows, config, root)
    engine = v.Engine(config, pool, state=None)
    if v.advance_known(engine, effective) or not engine.state['completed']:
        raise ValueError('Replay did not complete from the effective receipts')
    engine.advance(effective)
    saved = v.read_json(D100 / 'selection-state.json')
    for key in ('selected', 'first_batch', 'held_conversations', 'locked', 'locked_review_sha256'):
        if v.canonical_sha(engine.state[key]) != v.canonical_sha(saved[key]):
            raise ValueError(f'Replay differs from the saved D-100 state: {key}')
    first = set(engine.state['first_batch'])
    groups = {'evaluation_first': [c for c in engine.state['selected']['evaluation'] if c in first],
              'evaluation_optional': [c for c in engine.state['selected']['evaluation'] if c not in first],
              **{p: engine.state['selected'][p] for p in v.PHASES[1:]}}
    baseline = v.read_json(D100 / 'verification.json')['protected_files_unchanged']
    for path in ('data/processed/transcripts_analysis.csv', 'codebook_schema.json'):
        if sha(root / path) != baseline[path]:
            raise ValueError(f'{path} changed since the D-100 allocation')
    students = set()
    for g, cids in groups.items():
        m = v.read_json(ALLOC / g / 'manifest.json')
        targets = [(c, int(r['Segment Order'])) for c in cids for r in rows[c]]
        if (m.get('group') != g or m.get('conversation_ids') != cids or m.get('target_count') != len(targets)
                or [(t['conversation_id'], t['segment_order']) for t in m['targets']] != targets):
            raise ValueError(f'allocation-20261010 manifest differs from the replay: {g}')
        for c in cids:
            if effective[c]['status'] not in v.ACCEPTED or len(rows[c]) != engine.pool[c]['n']:
                raise ValueError(f'Conversation not accepted or incomplete: {c}')
            if engine.pool[c]['student'] in students:
                raise ValueError(f'Student appears in two selected conversations: {c}')
            students.add(engine.pool[c]['student'])
    evidence = {'original_review_lock_sha256': lock_hash,
                'd100_ledger_sha256': sha(D100 / 'source-adjudications.json'),
                'effective_receipts_sha256': sha(D100 / 'effective-source-reviews.json'),
                'd100_state_sha256': sha(D100 / 'selection-state.json'),
                'allocation_manifest_sha256': {g: sha(ALLOC / g / 'manifest.json') for g in GROUPS},
                'transcripts_analysis_sha256': sha(root / 'data/processed/transcripts_analysis.csv'),
                'codebook_schema_sha256': sha(root / 'codebook_schema.json'),
                'distinct_students_all_groups': len(students),
                'gates': ['original 298 lock', 'D-100 four-case ledger', 'replay equals saved D-100 state',
                          'allocation-20261010 manifests equal replay', 'corpus and schema equal D-100 baseline',
                          'complete accepted conversations, one per student']}
    return effective, rows, groups, engine.pool, evidence


def provisional_guide():
    return {'status': 'provisional (dry run only)',
            'paths': {p: sha(ROOT / p) for p in PROVISIONAL_GUIDE},
            'note': 'The stable post-calibration guide does not exist yet. A formal packet records it here.'}


def stable_guide(path, version, decision):
    path = Path(path).resolve()
    if not path.is_file() or not path.is_relative_to(ROOT.resolve()):
        raise ValueError('The guide must be a file inside the repository')
    register = (ROOT / 'docs/decisions.md').read_text(encoding='utf-8-sig')
    if not re.search(rf'^### {re.escape(decision)} ', register, re.M):
        raise ValueError(f'{decision} is not recorded in docs/decisions.md')
    return {'status': 'stable', 'path': rel(path), 'sha256': sha(path), 'version': version, 'decision': decision,
            'portable_markdown': 'guide.md', 'portable_html': 'guide.html'}


def render_guide(text, version):
    """Render the guide's headings, paragraphs, lists and tables without external assets.

    The verbatim Markdown companion remains authoritative. Escape text before adding the
    few supported formatting elements; source HTML and Markdown links are never executed.
    """
    def inline(value):
        value = html.escape(value)
        value = re.sub(r'`([^`]+)`', r'<code>\1</code>', value)
        return re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)

    lines, blocks, i = text.splitlines(), [], 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i])
                i += 1
            blocks.append('<pre>' + html.escape('\n'.join(code)) + '</pre>')
            i += 1
            continue
        heading = re.match(r'^(#{1,6})\s+(.+)$', line)
        if heading:
            level = len(heading[1])
            blocks.append(f'<h{level}>{inline(heading[2])}</h{level}>')
            i += 1
            continue
        if (line.startswith('|') and i + 1 < len(lines)
                and re.fullmatch(r'[\s|:\-]+', lines[i + 1]) and '-' in lines[i + 1]):
            cells = lambda value: [inline(x.strip()) for x in value.strip().strip('|').split('|')]
            blocks.append('<table><thead><tr>' + ''.join('<th>' + c + '</th>' for c in cells(line)) + '</tr></thead><tbody>')
            i += 2
            while i < len(lines) and lines[i].strip().startswith('|'):
                blocks.append('<tr>' + ''.join('<td>' + c + '</td>' for c in cells(lines[i])) + '</tr>')
                i += 1
            blocks.append('</tbody></table>')
            continue
        list_item = re.match(r'^(?:[-*]|\d+\.)\s+(.+)$', line)
        if list_item:
            tag = 'ol' if re.match(r'^\d+\.', line) else 'ul'
            blocks.append(f'<{tag}>')
            while i < len(lines):
                item = re.match(r'^(?:[-*]|\d+\.)\s+(.+)$', lines[i].strip())
                if not item:
                    break
                blocks.append('<li>' + inline(item[1]) + '</li>')
                i += 1
            blocks.append(f'</{tag}>')
            continue
        if re.fullmatch(r'-{3,}', line):
            blocks.append('<hr>')
            i += 1
            continue
        paragraph = [line]
        i += 1
        while i < len(lines) and lines[i].strip():
            if re.match(r'^(?:#{1,6}\s|[-*]\s|\d+\.\s|\||```)', lines[i].strip()):
                break
            paragraph.append(lines[i].strip())
            i += 1
        blocks.append('<p>' + inline(' '.join(paragraph)) + '</p>')
    css = ('body{font:17px/1.6 Arial,sans-serif;max-width:1000px;margin:30px auto;padding:0 24px;color:#182432}'
           'header{background:#f0f4f8;padding:15px}table{border-collapse:collapse;width:100%;margin:20px 0}'
           'th,td{border:1px solid #a8b6c7;padding:10px;text-align:left;vertical-align:top}'
           'th{background:#edf2f7}h2{margin-top:34px}code,pre{font-family:Consolas,monospace}'
           'pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f6f7f9;padding:15px}')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Coding guide {html.escape(version)}</title><style>{css}</style><body>'
            '<header><a href="coding.html">Open the coding form</a> · <a href="guide.md">Verbatim guide text</a></header>'
            + ''.join(blocks) + '</body></html>')


def check_output(out, formal):
    out, root = Path(out).resolve(), ROOT.resolve()
    if out.exists():
        raise FileExistsError('Output exists; packets and returned records are never overwritten. Choose a new directory.')
    if formal and not out.is_relative_to(root / 'data/coding'):
        raise ValueError('A formal packet must be written under data/coding/')
    if out.is_relative_to(root) and not out.is_relative_to(root / 'data'):
        raise ValueError('Student text must stay under data/ inside the repository')
    return out


# ---------------------------------------------------------------- derivation and screen

def load_deidentification(path):
    """Explicit, conversation-scoped replacements plus screen tokens reviewed as harmless.

    File (kept under data/, never committed): {"status": "proposed"|"adopted", "rules": [{"conversation_id",
    "pattern", "replacement", "kind", "basis"}], "reviewed_benign": [{"kind", "token", "basis"}]}.
    Patterns are case-insensitive regular expressions, tried in the listed order at each position.
    """
    if path is None:
        return {'status': 'none', 'rules': [], 'reviewed_benign': []}, None
    path = Path(path).resolve()
    if path.is_relative_to(ROOT.resolve()) and not path.is_relative_to((ROOT / 'data').resolve()):
        raise ValueError('The de-identification file names real identifiers; keep it under data/')
    spec = v.read_json(path)
    if spec.get('status') not in ('proposed', 'adopted'):
        raise ValueError('De-identification status must be proposed or adopted')
    for r in spec.get('rules', []):
        if not all(isinstance(r.get(k), str) and r[k] for k in ('conversation_id', 'pattern', 'replacement', 'kind', 'basis')):
            raise ValueError('Each de-identification rule needs conversation_id, pattern, replacement, kind and basis')
        re.compile(r['pattern'], re.I)
    for t in spec.get('reviewed_benign', []):
        if t.get('kind') not in SCREENS or not isinstance(t.get('token'), str) or not t.get('basis'):
            raise ValueError('Each reviewed token needs a screen kind, the token and a basis')
    return spec, {'path': rel(path) if path.is_relative_to(ROOT.resolve()) else str(path), 'sha256': sha(path),
                  'status': spec['status'], 'rules': len(spec.get('rules', [])),
                  'reviewed_benign_tokens': len(spec.get('reviewed_benign', []))}


def make_deidentifier(spec):
    """Shared-link removal (as before) plus the explicit conversation-scoped rules, without overlaps."""
    by_c = {}
    for r in spec.get('rules', []):
        by_c.setdefault(r['conversation_id'], []).append(r)

    def deidentify(text, context):
        found = [{'start': m.start(), 'end': m.end(), 'text': m.group(), 'replacement': '[shared chat link removed]',
                  'kind': 'shared_url', 'basis': 'Coding materials do not expose shared chat links'}
                 for m in v.SHARE_URL.finditer(text)]
        rules = by_c.get(context['conversation_id'], [])
        if rules:
            combined = re.compile('|'.join(f'(?P<r{i}>{r["pattern"]})' for i, r in enumerate(rules)), re.I)
            for m in combined.finditer(text):
                if m.end() == m.start():
                    continue
                r = rules[int(m.lastgroup[1:])]
                found.append({'start': m.start(), 'end': m.end(), 'text': m.group(), 'replacement': r['replacement'],
                              'kind': r['kind'], 'basis': r['basis']})
        kept, end = [], 0
        for item in sorted(found, key=lambda x: (x['start'], -x['end'])):
            if item['start'] >= end:
                kept.append(item)
                end = item['end']
        return {'text': v.apply_replacements(text, kept), 'replacements': kept}
    return deidentify


def derive_group(cids, rows, effective, pool, deidentify=None):
    records, traces = [], []
    for cid in cids:
        for row in rows[cid]:
            rec, trace = v.derive_record(row, effective[cid], deidentify)
            records.append({'conversation_id': cid, 'segment_order': rec['segment_order'],
                            'student_id': pool[cid]['student'], 'student_prompt': rec['student_prompt'],
                            'ai_response': rec['ai_response'], 'notes': rec['non_dialogue_notes'],
                            'truncated': rec['availability']['truncated_capture']})
            traces.extend(trace)
    return records, traces


SCREENS = {
    'email address': re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}'),
    'unikey-like token': re.compile(r'\b[a-z]{4}\d{4}\b'),
    'nine-digit number': re.compile(r'(?<![\d.])\d{9}(?![\d.])'),
    'Australian mobile number': re.compile(r'(?<!\d)(?:\+?61\s?|0)4\d{2}\s?\d{3}\s?\d{3}(?!\d)'),
    'user folder name': re.compile(r'(?:[A-Za-z]:\\+Users\\+|/Users/|/home/)([^\\/\s"\'<>:]+)'),
    'shell prompt user': re.compile(r'(?m)^\s*([A-Za-z0-9._-]+)@[A-Za-z0-9._-]+[ :][^\n]{0,60}?[%$#] '),
    'SSH greeting name': re.compile(r"Hi ([^!\s]+)! You've successfully authenticated"),
    'file owner in listing': re.compile(r'(?m)^[-dlcbps][-rwxsStT@+.]{9,10}\s+\d+\s+([A-Za-z0-9._-]+)\s'),
    'other web address': re.compile(r'https?://([^/\s<>"\']+)'),
}


def mask(s):
    return s[:2] + '*' * (len(s) - 2) if len(s) > 2 else '*' * len(s)


def screen(records, spec=None):
    """Pattern screen only. Candidates need a check; a screen is not a measurement.

    A candidate is 'replaced' when it is a replacement marker, 'reviewed' when its exact token is listed as
    harmless in the de-identification file, and otherwise 'unresolved'. Names in free text are not screened.
    """
    spec = spec or {'rules': [], 'reviewed_benign': []}
    markers = {r['replacement'] for r in spec.get('rules', [])}
    benign = {(t['kind'], t['token']) for t in spec.get('reviewed_benign', [])}
    report, unresolved_total = {}, 0
    for kind, pattern in SCREENS.items():
        hits = []
        for r in records:
            fields = [('student_prompt', r['student_prompt']), ('ai_response', r['ai_response'])]
            fields += [(f'note:{n["kind"]}:{n["field"]}', n['text']) for n in r['notes']]
            for field, text in fields:
                for m in pattern.finditer(text):
                    shown = m.group(1) if m.groups() else m.group()
                    entry = {'conversation_id': r['conversation_id'], 'segment_order': r['segment_order'],
                             'field': field, 'start': m.start()}
                    if kind == 'other web address':
                        entry['domain'] = shown.lower()
                        entry['resolution'] = 'web address (reported, not gated)'
                    else:
                        entry['masked'] = mask(shown)
                        entry['resolution'] = ('replaced' if shown in markers or any(mk in shown for mk in markers)
                                               else 'reviewed' if (kind, shown) in benign else 'unresolved')
                        unresolved_total += entry['resolution'] == 'unresolved'
                    hits.append(entry)
        report[kind] = {'candidates': len(hits),
                        'interactions': len({(h['conversation_id'], h['segment_order']) for h in hits}),
                        'unresolved': sum(h['resolution'] == 'unresolved' for h in hits),
                        'entries': hits}
    report['unresolved_total'] = unresolved_total
    report['scope'] = ('Regex screen of the text shown to coders, after any listed replacements. Candidates are not '
                       'confirmed identifiers. Reviewed tokens are as listed in the de-identification file; names written '
                       'in free text are not screened. This program changes no text beyond the listed rules.')
    return report


# ---------------------------------------------------------------- rendering

STYLE = ('body{font:17px/1.5 Arial,sans-serif;max-width:1080px;margin:28px auto;padding:0 24px;color:#182432;background:#fff}'
         'header{background:#f0f4f8;padding:20px}.dry{background:#ffe1e1;border:2px solid #b3261e;padding:12px;font-weight:bold}'
         'section.conversation{border-top:3px solid #4b5d73;margin-top:40px}article{border-top:1px solid #b9c3cf;margin-top:26px;padding-top:8px}'
         'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.45 Consolas,monospace;background:#f6f7f9;padding:15px}'
         'fieldset{margin:12px 0;border:1px solid #a8b6c7}fieldset:disabled{opacity:.55}legend{font-weight:bold}label{display:block;margin:8px}'
         'label small{display:block;margin-left:29px;color:#465260;font-size:15px}textarea{display:block;width:94%;min-height:65px;font:inherit}'
         'button{font:inherit;padding:9px 20px}nav{line-height:2}input[type=radio],input[type=checkbox]{width:18px;height:18px;vertical-align:middle}'
         '.capture{background:#fff7dc;padding:10px 14px;margin:12px 0}.source{background:#e8f1fb;padding:10px 14px}.gap{background:#eee;padding:8px 14px;font-style:italic}'
         'article .state::before{content:"not complete";color:#8a5300;font-size:15px;margin-left:12px}'
         'article.done .state::before{content:"complete";color:#1d6b35}#progress{font-weight:bold}')

JS = r'''const config=CONFIG;
let unsavedChanges=false;
const targets=[...document.querySelectorAll('article[data-target="true"]')];
const byKey=new Map(targets.map(el=>[el.dataset.cid+'#'+el.dataset.turn,el]));
const draftPrefix='honours-coding-draft:v1:'+config.packetFingerprint;
const activeKey=draftPrefix+':active';
const writer=Date.now().toString(36)+'-'+Math.random().toString(36).slice(2);
let activeCoder='',expectedRaw=null,savingBlocked='',activeMetadataBlocked=false;
function draftKey(coder){return draftPrefix+':coder:'+encodeURIComponent(coder);}
function saveStatus(state,message){const node=document.getElementById('autosave-status');node.dataset.state=state;node.textContent=message;}
function displayTime(value){return new Date(value).toLocaleString();}
function lockCoding(locked){for(const el of targets)for(const input of el.querySelectorAll('.coding input,.coding textarea,.coding button'))input.disabled=locked||input.dataset.fixedDisabled==='true';}
for(const el of targets)for(const input of el.querySelectorAll('.coding input,.coding textarea,.coding button'))if(input.disabled)input.dataset.fixedDisabled='true';
function pauseSaving(reason,message){savingBlocked=reason;saveStatus(reason==='conflict'?'conflict':'error',message);}
function axisInputs(el,a){return [...el.querySelectorAll('input[data-axis]')].filter(x=>x.dataset.axis===a);}
function unavailableText(el,a){const b=[...el.querySelectorAll('[data-unavailable-axis]')].find(x=>x.dataset.unavailableAxis===a);return b?b.value.trim():'';}
function isComplete(el){
 const truncated=el.dataset.sourceTruncated==='true'||el.querySelector('[data-truncated]').checked;
 return Object.keys(config.axes).every(a=>(truncated&&a.startsWith('AI '))||axisInputs(el,a).some(x=>x.checked)||unavailableText(el,a)!=='');
}
function updateProgress(){
 let done=0;const per=new Map();
 for(const el of targets){const ok=isComplete(el);el.classList.toggle('done',ok);if(ok)done++;const c=per.get(el.dataset.cid)||[0,0];c[1]++;if(ok)c[0]++;per.set(el.dataset.cid,c);}
 document.getElementById('progress').textContent='Complete: '+done+' of '+targets.length+' targets';
 for(const s of document.querySelectorAll('[data-conv-progress]')){const c=per.get(s.dataset.convProgress)||[0,0];s.textContent=c[0]+'/'+c[1];}
}
function validateRows(rows,coder,draft=false){
  if(!Array.isArray(rows)||rows.length!==targets.length)throw new Error('The file must contain all '+targets.length+' allocated targets, including unfinished ones.');
  const seen=new Set();
  for(const r of rows){
   if(!r||typeof r!=='object'||Array.isArray(r)||r.schema_version!==config.version||r.stage!==config.stage)throw new Error('Wrong coding version or stage.');
   if(typeof r.coder_id!=='string'||!r.coder_id.trim()||r.coder_id!==coder)throw new Error('The coder names do not match.');
   const key=r.conversation_id+'#'+r.segment_order;
   if(typeof r.conversation_id!=='string'||!Number.isInteger(r.segment_order)||!byKey.has(key)||seen.has(key))throw new Error('Unexpected or repeated target: '+key);
   seen.add(key);
   if(!r.labels||typeof r.labels!=='object'||Array.isArray(r.labels)||Object.keys(r.labels).length!==Object.keys(config.axes).length||Object.keys(r.labels).some(a=>!Object.hasOwn(config.axes,a)))throw new Error('The four label fields do not match this form.');
   for(const [a,codes] of Object.entries(config.axes)){
    const v=r.labels[a];if(v===null)continue;
    if(config.selection[a]==='single'){if(typeof v!=='string'||!codes.includes(v))throw new Error('Invalid choice on '+key+': '+a);}
    else if(!Array.isArray(v)||!v.length||new Set(v).size!==v.length||v.some(x=>!codes.includes(x)))throw new Error('Invalid choices on '+key+': '+a);
   }
   if(!Array.isArray(r.flags)||new Set(r.flags).size!==r.flags.length||r.flags.some(f=>!['needs discussion','truncated capture',config.default.flag].includes(f)))throw new Error('Invalid flags on '+key);
   if(typeof r.notes!=='string'||!r.unavailable||typeof r.unavailable!=='object'||Array.isArray(r.unavailable))throw new Error('Invalid notes on '+key);
   for(const [a,reason] of Object.entries(r.unavailable))if(!Object.hasOwn(config.axes,a)||typeof reason!=='string'||(!draft&&(!reason.trim()||r.labels[a]!==null)))throw new Error('Invalid missing-evidence reason on '+key);
   const fixedTruncated=byKey.get(key).dataset.sourceTruncated==='true';
   if(fixedTruncated&&(!r.flags.includes('truncated capture')||r.labels['AI delegation']!==null||r.labels['AI response type']!==null))throw new Error('The source check requires the truncated flag and blank AI fields on '+key);
   if(!draft){
    for(const [a,neg] of [['AI delegation','ND'],['AI response type','none of the four applies']])if(r.labels[a]?.includes(neg)&&r.labels[a].length>1)throw new Error('Negative and positive choices conflict on '+key);
    if(r.flags.includes('truncated capture')&&(r.labels['AI delegation']!==null||r.labels['AI response type']!==null))throw new Error('Truncated responses must have blank AI fields on '+key);
    if(r.flags.includes(config.default.flag)&&(r.labels[config.default.axis]!==config.default.value||!r.notes.trim()))throw new Error('The default flag needs Task automation and a reason on '+key);
   }
  }
}
function applyRows(rows){
  for(const r of rows){
   const el=byKey.get(r.conversation_id+'#'+r.segment_order);
   for(const input of el.querySelectorAll('input[data-axis]')){const v=r.labels[input.dataset.axis];input.checked=Array.isArray(v)?v.includes(input.value):v===input.value;}
   el.querySelector('[data-notes]').value=r.notes;
   el.querySelector('[data-discussion]').checked=r.flags.includes('needs discussion');
   el.querySelector('[data-default]').checked=r.flags.includes(config.default.flag);
   el.querySelector('[data-truncated]').checked=el.dataset.sourceTruncated==='true'||r.flags.includes('truncated capture');
   for(const box of el.querySelectorAll('[data-unavailable-axis]'))box.value=r.unavailable[box.dataset.unavailableAxis]||'';
  }
}
function blankRows(coder){return targets.map(el=>({schema_version:config.version,coder_id:coder,stage:config.stage,conversation_id:el.dataset.cid,segment_order:Number(el.dataset.turn),labels:Object.fromEntries(Object.keys(config.axes).map(a=>[a,null])),flags:el.dataset.sourceTruncated==='true'?['truncated capture']:[],notes:'',unavailable:{}}));}
function readDraft(raw,coder){
 const saved=JSON.parse(raw);
 if(!saved||typeof saved!=='object'||Array.isArray(saved)||saved.format!==1||saved.packetFingerprint!==config.packetFingerprint||saved.coder!==coder||typeof saved.savedAt!=='string'||!Number.isFinite(Date.parse(saved.savedAt))||typeof saved.revision!=='string'||!saved.revision||typeof saved.writer!=='string'||typeof saved.needsDownload!=='boolean')throw new Error('The saved draft does not match this page and coder.');
 validateRows(saved.rows,coder,true);return saved;
}
function rememberCoder(){
 if(activeMetadataBlocked)return;
 localStorage.setItem(activeKey,JSON.stringify({name:activeCoder}));
}
function collect(coder,draft=false){
 return targets.map(el=>{
  const labels={};for(const a of Object.keys(config.axes)){const v=axisInputs(el,a).filter(x=>x.checked).map(x=>x.value);labels[a]=v.length?(config.selection[a]==='single'?v[0]:v):null;}
  const flags=[];if(el.querySelector('[data-discussion]').checked)flags.push('needs discussion');if(el.querySelector('[data-default]').checked)flags.push(config.default.flag);
  if(el.dataset.sourceTruncated==='true'||el.querySelector('[data-truncated]').checked)flags.push('truncated capture');
  const unavailable={};for(const box of el.querySelectorAll('[data-unavailable-axis]')){if(draft||box.value.trim())unavailable[box.dataset.unavailableAxis]=draft?box.value:box.value.trim();}
  return {schema_version:config.version,coder_id:coder,stage:config.stage,conversation_id:el.dataset.cid,segment_order:Number(el.dataset.turn),labels,flags,notes:el.querySelector('[data-notes]').value,unavailable};
 });
}
function saveDraft(allowPending=false){
 if(!activeCoder||savingBlocked||(!allowPending&&document.getElementById('coder').value.trim()!==activeCoder))return false;
 try{
  const key=draftKey(activeCoder),current=localStorage.getItem(key);
  if(current!==expectedRaw){pauseSaving('conflict','Another tab or window changed this saved draft. Automatic saving is paused. Download your current coding, then reopen this page to use the latest saved draft.');return false;}
  const rows=collect(activeCoder,true);validateRows(rows,activeCoder,true);
  const saved={format:1,packetFingerprint:config.packetFingerprint,coder:activeCoder,savedAt:new Date().toISOString(),revision:writer+'-'+Date.now().toString(36)+'-'+Math.random().toString(36).slice(2),writer,rows,needsDownload:unsavedChanges};
  const raw=JSON.stringify(saved);localStorage.setItem(key,raw);
  if(localStorage.getItem(key)!==raw){pauseSaving('conflict','Another tab or window changed this saved draft. Automatic saving is paused. Download your current coding, then reopen this page.');return false;}
  expectedRaw=raw;rememberCoder();
  saveStatus('saved','Saved on this browser for '+activeCoder+' at '+displayTime(saved.savedAt)+'. Also download a backup before closing.');return true;
 }catch(error){saveStatus('error','Automatic saving is unavailable. Your choices are still on this page. Download your current coding before closing.');return false;}
}
function activateCoder(automatic=false){
 const input=document.getElementById('coder'),coder=input.value.trim();
 if(!coder){lockCoding(true);saveStatus('unnamed','Enter your coder name, then choose Start or resume coding.');return;}
 if(activeCoder===coder&&!savingBlocked){lockCoding(false);saveDraft();return;}
 if(!automatic&&activeCoder&&unsavedChanges&&!confirm('Switch from '+activeCoder+' to '+coder+'? The new name will open its own saved draft or a blank page. Download your current coding first if you need a separate backup.')){input.value=activeCoder;lockCoding(false);return;}
 if(activeCoder&&!savingBlocked)saveDraft(true);
 activeCoder=coder;input.value=coder;savingBlocked='';expectedRaw=null;unsavedChanges=false;
 applyRows(blankRows(coder));lockCoding(false);
 try{
  expectedRaw=localStorage.getItem(draftKey(coder));
  if(expectedRaw!==null){
   let saved;try{saved=readDraft(expectedRaw,coder);}catch(error){pauseSaving('invalid','The saved draft could not be restored. It has been kept unchanged. Resume a downloaded file or code here and download it; automatic saving is paused for this name.');updateProgress();return;}
   applyRows(saved.rows);unsavedChanges=saved.needsDownload;
   saveStatus('restored','Restored '+coder+'\'s draft saved at '+displayTime(saved.savedAt)+'. Continue coding, and download a backup before closing.');
  }else saveStatus('ready','Coding as '+coder+'. Changes will be saved on this browser.');
  rememberCoder();
 }catch(error){saveStatus('error','Automatic saving is unavailable. You can still code and download your current coding before closing.');}
 updateProgress();
}
async function importCoding(file){
 if(!file)return;
 try{
  const lines=(await file.text()).replace(/^﻿/,'').split(/\r?\n/).filter(x=>x.trim());
  const rows=lines.map(x=>JSON.parse(x)),coder=rows[0]?.coder_id;
  validateRows(rows,coder);
  if(unsavedChanges&&!confirm('Loading this file replaces the choices currently on this page. Download your current coding first if you need to keep it. Continue?'))return;
  let stored=null,storageReadable=true,blocked='';
  try{stored=localStorage.getItem(draftKey(coder));if(stored!==null){try{readDraft(stored,coder);}catch(error){blocked='invalid';}}}catch(error){storageReadable=false;}
  if(stored!==null&&!blocked&&!confirm('This browser already has a draft for '+coder+'. Replace it with this downloaded file?'))return;
  if(activeCoder&&activeCoder!==coder&&!savingBlocked)saveDraft(true);
  activeCoder=coder;document.getElementById('coder').value=coder;expectedRaw=stored;savingBlocked=blocked;
  applyRows(rows);unsavedChanges=false;lockCoding(false);updateProgress();
  document.getElementById('resume-status').textContent='Loaded '+rows.length+' targets for '+coder+'. Continue coding, then download your updated file.';
  if(blocked)pauseSaving('invalid','Loaded your file. The unreadable browser draft has been kept unchanged, so automatic saving is paused for this name. Download your updated coding before closing.');
  else if(!storageReadable)saveStatus('error','Loaded your file. Automatic saving is unavailable; download your updated coding before closing.');
  else saveDraft();
 }catch(error){alert('The file was not loaded. '+error.message);}
 finally{document.getElementById('resume-file').value='';}
}
function download(){
 const coder=activeCoder;if(!coder){alert('Enter your coder name and choose Start or resume coding, or resume your own saved file.');return;}
 if(document.getElementById('coder').value.trim()!==coder){alert('These choices belong to '+coder+'. Restore that name or choose Start or resume coding to switch to another coder.');return;}
 const rows=collect(coder);
 for(const r of rows){
  const prefix=r.conversation_id+' #'+r.segment_order+': ';
  for(const [axis,neg] of [['AI delegation','ND'],['AI response type','none of the four applies']]){const v=r.labels[axis];if(v&&v.includes(neg)&&v.length>1){alert(prefix+'negative and positive selections conflict.');return;}}
  if(r.flags.includes('truncated capture')&&(r.labels['AI delegation']!==null||r.labels['AI response type']!==null)){alert(prefix+'clear both AI fields for a truncated response.');return;}
  if(r.flags.includes(config.default.flag)&&(r.labels[config.default.axis]!==config.default.value||!r.notes.trim())){alert(prefix+'a Task automation default needs that choice and a short reason.');return;}
  for(const a of Object.keys(r.unavailable)){if(r.labels[a]!==null){alert(prefix+'clear '+a+' or remove its missing-evidence reason.');return;}}
 }
 const blob=new Blob([rows.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/x-ndjson'}),u=URL.createObjectURL(blob),a=document.createElement('a');
 a.href=u;a.download=coder.replace(/[^a-zA-Z0-9_-]/g,'_')+'-'+config.stage+'-'+config.group+'.jsonl';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);unsavedChanges=false;saveDraft();
}
function edited(event){
 if(event.target.id==='resume-file')return;
 if(event.target.id==='coder'){
  const pending=document.getElementById('coder').value.trim()!==activeCoder;lockCoding(!activeCoder||pending);
  if(pending||!activeCoder)saveStatus('pending','Choose Start or resume coding to open this name\'s own draft. Your previous choices will not be copied to another name.');
  else if(savingBlocked)saveStatus(savingBlocked==='conflict'?'conflict':'error','Automatic saving is paused. Download your current coding before closing, then reopen this page.');
  else saveDraft();
  return;
 }
 if(!activeCoder||document.getElementById('coder').value.trim()!==activeCoder)return;
 unsavedChanges=true;updateProgress();saveDraft();
}
document.addEventListener('input',edited);
document.addEventListener('change',edited);
document.getElementById('activate-coder').addEventListener('click',()=>activateCoder());
document.getElementById('coder').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();activateCoder();}});
document.addEventListener('click',event=>{const b=event.target.closest('[data-clear-axis]');if(b&&!b.disabled){unsavedChanges=true;for(const input of b.closest('article').querySelectorAll('[data-axis]'))if(input.dataset.axis===b.dataset.clearAxis)input.checked=false;updateProgress();saveDraft();}});
window.addEventListener('beforeunload',event=>{if(unsavedChanges){event.preventDefault();event.returnValue='';}});
window.addEventListener('storage',event=>{if(activeCoder&&event.storageArea===localStorage&&(event.key===draftKey(activeCoder)||event.key===null)&&event.newValue!==expectedRaw)pauseSaving('conflict','Another tab or window changed this saved draft. Automatic saving is paused. Download your current coding, then reopen this page to use the latest saved draft.');});
function restoreOnOpen(){
 lockCoding(true);
 try{
  const raw=localStorage.getItem(activeKey);
  if(raw===null){saveStatus('unnamed','Enter your coder name, then choose Start or resume coding.');return;}
  let last;try{last=JSON.parse(raw);if(!last||typeof last!=='object'||typeof last.name!=='string'||!last.name.trim())throw new Error('Invalid saved name');}catch(error){activeMetadataBlocked=true;saveStatus('error','The previous coder name could not be restored. Enter your own name to look for its draft, or resume your downloaded file.');return;}
  document.getElementById('coder').value=last.name;activateCoder(true);
 }catch(error){saveStatus('error','Automatic saving is unavailable in this browser. Enter your name to code, and download your current coding before closing.');}
}
restoreOnOpen();updateProgress();
'''


def form_html(cid, n, truncated):
    groups = []
    for a, codes in DIRECT_CODES.items():
        typ = 'radio' if DIRECT_FIELDS[a]['selection'] == 'single' else 'checkbox'
        hints = DIRECT_FIELDS[a].get('hints', {})
        labels = ''.join(f'<label><input type="{typ}" name="{html.escape(cid)}-{n}-{html.escape(a)}" data-axis="{html.escape(a)}" '
                         f'value="{html.escape(c)}"> {html.escape(c)}'
                         + (f'<small>{html.escape(hints[c])}</small>' if c in hints else '') + '</label>' for c in codes)
        instruction = 'Select one' if typ == 'radio' else 'Select all supported values'
        locked = truncated and a in AI_AXES
        if locked:
            instruction = 'Locked blank: the source check marks this response truncated (D-32).'
        groups.append(f'<fieldset{" disabled" if locked else ""}><legend>{html.escape(a[:1].upper() + a[1:])}</legend>'
                      f'<p>{instruction}</p>{labels}</fieldset>')
    groups.append('<label>Notes or competing reading<textarea data-notes></textarea></label>')
    groups.append('<label><input type="checkbox" data-discussion> Needs discussion</label>'
                  '<label><input type="checkbox" data-default> Task automation default because constraint origin is unclear (add a reason)</label>')
    groups.append(f'<label><input type="checkbox" data-truncated{" checked disabled" if truncated else ""}> '
                  'Assistant response is truncated (leave both AI fields blank)</label>')
    groups.append('<details><summary>Missing evidence or clear a selection</summary><p>For a judgement that cannot be made, '
                  'clear that field and explain the missing evidence below.</p>'
                  + ''.join(f'<label>{html.escape(a)}<textarea data-unavailable-axis="{html.escape(a)}" placeholder="Leave empty unless '
                            f'essential evidence is unavailable"></textarea><button type="button" data-clear-axis="{html.escape(a)}">'
                            'Clear this field</button></label>' for a in DIRECT_CODES if not (truncated and a in AI_AXES))
                  + '</details>')
    return '<div class="coding">' + ''.join(groups) + '</div>'


def notes_html(notes):
    out, m2 = [], []
    where = {'Student Prompt': 'student field', 'LLM Response': 'assistant field'}
    for n in notes:
        if n['kind'] == 'M1':
            out.append(f'<aside class="capture"><strong>Submission note from the {where.get(n["field"], n["field"])}. '
                       'Not part of the conversation: do not code it or use it as context.</strong>'
                       f'<pre>{html.escape(n["text"])}</pre></aside>')
        else:
            m2.append(f'<pre>{html.escape(n["text"])}</pre>')
    if m2:
        out.append(f'<details class="capture"><summary>Interface text removed from this record ({len(m2)}), such as copy or edit '
                   'buttons. Not part of the conversation.</summary>' + ''.join(m2) + '</details>')
    return ''.join(out)


def packet_fingerprint(records, group, guide, dry_run):
    """Isolate drafts by instrument, displayed evidence and allocation, without storing that text."""
    identity = {'version': DIRECT_VERSION, 'stage': STAGE, 'group': group, 'preview': dry_run,
                'axes': DIRECT_CODES, 'fields': DIRECT_FIELDS, 'default': DIRECT_DEFAULT,
                'guide_version': guide.get('version'), 'guide_sha256': guide.get('sha256'),
                'records': [{k: r[k] for k in ('conversation_id', 'segment_order', 'student_prompt',
                                               'ai_response', 'notes', 'truncated')} for r in records]}
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def render(records, group, cids, guide, dry_run):
    by_c = {c: [r for r in records if r['conversation_id'] == c] for c in cids}
    n = len(records)
    banner = ('<p class="dry">PREVIEW ONLY. Built for checking with a provisional guide reference. This is not a coding packet: '
              'do not send it or code from it.</p>') if dry_run else ''
    guide_line = (f'<a href="guide.html" target="_blank" rel="noopener">Open coding guide {html.escape(guide["version"])}</a> '
                  '(included in this folder; keep it open while coding).' if guide['status'] == 'stable'
                  else 'The stable post-calibration guide is not attached to this preview.')
    head = (f'<header>{banner}<h1>Independent coding: {TITLES[group]}</h1>'
            f'<p>{n} target student interactions in {len(cids)} complete conversations. Every interaction shown is a target. '
            'Read each conversation in order; the earlier interactions are its context.</p>'
            f'<p>{guide_line} No expected labels are shown.</p>'
            '<p>Text marked as a submission note or interface text was separated from the dialogue during source checks. '
            'It is not part of the conversation: do not code it or use it as context. Where the source check found an assistant '
            'response truncated (D-32), both AI fields are locked blank with the truncated flag set; code the two student '
            'fields from the evidence.</p>'
            '<p>Open this file locally. Changes are saved on this browser when automatic saving is available. Reopen the same '
            'file in the same browser to restore your last draft. No data is sent to a server.</p>'
            '<label>Your coder name <input id="coder" autocomplete="off"></label>'
            '<button id="activate-coder" type="button">Start or resume coding</button>'
            '<p id="autosave-status" role="status" aria-live="polite">Checking for a saved draft...</p>'
            '<button onclick="download()">Download my current coding</button>'
            '<label>Resume your own saved coding <input id="resume-file" type="file" accept=".jsonl,.ndjson" '
            'onchange="importCoding(this.files[0])"></label>'
            '<p id="resume-status" aria-live="polite">You can also resume your own previously downloaded file. Loading replaces '
            'the choices on this page. A different coder name opens that name\'s own draft or a blank page.</p>'
            '<p>You may save an incomplete draft. Missing judgements are flagged when the returned file is checked. Keep your '
            'first returned file unchanged. Download a backup regularly and before closing. Local-file storage may be restricted '
            'by your browser; private browsing, clearing browser data, moving the file or changing devices may prevent recovery. '
            'Check the saving message above. Keep only one tab open for this coding file.</p>'
            '<p id="progress" aria-live="polite"></p></header>')
    nav = '<nav>' + ''.join(f'<a href="#{html.escape(c)}">{html.escape(c)}</a> <span data-conv-progress="{html.escape(c)}"></span>'
                            ' &nbsp; ' for c in cids) + '</nav>'
    body = []
    for c in cids:
        rows = by_c[c]
        part = [f'<section class="conversation" id="{html.escape(c)}"><h2>Conversation {html.escape(c)} · {len(rows)} interactions</h2>']
        prev = None
        for r in rows:
            k = r['segment_order']
            if prev is None and k != 1:
                part.append(f'<p class="gap">The record of this conversation begins at #{k}.</p>')
            elif prev is not None and k != prev + 1:
                part.append(f'<p class="gap">Interactions #{prev + 1} to #{k - 1} are not in the record. '
                            'The interaction below is not adjacent to the one above.</p>')
            prev = k
            source = ('<p class="source">Source check (D-32): this assistant response is incomplete in the capture. '
                      'Both AI fields stay blank with the truncated flag set.</p>') if r['truncated'] else ''
            part.append(f'<article id="{html.escape(c)}-{k}" data-cid="{html.escape(c)}" data-turn="{k}" data-target="true" '
                        f'data-source-truncated="{str(r["truncated"]).lower()}"><h3>{html.escape(c)} #{k}<span class="state"></span></h3>'
                        '<details open><summary>Student prompt and assistant response</summary>'
                        f'<h4>Student</h4><pre>{html.escape(r["student_prompt"])}</pre>'
                        f'<h4>Assistant</h4><pre>{html.escape(r["ai_response"])}</pre></details>'
                        f'{source}{notes_html(r["notes"])}{form_html(c, k, r["truncated"])}</article>')
        part.append('</section>')
        body.append(''.join(part))
    config = {'version': DIRECT_VERSION, 'stage': STAGE, 'group': group, 'axes': DIRECT_CODES,
              'packetFingerprint': packet_fingerprint(records, group, guide, dry_run),
              'selection': {a: f['selection'] for a, f in DIRECT_FIELDS.items()}, 'default': DIRECT_DEFAULT}
    payload = json.dumps(config, ensure_ascii=False).replace('<', '\\u003c')
    title = ('PREVIEW ' if dry_run else '') + f'Independent coding: {TITLES[group]}'
    return (f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{STYLE}</style><body>{head}{nav}{"".join(body)}'
            f'<script>{JS.replace("CONFIG", payload)}</script></body></html>')


def blank(coder, records):
    return [{'schema_version': DIRECT_VERSION, 'coder_id': coder, 'stage': STAGE, 'conversation_id': r['conversation_id'],
             'segment_order': r['segment_order'], 'labels': {a: None for a in DIRECT_CODES},
             'flags': [TRUNCATED_FLAG] if r['truncated'] else [], 'notes': '', 'unavailable': {}} for r in records]


def write_packet(out, group, records, cids, coders, guide, evidence, traces, receipts, report, dry_run, deid_info=None):
    keys = [(r['conversation_id'], r['segment_order']) for r in records]
    if len(keys) != len(set(keys)) or any(k[0] not in cids for k in keys):
        raise ValueError('Duplicate target or record outside the group')
    blanks = {c: blank(c, records) for c in coders}
    for rows in blanks.values():
        for row in rows:
            errors = validate(row, allow_blank=True)
            if errors:
                raise ValueError(f'Blank record fails the validator: {row["conversation_id"]} #{row["segment_order"]}: {errors}')
    guide_bytes = None
    if guide['status'] == 'stable':
        guide_source = ROOT / guide['path']
        guide_bytes = guide_source.read_bytes()
        if hashlib.sha256(guide_bytes).hexdigest() != guide['sha256']:
            raise ValueError('The stable guide changed after its hash was recorded')
        guide_page = render_guide(guide_bytes.decode('utf-8-sig'), guide['version'])
    page = render(records, group, cids, guide, dry_run)
    built = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
    manifest = {'schema_version': DIRECT_VERSION, 'phase': STAGE, 'group': group, 'unit': 'student interaction',
                'status': 'dry_run_preview' if dry_run else 'formal_blank_packet',
                'target_count': len(records), 'conversation_count': len(cids), 'student_count': len(cids),
                'targets': [{'conversation_id': c, 'segment_order': k} for c, k in keys], 'conversation_ids': cids,
                'source_truncated_targets': [{'conversation_id': r['conversation_id'], 'segment_order': r['segment_order']}
                                             for r in records if r['truncated']],
                'guide': guide, 'allocation': 'data/coding/direct-20260924/allocation-20261010', 'provenance': evidence,
                'deidentification': {'file': deid_info, 'replacements_applied': sum(
                    1 for t in traces for x in t.get('replacements', []) if x.get('kind') != 'shared_url'),
                    'shared_links_hidden': sum(1 for t in traces for x in t.get('replacements', []) if x.get('kind') == 'shared_url'),
                    'screen_unresolved': report.get('unresolved_total'),
                    'clearance': 'Pattern screen plus listed rules; not a certification of complete de-identification'},
                'builder': 'coding/build_direct_packet.py', 'builder_sha256': sha(__file__), 'built_utc': built,
                'local_draft': {'format': 1, 'packet_fingerprint': packet_fingerprint(records, group, guide, dry_run),
                                'scope': 'This browser and packet, separately for each activated coder name; coding values only'},
                'labels_prefilled': False, 'coding_packet_released': False}
    if group == 'evaluation_first':
        manifest['purpose'] = ('Independent coding by both human coders; a pool for later student-grouped '
                               'training and validation. The allocation group key is historical; this is not a fixed final test set.')
    out.mkdir(parents=True)
    (out / 'responses').mkdir()
    private = out / 'private-researcher'
    private.mkdir()
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    (out / 'coding.html').write_text(page, encoding='utf-8')
    if guide_bytes is not None:
        (out / 'guide.md').write_bytes(guide_bytes)
        (out / 'guide.html').write_text(guide_page, encoding='utf-8')
    for coder, rows in blanks.items():
        (out / 'responses' / f'{coder}-blank.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    v.write_json(private / 'text-derivation.json', traces)
    v.write_json(private / 'source-checks.json', receipts)
    v.write_json(private / 'identifier-screen.json', report)
    v.write_json(private / 'roster.json', [{'conversation_id': c,
                                            'student_id': next(r['student_id'] for r in records if r['conversation_id'] == c),
                                            'interactions': sum(r['conversation_id'] == c for r in records)} for c in cids])
    preview = 'PREVIEW ONLY: not a coding packet; do not send it or code from it.\n\n' if dry_run else ''
    guide_text = (f'Coding guide: version {guide["version"]}, adopted in {guide["decision"]}. '
                  'Open guide.html in this folder; guide.md preserves the same source text.'
                  if guide['status'] == 'stable' else 'Coding guide: not yet attached (provisional preview).')
    (out / 'README.md').write_text(
        f'{preview}# Independent coding packet: {TITLES[group]}\n\n{guide_text}\n\n'
        f'Open coding.html locally. It holds {len(records)} target student interactions in {len(cids)} complete conversations; '
        'every interaction shown is a target and the earlier interactions are its context. Select one Student interaction type, '
        'every supported topic and every applicable AI value. No labels are prefilled.\n\n'
        'Enter your name and choose Start or resume coding. Changes are saved on this browser when available; reopening the '
        'same file in the same browser restores the last draft. Check the saving message. Download a backup regularly and '
        'before closing; browser restrictions or changes to the browser, file location or device can prevent recovery. '
        'Resume your own downloaded JSONL if needed. Keep one tab open for this file.\n\n'
        'Code independently. Download your file from the page and return it unchanged; later discussion is recorded separately '
        'and does not overwrite your first return. Use the missing-evidence fields for a judgement that cannot be made. '
        'Responses marked truncated by the source check (D-32) keep both AI fields blank; this is set in the page.\n\n'
        + (manifest['purpose'] + '\n\n' if group == 'evaluation_first' else '') +
        'Do not share private-researcher/ with a second coder.\n', encoding='utf-8')
    (private / 'README.txt').write_text(
        '研究者私有材料，不发给第二编码员。\n'
        'text-derivation.json：每个字段原文、去掉外置标注后、显示文本的哈希，及外置标注和去标识替换记录。\n'
        'source-checks.json：本组对话的现行来源核查回执（D-100生效版）。\n'
        'identifier-screen.json：联系方式、用户名等的正则筛查候选，已遮掩；未人工核对，不是去标识结论。\n'
        'roster.json：对话、学生编号和互动数。\n', encoding='utf-8')
    return {'output': str(out), 'group': group, 'status': manifest['status'], 'targets': len(records),
            'conversations': len(cids), 'source_truncated': len(manifest['source_truncated_targets']),
            'blank_files': sorted(f'{c}-blank.jsonl' for c in coders),
            'deidentification_replacements': manifest['deidentification']['replacements_applied'],
            'screen_candidates': {k: r['candidates'] for k, r in report.items() if isinstance(r, dict)},
            'screen_unresolved': report.get('unresolved_total')}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--group', choices=GROUPS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dry-run', action='store_true', help='Preview with the provisional guide reference; not a packet')
    p.add_argument('--guide', type=Path)
    p.add_argument('--guide-version')
    p.add_argument('--decision', help='decisions.md entry that adopts the stable guide, e.g. D-101')
    p.add_argument('--coders', nargs='+')
    p.add_argument('--deidentify', type=Path, help='Conversation-scoped replacement rules and reviewed screen tokens (under data/)')
    a = p.parse_args()
    if a.dry_run:
        if a.guide or a.guide_version or a.decision:
            p.error('a dry run uses the provisional guide reference only')
        guide = provisional_guide()
    else:
        if not (a.guide and a.guide_version and a.decision and a.deidentify):
            p.error('a formal build needs --guide, --guide-version, --decision and --deidentify')
        guide = stable_guide(a.guide, a.guide_version, a.decision)
    spec, deid_info = load_deidentification(a.deidentify)
    if not a.dry_run and spec['status'] != 'adopted':
        raise SystemExit('REFUSED: the de-identification file is not marked adopted by DW')
    out = check_output(a.output, formal=not a.dry_run)
    effective, rows, groups, pool, evidence = verified_allocation()
    cids = groups[a.group]
    records, traces = derive_group(cids, rows, effective, pool, make_deidentifier(spec))
    report = screen(records, spec)
    if not a.dry_run and report['unresolved_total']:
        raise SystemExit(f'REFUSED: {report["unresolved_total"]} identifier screen candidates are unresolved')
    summary = write_packet(out, a.group, records, cids, a.coders or DEFAULT_CODERS[a.group], guide, evidence, traces,
                           {c: effective[c] for c in cids}, report, a.dry_run, deid_info)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
