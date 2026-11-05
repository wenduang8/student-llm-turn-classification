"""Guards for receiving returns and comparing coders, tested with bad input as well as good.
All records here are synthetic interface tests on made-up keys; none is a student annotation."""
import copy, hashlib, json, tempfile, unittest
from pathlib import Path
import openpyxl
from direct_coding import DIRECT_VERSION
from receive_return import receive, check
from compare_coders import compare

TARGETS = [('SYN_A', 1), ('SYN_A', 2), ('SYN_B', 1)]

def rec(coder, cid, n, **labels):
    base = {'student interaction type': 'Task automation', 'topic': ['code'], 'AI delegation': ['ND'], 'AI response type': ['Explain']}
    base.update(labels)
    return {'schema_version': DIRECT_VERSION, 'coder_id': coder, 'stage': 'calibration', 'conversation_id': cid,
            'segment_order': n, 'labels': base, 'flags': [], 'notes': '', 'unavailable': {}}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.d = Path(self.tmp.name)
        self.alloc = self.d / 'alloc'; self.alloc.mkdir()
        (self.alloc / 'manifest.json').write_text(json.dumps({'schema_version': DIRECT_VERSION, 'phase': 'calibration',
            'targets': [{'conversation_id': c, 'segment_order': n} for c, n in TARGETS]}), encoding='utf-8')
    def tearDown(self): self.tmp.cleanup()
    def write(self, name, rows, raw=None):
        p = self.d / name
        p.write_bytes(raw if raw is not None else ('\n'.join(json.dumps(r) for r in rows) + '\n').encode('utf-8'))
        return p
    def good(self, coder='X'): return [rec(coder, c, n) for c, n in TARGETS]

class Intake(Base):
    def test_stores_exact_bytes_and_never_replaces_first(self):
        raw = ('﻿' + '\r\n'.join(json.dumps(r) for r in self.good()) + '\r\n').encode('utf-8')
        src = self.write('ret.jsonl', None, raw)
        dest, r = receive(src, self.alloc, 'X')
        self.assertEqual(dest.read_bytes(), raw); self.assertEqual(r['status'], 'passes')
        src.write_bytes(raw + b'\n')
        with self.assertRaises(FileExistsError):
            receive(src, self.alloc, 'X')
        self.assertEqual(dest.read_bytes(), raw)
        dest2, _ = receive(src, self.alloc, 'X', label='v2')
        self.assertNotEqual(dest, dest2)

    def test_bad_file_is_stored_then_reported(self):
        src = self.write('bad.jsonl', None, b'{not json}\n')
        dest, r = receive(src, self.alloc, 'X')
        self.assertTrue(dest.exists()); self.assertEqual(r['status'], 'fails')
        self.assertTrue(dest.with_name(dest.name + '.receipt.json').exists())

    def test_rejects_allocation_level_faults(self):
        m = json.loads((self.alloc / 'manifest.json').read_text())
        cases = {
            'missing target': self.good()[:2],
            'extra key': self.good() + [rec('X', 'SYN_C', 1)],
            'duplicate key': self.good() + [rec('X', 'SYN_A', 1)],
            'two coders': self.good()[:2] + [rec('Y', 'SYN_B', 1)],
        }
        for name, rows in cases.items():
            r = check(self.write(name + '.jsonl', rows), m)
            self.assertEqual(r['status'], 'fails', name)
        rows = self.good(); rows[0]['stage'] = 'independent'
        self.assertEqual(check(self.write('stage.jsonl', rows), m)['status'], 'fails')
        rows = self.good(); rows[0]['schema_version'] = 'direct-20260922'
        self.assertEqual(check(self.write('ver.jsonl', rows), m)['status'], 'fails')

    def test_blank_fails_complete_return_but_passes_draft(self):
        rows = self.good(); rows[1]['labels']['topic'] = None
        p = self.write('blank.jsonl', rows); m = json.loads((self.alloc / 'manifest.json').read_text())
        self.assertEqual(check(p, m)['status'], 'fails')
        self.assertEqual(check(p, m, draft=True)['status'], 'passes')
        self.assertEqual(check(p, m, draft=True)['completeness_by_axis']['topic']['blank'], 1)

    def test_invalid_label_value_fails(self):
        rows = self.good(); rows[2]['labels']['AI delegation'] = ['ND', 'offer']
        m = json.loads((self.alloc / 'manifest.json').read_text())
        self.assertEqual(check(self.write('neg.jsonl', rows), m)['status'], 'fails')

    def test_source_truncation_cannot_be_removed_or_coded(self):
        m = json.loads((self.alloc / 'manifest.json').read_text())
        m['source_truncated_targets'] = [{'conversation_id': 'SYN_A', 'segment_order': 2}]
        (self.alloc / 'manifest.json').write_text(json.dumps(m), encoding='utf-8')
        rows = self.good()
        rows[1]['flags'] = ['truncated capture']
        rows[1]['labels']['AI delegation'] = rows[1]['labels']['AI response type'] = None
        self.assertEqual(check(self.write('valid-truncated.jsonl', rows), m)['status'], 'passes')

        # Removing the flag is invalid even if a coder adds generic missing-evidence reasons.
        no_flag = copy.deepcopy(rows)
        no_flag[1]['flags'] = []
        no_flag[1]['unavailable'] = {'AI delegation': 'capture missing', 'AI response type': 'capture missing'}
        for draft in (False, True):
            report = check(self.write(f'no-flag-{draft}.jsonl', no_flag), m, draft=draft)
            self.assertEqual(report['status'], 'fails')
            self.assertIn('source check requires the truncated capture flag', str(report['line_errors']))

        # A fully filled manual edit used to pass the ordinary record validator.
        edited = self.good()
        source = self.write('manual-edit.jsonl', edited)
        dest, report = receive(source, self.alloc, 'X')
        self.assertEqual(dest.read_bytes(), source.read_bytes())
        self.assertEqual(report['status'], 'fails')
        self.assertIn('both AI fields to stay blank', str(report['line_errors']))
        self.assertTrue(dest.with_name(dest.name + '.receipt.json').exists())

    def test_source_truncated_manifest_key_must_be_target(self):
        m = json.loads((self.alloc / 'manifest.json').read_text())
        m['source_truncated_targets'] = [{'conversation_id': 'SYN_OUTSIDE', 'segment_order': 1}]
        with self.assertRaisesRegex(ValueError, 'must be allocated targets'):
            check(self.write('outside.jsonl', self.good()), m)

    def test_wrong_field_types_are_reported_not_crashed(self):
        m = json.loads((self.alloc / 'manifest.json').read_text())
        variants = []
        r = self.good(); r[1]['segment_order'] = [2]; variants.append(r)
        r = self.good(); r[0]['conversation_id'] = {'id': 'SYN_A'}; variants.append(r)
        r = self.good(); r[0]['coder_id'] = ['X']; variants.append(r)
        r = self.good(); r[2]['labels'] = ['code']; variants.append(r)
        r = self.good(); r[2]['flags'] = 'truncated capture'; r[2]['unavailable'] = ['topic']; variants.append(r)
        r = self.good(); r[0]['segment_order'] = True; variants.append(r)
        r = self.good(); r[0]['notes'] = {'x': 1}; variants.append(r)               # regression: passed before 30 Sep review 2
        r = [rec(['X'], c, n) for c, n in TARGETS]; variants.append(r)               # regression: coder_id list in every row passed
        for i, rows in enumerate(variants):
            dest, rep = receive(self.write(f'types{i}.jsonl', rows), self.alloc, f'X{i}')
            self.assertEqual(rep['status'], 'fails', i)
            self.assertTrue(dest.with_name(dest.name + '.receipt.json').exists(), i)
            self.assertFalse(any('checker stopped' in x for x in rep['problems']), i)   # checks completed
            self.assertEqual(check(dest, m)['status'], 'fails', i)                       # and do so without the fallback
        self.assertEqual(check(self.write('list.jsonl', None, b'[1, 2]\n'), m)['status'], 'fails')

    def test_checker_failure_still_writes_receipt(self):
        import receive_return
        from unittest import mock
        src = self.write('ok.jsonl', self.good())
        with mock.patch.object(receive_return, 'check', side_effect=RuntimeError('boom')):
            dest, rep = receive_return.receive(src, self.alloc, 'X')
        self.assertEqual(rep['status'], 'fails')
        self.assertEqual(dest.read_bytes(), src.read_bytes())
        saved = json.loads(dest.with_name(dest.name + '.receipt.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['status'], 'fails'); self.assertIn('boom', saved['problems'][0])

    def test_type_errors_are_named(self):
        m = json.loads((self.alloc / 'manifest.json').read_text())
        r = self.good(); r[1]['notes'] = ['a']; r[2]['stage'] = ['calibration']
        rep = check(self.write('named.jsonl', r), m)
        msgs = ' '.join(e for x in rep['line_errors'] for e in x['errors'])
        self.assertIn('must be text: notes', msgs); self.assertIn('stage', msgs)

class Compare(Base):
    def test_formula_like_notes_stay_text(self):
        notes = ['=1+1', '+SUM(A1:A2)', '-2+3', '@cmd', '=HYPERLINK("x")']
        a = [rec('A', c, n) for c, n in TARGETS]; b = [rec('B', c, n) for c, n in TARGETS]
        for r, t in zip(a, notes): r['notes'] = t
        b[0]['notes'] = '=A1'
        out = self.d / 'text.xlsx'
        compare(self.alloc, self.stored('A', a), self.stored('B', b), out)
        ws = openpyxl.load_workbook(out)['标记与备注']
        got = [(ws.cell(row=i + 2, column=5).value, ws.cell(row=i + 2, column=5).data_type) for i in range(len(TARGETS))]
        self.assertEqual(got, [(t, 's') for t in notes[:len(TARGETS)]])
        self.assertEqual((ws['F2'].value, ws['F2'].data_type), ('=A1', 's'))

    def stored(self, coder, rows, draft=False):
        dest, r = receive(self.write(coder + '.jsonl', rows), self.alloc, coder, draft=draft)
        return dest
    def test_classifies_each_axis_and_leaves_inputs_unchanged(self):
        a = [rec('A', c, n) for c, n in TARGETS]
        b = copy.deepcopy(a)
        for r in b: r['coder_id'] = 'B'
        b[0]['labels']['student interaction type'] = 'Guided generation'
        b[1]['labels']['topic'] = ['code', 'concept']
        b[2]['labels']['AI response type'] = ['Solve-Provide']
        a[2]['flags'] = ['truncated capture']; a[2]['labels']['AI delegation'] = a[2]['labels']['AI response type'] = None
        fa, fb = self.stored('A', a), self.stored('B', b)
        h = (sha(fa), sha(fb))
        out = self.d / 'cmp.xlsx'
        res = compare(self.alloc, fa, fb, out, describe_a='second coder, independent first round', describe_b='researcher version, not independent')
        info = {r[0]: r[1] for r in openpyxl.load_workbook(out)['说明'].iter_rows(min_row=2, values_only=True)}
        self.assertEqual(info['记录 B 的性质'], 'researcher version, not independent')
        self.assertEqual((sha(fa), sha(fb)), h)
        ws = openpyxl.load_workbook(out)['逐轴对照']
        got = {(r[0], r[1], r[2]): r[5] for r in ws.iter_rows(min_row=2, values_only=True)}
        self.assertEqual(got[('SYN_A', 1, 'student interaction type')], '不同')
        self.assertEqual(got[('SYN_A', 2, 'topic')], '部分重叠')
        self.assertEqual(got[('SYN_B', 1, 'AI response type')], '仅一方留空')
        self.assertEqual(got[('SYN_B', 1, 'AI delegation')], '仅一方留空')
        self.assertEqual(got[('SYN_A', 1, 'topic')], '相同')
        self.assertEqual(res['discussion_rows'], 4)
        with self.assertRaises(FileExistsError):
            compare(self.alloc, fa, fb, out)
    def test_refuses_changed_failed_or_same_coder_files(self):
        fa = self.stored('A', [rec('A', c, n) for c, n in TARGETS])
        fb = self.stored('B', [rec('B', c, n) for c, n in TARGETS])
        fb.write_bytes(fb.read_bytes() + b'\n')
        with self.assertRaises(ValueError): compare(self.alloc, fa, fb, self.d / 'x1.xlsx')
        rows = [rec('C', c, n) for c, n in TARGETS]; rows[0]['labels']['topic'] = None
        fc = self.stored('C', rows)
        with self.assertRaises(ValueError): compare(self.alloc, fa, fc, self.d / 'x2.xlsx')
        fd = self.stored('D', [rec('D', c, n) for c, n in TARGETS[:2]] + [rec('D', 'SYN_B', 1, topic=None)], draft=True)
        with self.assertRaises(ValueError): compare(self.alloc, fa, fd, self.d / 'x3.xlsx')
        fe = self.stored('A2', [rec('A', c, n) for c, n in TARGETS])
        with self.assertRaises(ValueError): compare(self.alloc, fa, fe, self.d / 'x4.xlsx')
        self.assertFalse(any((self.d / f'x{i}.xlsx').exists() for i in range(1, 5)))

class RefBase(Base):
    """Helpers for record B as a fixed reference without a receipt (D-150)."""
    def stored(self, coder, rows):
        dest, r = receive(self.write(coder + '.jsonl', rows), self.alloc, coder)
        return dest
    def reference(self, rows, name='ref.jsonl', listed=None, extra=''):
        p = self.write(name, rows)
        reg = self.d / 'README.md'
        h = sha(p) if listed is None else listed
        reg.write_text('| 文件 | 是什么 | SHA-256 |\n|---|---|---|\n' + f'| {name} | reference | {h} |\n' + extra, encoding='utf-8')
        return p, reg
    def ref_rows(self):
        rows = [rec('R', c, n) for c, n in TARGETS]
        for r in rows: r['review_history'] = ['kept as extra key']; r['stage'] = 'calibration'
        return rows

class Reference(RefBase):
    """Record B as a fixed reference, checked against its registry SHA-256."""
    def test_reference_is_compared_without_receipt_and_left_unchanged(self):
        a = [rec('A', c, n) for c, n in TARGETS]; a[0]['labels']['topic'] = ['code', 'concept']
        fa = self.stored('A', a)
        fr, reg = self.reference(self.ref_rows())
        h = sha(fr)
        out = self.d / 'ref.xlsx'
        res = compare(self.alloc, fa, fr, out, reference_b=reg)
        self.assertEqual(sha(fr), h)
        self.assertFalse(fr.with_name(fr.name + '.receipt.json').exists())
        self.assertFalse((self.alloc / 'responses' / 'received' / 'R-calibration-first.jsonl').exists())
        info = {r[0]: r[1] for r in openpyxl.load_workbook(out)['说明'].iter_rows(min_row=2, values_only=True)}
        self.assertIn('参照文件', info['记录 B'])
        ws = openpyxl.load_workbook(out)['讨论记录']
        hdr = [c.value for c in ws[1]]
        self.assertEqual(hdr[6:9], ['仅 A 选', '仅 B 选', '分歧类型'])
        row = [c.value for c in ws[2]]
        self.assertEqual((row[2], row[6], row[7]), ('topic', 'concept', None))
        self.assertEqual(res['discussion_rows'], 1)
    def test_reference_refused_unless_registry_hash_matches(self):
        fa = self.stored('A', [rec('A', c, n) for c, n in TARGETS])
        fr, reg = self.reference(self.ref_rows(), listed='0' * 64)
        with self.assertRaises(ValueError): compare(self.alloc, fa, fr, self.d / 'y1.xlsx', reference_b=reg)
        fr, reg = self.reference(self.ref_rows(), name='ref2.jsonl', listed='')
        reg.write_text(reg.read_text(encoding='utf-8').replace('| reference |  |', f'| reference | {sha(fr)[:16]}… |'), encoding='utf-8')
        with self.assertRaises(ValueError): compare(self.alloc, fa, fr, self.d / 'y2.xlsx', reference_b=reg)
        fr, reg = self.reference(self.ref_rows(), name='ref3.jsonl', extra=f'| ref3.jsonl | listed twice | {"1" * 64} |\n')
        with self.assertRaises(ValueError): compare(self.alloc, fa, fr, self.d / 'y3.xlsx', reference_b=reg)
        self.assertFalse(any((self.d / f'y{i}.xlsx').exists() for i in range(1, 4)))
    def test_reference_refused_when_records_are_wrong(self):
        fa = self.stored('A', [rec('A', c, n) for c, n in TARGETS])
        cases = {'missing target': self.ref_rows()[:2], 'duplicate': self.ref_rows() + [self.ref_rows()[0]]}
        bad = self.ref_rows(); bad[1]['labels']['topic'] = ['not a topic']; cases['invalid label'] = bad
        two = self.ref_rows(); two[2]['coder_id'] = 'S'; cases['two coders'] = two
        same = self.ref_rows()
        for r in same: r['coder_id'] = 'A'
        cases['same coder as A'] = same
        for i, (name, rows) in enumerate(cases.items()):
            fr, reg = self.reference(rows, name=f'bad{i}.jsonl')
            with self.subTest(name), self.assertRaises(ValueError):
                compare(self.alloc, fa, fr, self.d / f'z{i}.xlsx', reference_b=reg)
            self.assertFalse((self.d / f'z{i}.xlsx').exists())
    def test_reference_does_not_relax_record_a(self):
        fr, reg = self.reference(self.ref_rows())
        fa = self.write('unstored.jsonl', [rec('A', c, n) for c, n in TARGETS])
        with self.assertRaises(ValueError): compare(self.alloc, fa, fr, self.d / 'w.xlsx', reference_b=reg)

class Groups(RefBase):
    def test_groups_order_rows_and_must_cover_them(self):
        a = [rec('A', c, n) for c, n in TARGETS]
        a[0]['labels']['topic'] = ['code', 'concept']; a[2]['labels']['student interaction type'] = 'Guided generation'
        a[1]['flags'] = ['needs discussion']
        fa = self.stored('A', a); fr, reg = self.reference(self.ref_rows())
        g = {'groups': [{'id': 'S', 'axis': 'student interaction type', 'title': 's', 'experience': 'e', 'refs': 'D-1'},
                        {'id': 'T', 'axis': 'topic', 'title': 't'}, {'id': 'N', 'axis': '（整条）', 'title': 'n'}],
             'assign': [{'conversation_id': 'SYN_A', 'segment_order': 1, 'axis': 'topic', 'group': 'T', 'representative': True},
                        {'conversation_id': 'SYN_A', 'segment_order': 2, 'axis': '（整条）', 'group': 'N'},
                        {'conversation_id': 'SYN_B', 'segment_order': 1, 'axis': 'student interaction type', 'group': 'S'}]}
        gp = self.d / 'g.json'; gp.write_text(json.dumps(g), encoding='utf-8')
        out = self.d / 'g.xlsx'
        res = compare(self.alloc, fa, fr, out, reference_b=reg, groups=gp)
        self.assertEqual(res['groups'], {'S': 1, 'T': 1, 'N': 1})
        wb = openpyxl.load_workbook(out)
        ws = wb['讨论记录']
        self.assertEqual([r[0] for r in ws.iter_rows(min_row=2, values_only=True)], ['S', 'T', 'N'])
        self.assertEqual([c.value for c in ws[3]][:3], ['T', '是', 'SYN_A'])
        dv = {str(v.sqref): v.formula1 for v in ws.data_validations.dataValidation}
        self.assertIn('K2:K4', dv); self.assertIn('N2:N4', dv)
        gs = list(wb['讨论组'].iter_rows(min_row=2, values_only=True))
        self.assertEqual(gs[1][3:5], (1, 'SYN_A #1'))
        g['assign'].pop(); gp.write_text(json.dumps(g), encoding='utf-8')
        with self.assertRaises(ValueError): compare(self.alloc, fa, fr, self.d / 'g2.xlsx', reference_b=reg, groups=gp)
        g['assign'] += [{'conversation_id': 'SYN_B', 'segment_order': 1, 'axis': 'student interaction type', 'group': 'S'},
                        {'conversation_id': 'SYN_B', 'segment_order': 1, 'axis': 'topic', 'group': 'T'}]
        gp.write_text(json.dumps(g), encoding='utf-8')
        with self.assertRaises(ValueError): compare(self.alloc, fa, fr, self.d / 'g3.xlsx', reference_b=reg, groups=gp)
        self.assertFalse((self.d / 'g2.xlsx').exists() or (self.d / 'g3.xlsx').exists())

if __name__ == '__main__':
    unittest.main()
