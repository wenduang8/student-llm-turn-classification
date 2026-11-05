"""Checks for coder_agreement.py on synthetic records with made-up keys; none is a student annotation."""
import json, unittest
from pathlib import Path
from test_return_intake import Base, rec, sha
from receive_return import receive
from coder_agreement import agreement, cohen_kappa

class Kappa(unittest.TestCase):
    def test_textbook_values(self):
        # 50 items: 20 yes/yes, 5 yes/no, 10 no/yes, 15 no/no -> po 0.70, pe 0.50, kappa 0.40
        pairs = [('y', 'y')] * 20 + [('y', 'n')] * 5 + [('n', 'y')] * 10 + [('n', 'n')] * 15
        self.assertAlmostEqual(cohen_kappa(pairs, ['y', 'n']), 0.4)
        self.assertEqual(cohen_kappa([('y', 'y')] * 3, ['y', 'n']), None)
        self.assertAlmostEqual(cohen_kappa([('y', 'n'), ('n', 'y')], ['y', 'n']), -1.0)

class Agreement(Base):
    def stored(self, coder, rows):
        return receive(self.write(coder + '.jsonl', rows), self.alloc, coder)[0]
    def test_exact_set_match_and_blanks(self):
        a = [rec('A', c, n) for c, n in self.targets()]
        b = [rec('B', c, n) for c, n in self.targets()]
        a[0]['labels']['topic'] = ['code', 'concept']; b[0]['labels']['topic'] = ['concept', 'code']   # same set
        b[1]['labels']['topic'] = ['code', 'concept']                                                   # partial
        b[0]['labels']['student interaction type'] = 'Guided generation'
        a[2]['flags'] = ['truncated capture']; a[2]['labels']['AI delegation'] = a[2]['labels']['AI response type'] = None
        fa, fb = self.stored('A', a), self.stored('B', b)
        h = (sha(fa), sha(fb))
        out = self.d / 'agr.json'
        r = agreement(self.alloc, fa, fb, out)
        self.assertEqual((sha(fa), sha(fb)), h)
        t, s, d = r['axes']['topic'], r['axes']['student interaction type'], r['axes']['AI delegation']
        self.assertEqual((t['agree'], t['scored'], t['status_counts'].get('部分重叠')), (2, 3, 1))
        self.assertEqual((s['agree'], s['scored']), (2, 3))
        self.assertIn('cohen_kappa', s); self.assertNotIn('cohen_kappa', t)
        self.assertEqual((d['scored'], d['one_side_empty_not_scored'], d['percent']), (2, 1, 100.0))
        # topic: A = {code,concept},{code},{code}; B = {code,concept},{code,concept},{code}
        # code both 3 of 3/3 -> 1.0; concept both 1, A 1, B 2 -> 2/3; micro 2*4/(4+5)
        self.assertAlmostEqual(t['f1']['micro_f1'], round(8 / 9, 3))
        self.assertEqual(t['f1']['per_code_f1']['concept'], round(2 / 3, 3))
        self.assertIsNone(t['f1']['per_code_f1']['tooling or environment'])
        self.assertAlmostEqual(t['f1']['macro_f1'], round((1 + 2 / 3) / 2, 3))
        # student type: A = TA, TA, TA; B = GG, TA, TA -> TA F1 2*2/(3+2) = 0.8, GG 0/(0+1) = 0.0
        self.assertEqual(s['f1']['per_class_f1']['Task automation'], 0.8)
        self.assertEqual(s['f1']['per_class_f1']['Guided generation'], 0.0)
        self.assertEqual(s['f1']['macro_f1'], 0.4)
        # per-code simple agreement C / max(A, B) (D-179)
        pt, ps, pr = t['per_code_simple_agreement'], s['per_code_simple_agreement'], r['axes']['AI response type']['per_code_simple_agreement']
        self.assertEqual(pt['code'], {'A': 3, 'B': 3, 'C': 3, 'simple_agreement': 1.0})
        self.assertEqual(pt['concept'], {'A': 1, 'B': 2, 'C': 1, 'simple_agreement': 0.5})
        self.assertIsNone(pt['tooling or environment']['simple_agreement'])
        self.assertEqual(ps['Task automation'], {'A': 3, 'B': 2, 'C': 2, 'simple_agreement': round(2 / 3, 3)})
        self.assertEqual(ps['Guided generation'], {'A': 0, 'B': 1, 'C': 0, 'simple_agreement': 0.0})
        # AI response type: rows 1 and 2 only (row 3 truncated in A); both Explain, neither Solve-Provide
        self.assertEqual(pr['Explain'], {'A': 2, 'B': 2, 'C': 2, 'simple_agreement': 1.0})
        self.assertEqual(pr['Solve-Provide (absent)'], {'A': 2, 'B': 2, 'C': 2, 'simple_agreement': 1.0})
        self.assertIsNone(pr['Explain (absent)']['simple_agreement'])
        self.assertTrue(out.with_suffix('.md').exists())
        self.assertIn('C / max(A, B)', out.with_suffix('.md').read_text(encoding='utf-8'))
        self.assertEqual(json.loads(out.read_text(encoding='utf-8'))['axes']['topic']['agree'], 2)
        with self.assertRaises(FileExistsError): agreement(self.alloc, fa, fb, out)
        with self.assertRaises(ValueError): agreement(self.alloc, fa, fb, self.d / 'agr.md')   # would overwrite its own JSON
    def targets(self):
        from test_return_intake import TARGETS
        return TARGETS

if __name__ == '__main__':
    unittest.main()
