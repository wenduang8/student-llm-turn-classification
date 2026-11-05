"""Synthetic allocation safeguards; no corpus is loaded and no files are exported."""
import copy
import itertools
import unittest
from unittest.mock import patch

import select_materials as selection


def config():
    return {
        'seed': 'synthetic-fixed-seed',
        'evaluation': {'minimum': 65, 'maximum': 80,
                       'lower_per_stratum': 20, 'upper_per_stratum': 30},
        'first_batch': {'maximum': 50},
        'development': {'minimum': 12, 'maximum': 18,
                        'minimum_students': 6, 'conversation_maximum': 6},
        'training_initial': {'minimum': 16, 'maximum': 22,
                             'conversation_maximum': 6},
        'training_long': {'conversation_minimum': 31, 'conversation_maximum': 60},
        'training_development_combined_maximum': 80,
    }


def rich_pool():
    rows = []
    for i in range(90):
        # The same student's several conversations can enter several candidate
        # queues. Only one may survive across all purposes.
        for n in (2, 5, 20, 35, 60):
            rows.append({'cid': f'SYN_{i:03d}_N{n:02d}', 'student': f'SYN_{i:03d}',
                         'n': n, 'stratum': selection.band(n)})
    return rows


def accept_pending(engine, reviews):
    for _ in range(10):
        pending = engine.advance(reviews)
        if engine.state['completed']:
            return
        if not pending:
            raise AssertionError('Uncompleted engine returned no source-review work')
        reviews.update({p['cid']: {'status': 'pass'} for p in pending})
    raise AssertionError('Synthetic allocation did not finish in ten review rounds')


class SourceReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.engine = selection.Engine(config(), rich_pool())

    def test_source_receipt_matches_inventory_and_normalized_file(self):
        pool=[{'cid':'SYN','n':2,'normalized_path':'data/synthetic-normalized.txt'}]
        review={'status':'pass','reason':'Synthetic check','reviewer_kind':'AI_source_review',
                'normalized_sha256':'synthetic-hash','analysis_rows_checked':2,'scope':'Synthetic only',
                'source_paths':[{'path':'data/synthetic-normalized.txt','sha256':'synthetic-hash'},
                                {'path':'data/synthetic-original.txt','sha256':'synthetic-hash'}]}
        with patch.object(selection,'sha',return_value='synthetic-hash'):
            selection.validate_reviews({'SYN':review},pool)
            for key,value in [('normalized_sha256','wrong-hash'),('analysis_rows_checked',1),
                              ('source_paths',review['source_paths'][:1])]:
                bad=copy.deepcopy(review);bad[key]=value
                with self.subTest(key=key),self.assertRaises(ValueError):
                    selection.validate_reviews({'SYN':bad},pool)

    def test_partial_review_does_not_apply_holds_before_round_is_complete(self):
        pending=self.engine.advance({})
        before=copy.deepcopy(self.engine.state)
        victim=pending[0]['cid']
        result=self.engine.advance({victim:{'status':'hold','reason':'Synthetic hold'}})
        self.assertEqual([p['cid'] for p in result],[p['cid'] for p in pending[1:]])
        self.assertEqual(self.engine.state,before)

    def test_pending_evaluation_blocks_lock_and_all_training_development(self):
        pending = self.engine.advance({})
        self.assertTrue(pending)
        self.assertNotIn('evaluation', self.engine.state['locked'])
        self.assertTrue(all(not ids for phase, ids in self.engine.state['selected'].items()
                            if phase != 'evaluation'))
        before = copy.deepcopy(self.engine.state)
        self.assertEqual(pending, self.engine.advance({}))
        self.assertEqual(before, self.engine.state)

    def test_last_fill_needs_explicit_pass_before_evaluation_locks(self):
        # Lexical rank makes the initial strata total exactly 60, followed by
        # three short conversations to reach 66. The final item is a fill item.
        with patch.object(selection, 'rank', side_effect=lambda seed, purpose, cid: cid):
            pending = self.engine.advance({})
            self.assertEqual(self.engine.total('evaluation'), 66)
            final_fill = pending[-1]
            self.assertEqual(final_fill['stratum'], '1-4')
            reviews = {p['cid']: {'status': 'pass'} for p in pending[:-1]}
            self.assertEqual(self.engine.advance(reviews), [final_fill])
            self.assertNotIn('evaluation', self.engine.state['locked'])
            reviews[final_fill['cid']] = {'status': 'pass'}
            next_pending = self.engine.advance(reviews)
            self.assertIn('evaluation', self.engine.state['locked'])
            self.assertEqual(self.engine.state['phase'], 'development')
            self.assertTrue(next_pending)

    def test_hold_replaces_from_saved_order_and_quarantines_student(self):
        pending = self.engine.advance({})
        victim = next(p for p in pending if p['stratum'] == '20+')
        reviews = {p['cid']: {'status': 'pass'} for p in pending}
        reviews[victim['cid']] = {'status': 'hold', 'reason': 'Synthetic incomplete source'}
        cursor = self.engine.state['cursors']['evaluation|20+']
        used = self.engine.used()
        queue = sorted((p for p in self.engine.pool.values() if p['stratum'] == '20+'),
                       key=lambda p: selection.rank(self.engine.c['seed'], 'evaluation', p['cid']))
        remaining_stratum = self.engine.total('evaluation', '20+') - victim['n']
        remaining_total = self.engine.total('evaluation') - victim['n']
        replacement = next(p for p in queue[cursor:]
                           if p['student'] not in used
                           and remaining_stratum + p['n'] <= 30
                           and remaining_total + p['n'] <= 80)
        next_pending = self.engine.advance(reviews)
        self.assertIn(replacement['cid'], [p['cid'] for p in next_pending])
        self.assertNotIn(victim['cid'], self.engine.state['selected']['evaluation'])
        self.assertIn(victim['student'], self.engine.state['held_students'])
        self.assertNotIn('evaluation', self.engine.state['locked'])
        self.assertGreater(self.engine.state['cursors']['evaluation|20+'], cursor)
        accept_pending(self.engine, reviews)
        selected_students = {self.engine.pool[cid]['student']
                             for cids in self.engine.state['selected'].values() for cid in cids}
        self.assertNotIn(victim['student'], selected_students)

    def test_locked_review_cannot_be_revoked_or_lost_silently(self):
        reviews = {}
        accept_pending(self.engine, reviews)
        for phase in self.engine.state['locked']:
            cid = self.engine.state['selected'][phase][0]
            for replacement in ({'status': 'hold', 'reason': 'new source evidence'}, None):
                changed = copy.deepcopy(reviews)
                if replacement is None:
                    del changed[cid]
                else:
                    changed[cid] = replacement
                with self.subTest(phase=phase, review=replacement):
                    with self.assertRaises(ValueError):
                        self.engine.advance(changed)

    def test_export_rejects_pending_or_revoked_reviews_before_writing(self):
        self.engine.advance({})
        with patch.object(selection.Path, 'mkdir', side_effect=AssertionError('Export wrote before its gate')):
            with self.assertRaises(ValueError):
                selection.export_materials(self.engine, {})
        reviews = {}
        accept_pending(self.engine, reviews)
        cid = self.engine.state['selected']['evaluation'][0]
        reviews[cid] = {'status': 'hold', 'reason': 'new source evidence'}
        with patch.object(selection.Path, 'mkdir', side_effect=AssertionError('Export wrote before its gate')):
            with self.assertRaises(ValueError):
                selection.export_materials(self.engine, reviews)


class WholeConversationAndBudgetTests(unittest.TestCase):
    def test_first_batch_is_a_deterministic_whole_subset_with_three_strata(self):
        lengths = [45, 35, 30, 19, 15, 10, 5, 4, 4, 3, 2, 1]
        roster = [{'cid': f'SYN_C{i}', 'student': f'SYN_S{i}', 'n': n, 'stratum': selection.band(n)}
                  for i, n in enumerate(lengths)]
        selected = selection.first_batch(roster, 'fixed', cap=150)
        self.assertEqual(selected, selection.first_batch(list(reversed(roster)), 'fixed', cap=150))
        self.assertTrue(selected <= {r['cid'] for r in roster})
        chosen = [r for r in roster if r['cid'] in selected]
        self.assertEqual(sum(r['n'] for r in chosen), 150)
        self.assertEqual({r['stratum'] for r in chosen}, set(selection.STRATA))

    def test_all_students_are_disjoint_and_budgets_apply_after_review(self):
        engine = selection.Engine(config(), rich_pool())
        reviews = {}
        accept_pending(engine, reviews)
        groups = {phase: {engine.pool[cid]['student'] for cid in cids}
                  for phase, cids in engine.state['selected'].items()}
        for a, b in itertools.combinations(groups, 2):
            self.assertFalse(groups[a] & groups[b], (a, b))
        all_cids = [cid for cids in engine.state['selected'].values() for cid in cids]
        self.assertEqual(len(all_cids), len({engine.pool[cid]['student'] for cid in all_cids}))
        self.assertTrue(all(reviews[cid]['status'] == 'pass' for cid in all_cids))
        development = engine.state['selected']['development']
        self.assertGreaterEqual(len(development), 6)
        self.assertGreaterEqual(engine.total('development'), 12)
        self.assertLessEqual(engine.total('development'), 18)
        self.assertTrue(all(engine.pool[cid]['n'] <= 6 for cid in development))
        self.assertLessEqual(sum(engine.total(p) for p in groups if p != 'evaluation'), 80)
        self.assertEqual(len(engine.state['selected']['training_long']), 1)
        self.assertLessEqual(engine.total('training_long'),
                             80 - engine.total('development') - engine.total('training_initial'))

    def test_engine_continuation_survives_serialized_state_without_redraw(self):
        engine = selection.Engine(config(), rich_pool())
        pending = engine.advance({})
        reviews = {p['cid']: {'status': 'pass'} for p in pending}
        restored = selection.Engine(config(), list(reversed(rich_pool())), copy.deepcopy(engine.state))
        self.assertEqual(engine.advance(reviews), restored.advance(reviews))
        self.assertEqual(engine.state, restored.state)


if __name__ == '__main__':
    unittest.main()
