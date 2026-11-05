"""Synthetic D-87 safeguards. Never reads corpus, real reviews or old verdicts."""
import ast
import copy
import csv
import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import select_materials as old
import select_materials_v2 as selection
from test_select_materials import config as old_config, rich_pool


class SyntheticDirectory:
    """Use normal inherited Windows permissions instead of mkdtemp's 0700 ACL."""
    def __init__(self, prefix):
        self.base = (Path(__file__).resolve().parents[1]
                     / 'data/maintenance/20261006-source-recheck/synthetic-test-tmp').resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base / (prefix + uuid.uuid4().hex)
        self.path.mkdir()
        self.name = str(self.path)

    def cleanup(self):
        target = self.path.resolve()
        if target.parent != self.base or not target.name.startswith('synthetic-'):
            raise ValueError('Unsafe synthetic cleanup target')
        if target.exists():
            shutil.rmtree(target)


def config():
    value = old_config()
    value.update(protocol='synthetic-v2', schema_version='direct-synthetic',
                 review_version=selection.VERSION, rules_path='rules.md', rules_sha256='synthetic')
    value['first_batch']['minimum_goal'] = 40
    return value


def finish(engine, reviews):
    for _ in range(len(engine.pool) + len(selection.PHASES) + 1):
        pending = selection.advance_known(engine, reviews)
        if engine.state['completed']:
            return
        reviews.update({p['cid']: {'status': 'pass'} for p in pending})
    raise AssertionError('Synthetic queue did not finish')


class QueueTests(unittest.TestCase):
    def test_all_pass_matches_v1_rosters_cursors_and_event_order(self):
        for seed in ['synthetic-0', 'synthetic-1', 'synthetic-17', '20261005']:
            cfg = config()
            cfg['seed'] = seed
            original = old.Engine(cfg, rich_pool())
            revised = selection.Engine(cfg, list(reversed(rich_pool())))
            reviews = {}
            while not original.state['completed']:
                pending = original.advance(reviews)
                reviews.update({p['cid']: {'status': 'pass'} for p in pending})
            # The new review boundary pauses at each unknown source. Compare
            # the final all-pass result, not the old multi-candidate batch size.
            finish(revised, {})
            self.assertEqual(original.state['selected'], revised.state['selected'])
            self.assertEqual(original.state['cursors'], revised.state['cursors'])
            self.assertEqual(original.state['locked'], revised.state['locked'])
            def events(engine):
                return [{**e, 'reason': e.get('reason', '').replace('student already selected or held', 'student already selected')}
                        for e in engine.state['events']]
            self.assertEqual(events(original), events(revised))
            self.assertEqual(original.state['first_batch'], revised.state['first_batch'])
            known = selection.Engine(cfg, rich_pool())
            selection.advance_known(known, {p['cid']: {'status': 'pass'} for p in rich_pool()})
            self.assertEqual(revised.state, known.state)

    def test_only_failed_conversation_is_held_same_student_can_replace(self):
        pool = [{'cid': 'a', 'student': 'S', 'n': 20, 'stratum': '20+'},
                {'cid': 'b', 'student': 'S', 'n': 20, 'stratum': '20+'},
                {'cid': 'c', 'student': 'T', 'n': 20, 'stratum': '20+'}]
        engine = selection.Engine(config(), pool)
        with patch.object(selection, 'rank', side_effect=lambda seed, purpose, cid: cid):
            self.assertTrue(engine.next_candidate('evaluation', '20+', 1, 100, 80, 30))
            engine.state['selected']['evaluation'].remove('a')
            engine.state['held_conversations'].append('a')
            self.assertTrue(engine.next_candidate('evaluation', '20+', 1, 100, 80, 30))
            self.assertEqual(engine.state['selected']['evaluation'], ['b'])
            self.assertNotIn('held_students', engine.state)
            # The same failed CID is not proposed in a later purpose.
            self.assertTrue(engine.next_candidate('training_initial', None, 1, 100, 80))
            self.assertEqual(engine.state['selected']['training_initial'], ['c'])

    def test_hold_through_advance_does_not_quarantine_student(self):
        engine = selection.Engine(config(), rich_pool())
        pending = engine.advance({})
        victim = pending[0]
        reviews = {p['cid']: {'status': 'pass'} for p in pending}
        reviews[victim['cid']] = {'status': 'hold', 'reason': 'Synthetic defect'}
        engine.advance(reviews)
        self.assertIn(victim['cid'], engine.state['held_conversations'])
        self.assertNotIn(victim['cid'], engine.state['selected']['evaluation'])
        # A temporarily used student is only used when another conversation is selected.
        used_cids = [c for ids in engine.state['selected'].values() for c in ids]
        self.assertEqual(victim['student'] in engine.used(),
                         any(engine.pool[c]['student'] == victim['student'] for c in used_cids))

    def test_partial_review_keeps_state_and_blocks_later_phases(self):
        engine = selection.Engine(config(), rich_pool())
        # Defensive gate for any saved multi-item round. Production now pauses
        # after its first unknown candidate and cannot create this situation.
        engine.next_candidate('evaluation', '20+', 20, 20, 80, 50)
        engine.next_candidate('evaluation', '20+', 20, 20, 80, 50)
        pending = engine.advance({})
        self.assertEqual(len(pending), 2)
        before = copy.deepcopy(engine.state)
        engine.advance({pending[0]['cid']: {'status': 'hold'}})
        self.assertEqual(before, engine.state)
        self.assertEqual(engine.state['locked'], [])
        self.assertTrue(all(not engine.state['selected'][p] for p in selection.PHASES[1:]))

    def test_final_fill_accepts_annotations_but_waits_for_full_review(self):
        engine = selection.Engine(config(), rich_pool())
        with patch.object(selection, 'rank', side_effect=lambda seed, purpose, cid: cid):
            pending = engine.advance({})
            reviews = {}
            while engine.total('evaluation') < 66:
                reviews.update({p['cid']: {'status': 'pass'} for p in pending})
                pending = engine.advance(reviews)
            fill = pending[-1]
            self.assertEqual(engine.total('evaluation'), 66)
            self.assertEqual(fill['stratum'], '1-4')
            self.assertEqual(engine.advance(reviews), [fill])
            self.assertNotIn('evaluation', engine.state['locked'])
            reviews[fill['cid']] = {'status': 'pass_with_annotations', 'annotations': ['synthetic']}
            engine.advance(reviews)
            self.assertIn('evaluation', engine.state['locked'])

    def test_locked_annotations_cannot_change_even_if_status_stays_pass(self):
        engine = selection.Engine(config(), rich_pool())
        reviews = {}
        finish(engine, reviews)
        cid = engine.state['selected']['evaluation'][0]
        reviews[cid]['annotations'] = ['new annotation']
        with self.assertRaises(ValueError):
            engine.advance(reviews)

    def test_old_state_is_rejected_new_state_resumes_without_redraw(self):
        cfg = config()
        with self.assertRaises(ValueError):
            selection.Engine(cfg, rich_pool(), old.Engine(cfg, rich_pool()).state)
        engine = selection.Engine(cfg, rich_pool())
        pending = engine.advance({})
        reviews = {p['cid']: {'status': 'pass'} for p in pending}
        restored = selection.Engine(cfg, list(reversed(rich_pool())), copy.deepcopy(engine.state))
        self.assertEqual(engine.advance(reviews), restored.advance(reviews))
        self.assertEqual(engine.state, restored.state)

    def test_complete_allocation_preserves_student_isolation_and_caps(self):
        engine = selection.Engine(config(), rich_pool())
        reviews = {}
        finish(engine, reviews)
        all_cids = [c for ids in engine.state['selected'].values() for c in ids]
        self.assertEqual(len(all_cids), len({engine.pool[c]['student'] for c in all_cids}))
        self.assertGreaterEqual(len(engine.state['selected']['development']), 6)
        self.assertLessEqual(engine.total('development'), 18)
        self.assertLessEqual(sum(engine.total(p) for p in selection.PHASES[1:]), 80)
        self.assertEqual(len(engine.state['selected']['training_long']), 1)
        self.assertLessEqual(sum(engine.pool[c]['n'] for c in engine.state['first_batch']), 50)
        self.assertEqual({engine.pool[c]['stratum'] for c in engine.state['first_batch']}, set(selection.STRATA))


class SourceOccupancyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = config()
        self.cfg['evaluation'] = {'minimum': 2, 'maximum': 2, 'lower_per_stratum': 0, 'upper_per_stratum': 2}
        self.cfg['first_batch'] = {'minimum_goal': 2, 'maximum': 2}
        self.cfg['development'] = {'minimum': 1, 'maximum': 1, 'minimum_students': 1, 'conversation_maximum': 1}
        self.cfg['training_initial'] = {'minimum': 1, 'maximum': 1, 'conversation_maximum': 1}
        self.cfg['training_long'] = {'conversation_minimum': 1, 'conversation_maximum': 1}
        self.cfg['training_development_combined_maximum'] = 3
        self.pool = [{'cid': cid, 'student': student, 'n': 1, 'stratum': '1-4'}
                     for cid, student in [('a', 'S'), ('b', 'S'), ('c', 'T'), ('d', 'U'),
                                          ('e', 'V'), ('f', 'W'), ('g', 'X')]]
        self.ranking = patch.object(selection, 'rank', side_effect=lambda seed, purpose, cid: cid)
        self.ranking.start()
        self.addCleanup(self.ranking.stop)

    def test_unknown_pauses_before_same_student_or_budget_can_be_skipped(self):
        engine = selection.Engine(self.cfg, self.pool)
        pending = engine.advance({})
        self.assertEqual([p['cid'] for p in pending], ['a'])
        self.assertEqual(engine.state['selected']['evaluation'], ['a'])
        self.assertEqual(engine.state['cursors'], {'evaluation|1-4': 1})
        self.assertFalse(any(e['action'] == 'skip' for e in engine.state['events']))
        before = copy.deepcopy(engine.state)
        self.assertEqual(engine.advance({}), pending)
        self.assertEqual(engine.state, before)

    def test_hold_allows_b_before_c_and_known_hold_has_identical_result(self):
        engine = selection.Engine(self.cfg, self.pool)
        engine.advance({})
        reviews = {'a': {'status': 'hold', 'reason': 'Synthetic source defect'}}
        pending = engine.advance(reviews)
        self.assertEqual([p['cid'] for p in pending], ['b'])
        self.assertEqual(engine.state['selected']['evaluation'], ['b'])
        self.assertEqual(engine.state['cursors']['evaluation|1-4'], 2)
        self.assertEqual(engine.state['held_conversations'], ['a'])
        known = selection.Engine(self.cfg, self.pool)
        self.assertEqual(known.advance(reviews), pending)
        self.assertEqual(known.state, engine.state)

    def test_pass_rejects_b_and_locked_student_stays_isolated_across_purposes(self):
        engine = selection.Engine(self.cfg, self.pool)
        engine.advance({})
        pending = engine.advance({'a': {'status': 'pass'}})
        self.assertEqual([p['cid'] for p in pending], ['c'])
        self.assertEqual(engine.state['selected']['evaluation'], ['a', 'c'])
        reviews = {p['cid']: {'status': 'pass'} for p in self.pool}
        selection.advance_known(engine, reviews)
        self.assertTrue(engine.state['completed'])
        self.assertEqual(engine.state['selected']['development'], ['d'])
        self.assertNotIn('b', [cid for ids in engine.state['selected'].values() for cid in ids])
        locked = copy.deepcopy(engine.state)
        with self.assertRaises(ValueError):
            engine.advance({**reviews, 'a': {'status': 'hold'}})
        self.assertEqual(engine.state, locked)

    def test_cross_stratum_pending_hold_does_not_exclude_short_b(self):
        cfg = copy.deepcopy(self.cfg)
        cfg['evaluation'] = {'minimum': 26, 'maximum': 30, 'lower_per_stratum': 1, 'upper_per_stratum': 25}
        pool = [{'cid': 'a', 'student': 'S', 'n': 20, 'stratum': '20+'},
                {'cid': 'b', 'student': 'S', 'n': 1, 'stratum': '1-4'},
                {'cid': 'c', 'student': 'T', 'n': 1, 'stratum': '1-4'},
                {'cid': 'd', 'student': 'U', 'n': 20, 'stratum': '20+'},
                {'cid': 'e', 'student': 'V', 'n': 5, 'stratum': '5-19'}]
        engine = selection.Engine(cfg, pool)
        self.assertEqual([p['cid'] for p in engine.advance({})], ['a'])
        self.assertNotIn('evaluation|1-4', engine.state['cursors'])
        pending = engine.advance({'a': {'status': 'hold'}, 'd': {'status': 'pass'}, 'e': {'status': 'pass'}})
        self.assertEqual([p['cid'] for p in pending], ['b'])
        self.assertEqual(engine.state['selected']['evaluation'], ['d', 'e', 'b'])
        self.assertFalse(any(event['cid'] == 'b' and event['action'] == 'skip'
                             for event in engine.state['events'] if 'cid' in event))

    def test_saved_unknown_resumes_consistently_and_old_policy_is_rejected(self):
        engine = selection.Engine(self.cfg, self.pool)
        engine.advance({})
        saved = json.loads(json.dumps(engine.state))
        resumed = selection.Engine(self.cfg, list(reversed(self.pool)), saved)
        reviews = {'a': {'status': 'hold'}}
        self.assertEqual(engine.advance(reviews), resumed.advance(reviews))
        self.assertEqual(engine.state, resumed.state)
        del saved['queue_policy']
        with self.assertRaises(ValueError):
            selection.Engine(self.cfg, self.pool, saved)

    def test_true_budget_skip_is_never_revisited_after_capacity_frees(self):
        pool = [{'cid': 'a', 'student': 'S', 'n': 1, 'stratum': '1-4'},
                {'cid': 'b', 'student': 'T', 'n': 2, 'stratum': '1-4'},
                {'cid': 'c', 'student': 'U', 'n': 1, 'stratum': '1-4'},
                {'cid': 'd', 'student': 'V', 'n': 1, 'stratum': '1-4'}]
        engine = selection.Engine(self.cfg, pool)
        engine.next_candidate('evaluation', '1-4', 1, 4, 2, 2)
        engine.next_candidate('evaluation', '1-4', 1, 4, 2, 2)
        self.assertEqual(engine.state['selected']['evaluation'], ['a', 'c'])
        engine.state['selected']['evaluation'].clear()
        engine.state['held_conversations'] = ['a', 'c']
        engine.next_candidate('evaluation', '1-4', 1, 4, 2, 2)
        self.assertEqual(engine.state['selected']['evaluation'], ['d'])
        skips = [e for e in engine.state['events'] if e['action'] == 'skip']
        self.assertEqual([(e['cid'], e['reason']) for e in skips], [('b', 'overall cap')])


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = SyntheticDirectory(prefix='synthetic-allocation-v2-')
        self.root = Path(self.tmp.name)
        (self.root / 'rules.md').write_text('synthetic rules', encoding='utf-8')
        self.cfg = config()
        self.cfg['rules_sha256'] = selection.sha(self.root / 'rules.md')
        for name in ['normal.txt', 'original.txt']:
            (self.root / name).write_text('synthetic source', encoding='utf-8')
        self.pool = [{'cid': 'SYN', 'student': 'S', 'n': 2, 'stratum': '1-4',
                      'normalized_path': 'normal.txt', 'assignment': 'P2'}]
        self.rows = {'SYN': [{'Conversation ID': 'SYN', 'Segment Order': '1',
                             'Student Prompt': '学生🙂\n<question>', 'LLM Response': 'Actual <answer>\n用途说明🙂'},
                            {'Conversation ID': 'SYN', 'Segment Order': '2',
                             'Student Prompt': 'Next?', 'LLM Response': 'unfinished { code'}]}
        self.receipt = {'status': 'pass', 'reviewer_kind': 'AI_source_review_v2',
                        'rules_version': selection.VERSION, 'rules_sha256': self.cfg['rules_sha256'],
                        'reason': 'Synthetic source checked', 'issue_codes': [], 'analysis_rows_checked': 2,
                        'normalized_sha256': selection.sha(self.root / 'normal.txt'),
                        'source_paths': [{'path': name, 'sha256': selection.sha(self.root / name)}
                                         for name in ['normal.txt', 'original.txt']],
                        'scope': 'Synthetic complete local source', 'annotations': [],
                        'truncated_orders': [], 'sensitive_content_present': False}

    def tearDown(self):
        self.tmp.cleanup()

    def annotated(self):
        receipt = copy.deepcopy(self.receipt)
        text = self.rows['SYN'][0]['LLM Response']
        start = text.index('用途')
        receipt.update(status='pass_with_annotations', issue_codes=['M1', 'D32'], truncated_orders=[2],
                       annotations=[{'segment_order': 1, 'field': 'LLM Response', 'start': start, 'end': len(text),
                                     'text': text[start:], 'field_sha256': selection.text_sha(text),
                                     'kind': 'M1', 'basis': 'Synthetic original submission column'}])
        return receipt

    def validate(self, receipt):
        selection.validate_reviews({'SYN': receipt}, self.pool, self.rows, self.cfg, self.root)

    def test_valid_plain_annotated_d32_and_hold(self):
        self.validate(self.receipt)
        self.validate(self.annotated())
        held = copy.deepcopy(self.receipt)
        held.update(status='hold', issue_codes=['H1'])
        self.validate(held)

    def test_annotation_wrong_hash_span_field_or_overlap_rejected(self):
        for key, value in [('field_sha256', 'wrong'), ('start', True), ('end', 1000),
                           ('text', 'wrong'), ('field', 'topic'), ('segment_order', 3)]:
            receipt = self.annotated()
            receipt['annotations'][0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(receipt)
        receipt = self.annotated()
        receipt['annotations'].append(copy.deepcopy(receipt['annotations'][0]))
        with self.assertRaises(ValueError):
            self.validate(receipt)

    def test_status_issue_types_and_d32_orders_rejected(self):
        for key, value in [('status', 'pass'), ('truncated_orders', [True]), ('truncated_orders', [3]),
                           ('truncated_orders', [2, 2]), ('issue_codes', ['M1']),
                           ('analysis_rows_checked', True), ('sensitive_content_present', 'false')]:
            receipt = self.annotated()
            receipt[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.validate(receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt['issue_codes'] = ['H2']
        with self.assertRaises(ValueError):
            self.validate(receipt)

    def test_source_hash_and_path_traversal_rejected(self):
        receipt = copy.deepcopy(self.receipt)
        receipt['source_paths'][1]['path'] = '../outside.txt'
        with self.assertRaises(ValueError):
            self.validate(receipt)
        (self.root / 'original.txt').write_text('changed', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.validate(self.receipt)

    def test_derive_preserves_raw_separates_notes_and_flags_only_d32_order(self):
        receipt = self.annotated()
        self.validate(receipt)
        original = copy.deepcopy(self.rows)
        first, trace = selection.derive_record(self.rows['SYN'][0], receipt)
        last, _ = selection.derive_record(self.rows['SYN'][1], receipt)
        self.assertEqual(first['ai_response'], 'Actual <answer>\n')
        self.assertEqual(first['non_dialogue_notes'][0]['text'], '用途说明🙂')
        self.assertEqual(first['student_prompt'], original['SYN'][0]['Student Prompt'])
        self.assertFalse(first['availability']['ai_fields_unavailable'])
        self.assertTrue(last['availability']['ai_fields_unavailable'])
        self.assertEqual(last['ai_response'], original['SYN'][1]['LLM Response'])
        self.assertNotIn('labels', first)
        self.assertEqual(self.rows, original)
        self.assertEqual(trace[1]['raw_field_sha256'], selection.text_sha(original['SYN'][0]['LLM Response']))

    def test_shared_urls_hidden_with_trace_and_bad_callback_rejected(self):
        row = copy.deepcopy(self.rows['SYN'][0])
        row['Student Prompt'] = 'Question https://chatgpt.com/share/private-id'
        result, trace = selection.derive_record(row, self.receipt)
        self.assertNotIn('https://chatgpt.com/share/', result['student_prompt'])
        self.assertEqual(trace[0]['replacements'][0]['text'], 'https://chatgpt.com/share/private-id')
        with self.assertRaises(ValueError):
            selection.derive_record(row, self.receipt, lambda text, context: {'text': 'changed', 'replacements': []})

    def test_reader_escapes_html_and_has_no_original_source_links(self):
        receipt = self.annotated()
        rows = [selection.derive_record(row, receipt)[0] for row in self.rows['SYN']]
        out = self.root / 'render'
        out.mkdir()
        (out / 'evaluation_first').mkdir()
        summaries = [{'group': 'evaluation_first', 'interactions': 2, 'students': 1}]
        result = {'evaluation_total': 2, 'development_training_total': 0,
                  'remaining_training_budget': 200, 'shortfalls': {'evaluation': 198}}
        selection.render_readers(out, {'evaluation_first': rows}, summaries, result)
        text = (out / 'conversations/SYN.html').read_text(encoding='utf-8')
        self.assertIn('&lt;answer&gt;', text)
        self.assertIn('<aside>', text)
        self.assertIn('D32', text)
        self.assertNotIn('normalized', text)
        self.assertNotIn('original.txt', text)
        self.assertNotIn('<question>', text)


class LockTests(unittest.TestCase):
    def setUp(self):
        self.tmp = SyntheticDirectory(prefix='synthetic-review-lock-')
        self.root = Path(self.tmp.name)
        self.task = self.root / 'task'
        self.task.mkdir()
        self.cfg = config()
        selection.write_json(self.task / 'selection-config.json', self.cfg)
        self.reviews = {'A': {'status': 'pass'}, 'B': {'status': 'hold'}}
        selection.write_json(self.task / 'targets.json', ['A', 'B'])
        selection.write_json(self.task / 'locked.json', self.reviews)
        self.lock = {'lock_version': 'original-298-lock-v1', 'rules_version': selection.VERSION,
                     'rules_sha256': self.cfg['rules_sha256'],
                     'config_sha256': selection.sha(self.task / 'selection-config.json'),
                     'original_targets_path': 'task/targets.json',
                     'original_targets_sha256': selection.sha(self.task / 'targets.json'),
                     'locked_reviews_path': 'task/locked.json',
                     'locked_reviews_sha256': selection.sha(self.task / 'locked.json'), 'target_count': 2,
                     'receipt_sha256': {k: selection.canonical_sha(v) for k, v in self.reviews.items()}}

    def tearDown(self):
        self.tmp.cleanup()

    def validate(self, reviews=None):
        return selection.validate_original_lock(self.cfg, self.reviews if reviews is None else reviews,
                                                self.root, self.task, expected_count=2)

    def test_complete_lock_allows_additional_new_reviews(self):
        selection.write_json(self.task / 'original-298-lock.json', self.lock)
        self.assertEqual(self.validate({**self.reviews, 'C': {'status': 'pass'}}),
                         selection.sha(self.task / 'original-298-lock.json'))

    def test_missing_lock_missing_receipt_or_modified_locked_subset_stops(self):
        with self.assertRaises(ValueError):
            self.validate()
        selection.write_json(self.task / 'original-298-lock.json', self.lock)
        with self.assertRaises(ValueError):
            self.validate({'A': self.reviews['A']})
        with self.assertRaises(ValueError):
            self.validate({**self.reviews, 'A': {'status': 'pass', 'changed': True}})

    def test_incomplete_lock_or_changed_frozen_file_stops(self):
        broken = copy.deepcopy(self.lock)
        del broken['receipt_sha256']['B']
        selection.write_json(self.task / 'original-298-lock.json', broken)
        with self.assertRaises(ValueError):
            self.validate()
        selection.write_json(self.task / 'original-298-lock.json', self.lock)
        selection.write_json(self.task / 'locked.json', {'A': {'status': 'pass'}})
        with self.assertRaises(ValueError):
            self.validate()

    def test_export_gate_stops_before_creating_output(self):
        engine = selection.Engine(config(), rich_pool())
        with patch.object(selection.Path, 'mkdir', side_effect=AssertionError('Unexpected write')):
            with self.assertRaises(ValueError):
                selection.export_materials(engine, {})
        reviews = {}
        finish(engine, reviews)
        with patch.object(selection.Path, 'mkdir', side_effect=AssertionError('Unexpected write')):
            with self.assertRaises(ValueError):
                selection.export_materials(engine, reviews, root=self.root, task=self.task)

    def test_full_synthetic_298_lock_to_export_and_no_overwrite(self):
        # A real 298-receipt lock and real synthetic files, without mocking gates.
        cfg = config()
        cfg['evaluation'] = {'minimum': 3, 'maximum': 3, 'lower_per_stratum': 0, 'upper_per_stratum': 3}
        cfg['first_batch'] = {'minimum_goal': 3, 'maximum': 3}
        cfg['development'] = {'minimum': 2, 'maximum': 2, 'minimum_students': 2, 'conversation_maximum': 1}
        cfg['training_initial'] = {'minimum': 2, 'maximum': 2, 'conversation_maximum': 1}
        cfg['training_long'] = {'conversation_minimum': 1, 'conversation_maximum': 1}
        cfg['training_development_combined_maximum'] = 5
        for name in ['rules.md', 'normal.txt', 'original.txt']:
            (self.root / name).write_text('Synthetic only', encoding='utf-8')
        cfg['rules_sha256'] = selection.sha(self.root / 'rules.md')
        selection.write_json(self.task / 'selection-config.json', cfg)
        reply = 'AI visible unfinished {\nExternal note https://chatgpt.com/share/synthetic'
        start = reply.index('External note')
        pool, rows, reviews = [], [], {}
        for i in range(298):
            cid = f'SYN_{i:03}'
            pool.append({'cid': cid, 'student': f'S{i:03}', 'n': 1, 'stratum': '1-4',
                         'assignment': 'P2', 'normalized_path': 'normal.txt'})
            rows.append({'Conversation ID': cid, 'Segment Order': '1', 'Student Prompt': 'Question <x>',
                         'LLM Response': reply, 'Source File': 'original.txt'})
            reviews[cid] = {'status': 'pass_with_annotations', 'reviewer_kind': 'AI_source_review_v2',
                            'rules_version': selection.VERSION, 'rules_sha256': cfg['rules_sha256'],
                            'reason': 'Synthetic complete review', 'scope': 'Synthetic only',
                            'issue_codes': ['M1', 'D32'], 'analysis_rows_checked': 1,
                            'normalized_sha256': selection.sha(self.root / 'normal.txt'),
                            'source_paths': [{'path': name, 'sha256': selection.sha(self.root / name)}
                                             for name in ['normal.txt', 'original.txt']],
                            'annotations': [{'segment_order': 1, 'field': 'LLM Response', 'start': start,
                                             'end': len(reply), 'text': reply[start:],
                                             'field_sha256': selection.text_sha(reply), 'kind': 'M1',
                                             'basis': 'Synthetic non-dialogue column'}],
                            'truncated_orders': [1], 'sensitive_content_present': False}
        source = self.root / 'data/processed/transcripts_analysis.csv'
        source.parent.mkdir(parents=True)
        with source.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        selection.write_json(self.task / 'targets.json', list(reviews))
        selection.write_json(self.task / 'locked.json', reviews)
        lock = {**self.lock, 'target_count': 298, 'config_sha256': selection.sha(self.task / 'selection-config.json'),
                'rules_sha256': cfg['rules_sha256'], 'original_targets_sha256': selection.sha(self.task / 'targets.json'),
                'locked_reviews_sha256': selection.sha(self.task / 'locked.json'),
                'receipt_sha256': {k: selection.canonical_sha(v) for k, v in reviews.items()}}
        selection.write_json(self.task / 'original-298-lock.json', lock)
        engine = selection.Engine(cfg, pool)
        selection.advance_known(engine, reviews)
        engine.state['original_review_lock_sha256'] = selection.sha(self.task / 'original-298-lock.json')
        out = self.root / 'output'
        out.mkdir()  # An already-created EMPTY destination is safe.
        before = selection.sha(source)
        result = selection.export_materials(engine, reviews, root=self.root, task=self.task, out=out)
        self.assertEqual(result['evaluation_total'], 3)
        self.assertEqual(result['development_training_total'], 5)
        self.assertEqual(before, selection.sha(source))
        raw = [json.loads(line) for line in (out / 'private/evaluation_first/raw-interactions.jsonl').read_text(encoding='utf-8').splitlines()]
        derived = [json.loads(line) for line in (out / 'evaluation_first/interactions.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(raw[0]['ai_response'], reply)
        self.assertEqual(derived[0]['ai_response'], 'AI visible unfinished {\n')
        self.assertTrue(derived[0]['availability']['ai_fields_unavailable'])
        self.assertNotIn('https://chatgpt.com/share/', json.dumps(derived))
        self.assertFalse(result['coding_packet_released'])
        page = (out / 'conversations' / f'{derived[0]["conversation_id"]}.html').read_text(encoding='utf-8')
        self.assertIn('&lt;x&gt;', page)
        self.assertNotIn('original.txt', page)
        marker = out / 'researcher-progress.txt'
        marker.write_text('Do not overwrite', encoding='utf-8')
        with self.assertRaises(ValueError):
            selection.export_materials(engine, reviews, root=self.root, task=self.task, out=out)
        self.assertEqual(marker.read_text(encoding='utf-8'), 'Do not overwrite')


if __name__ == '__main__':
    unittest.main()
