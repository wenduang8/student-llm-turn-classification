"""Protect human judgements and missing-data distinctions in the direct interface."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from direct_coding import DIRECT_CODES, DIRECT_VERSION, read_records, validate


def example():
    # Synthetic interface test; not a student annotation or a codebook example.
    return {'schema_version': DIRECT_VERSION, 'coder_id': 'test', 'stage': 'calibration',
            'conversation_id': 'synthetic', 'segment_order': 1,
            'labels': {'student interaction type': 'Task automation', 'topic': ['code'], 'AI delegation': ['ND'],
                       'AI response type': ['Explain']},
            'flags': [], 'notes': '', 'unavailable': {}}


class DirectRecords(unittest.TestCase):
    def test_checks_do_not_change_judgements(self):
        row = example()
        before = copy.deepcopy(row)
        self.assertEqual(validate(row), [])
        self.assertEqual(row, before)

    def test_no_old_field_map_can_replace_direct_level(self):
        row = example()
        row['labels']['student interaction type'] = None
        row['student request'] = 'task'
        self.assertTrue(validate(row))
        self.assertIsNone(row['labels']['student interaction type'])

    def test_blank_is_a_draft_not_a_negative(self):
        row = example()
        row['labels'] = dict.fromkeys(DIRECT_CODES)
        self.assertEqual(validate(row, allow_blank=True), [])
        self.assertEqual(len(validate(row)), 4)

    def test_missing_evidence_has_axis_specific_reason(self):
        row = example()
        row['labels']['topic'] = None
        row['unavailable']['topic'] = 'The referenced material was not captured.'
        self.assertEqual(validate(row), [])
        row['labels']['topic'] = ['code']
        self.assertTrue(validate(row))

    def test_negative_values_cannot_hide_positive_acts(self):
        for axis, bad in [('AI delegation', ['ND', 'offer']),
                          ('AI response type', ['none of the four applies', 'Explain'])]:
            row = example()
            row['labels'][axis] = bad
            self.assertTrue(validate(row))

    def test_both_ai_fields_missing_for_truncated_capture(self):
        row = example()
        row['flags'] = ['truncated capture']
        self.assertEqual(len(validate(row)), 2)
        row['labels']['AI delegation'] = row['labels']['AI response type'] = None
        self.assertEqual(validate(row), [])

    def test_default_l2_does_not_erase_reason(self):
        row = example()
        row['flags'] = ['Task automation default: constraint origin unclear']
        self.assertTrue(validate(row))
        row['notes'] = 'Task is clear; constraint origin is not evidenced.'
        self.assertEqual(validate(row), [])
        row['labels']['student interaction type'] = 'Guided generation'
        self.assertTrue(validate(row))

    def test_nonempty_sets_and_descriptive_type(self):
        for axis, bad in [('topic', 'code'), ('topic', ['both']), ('topic', []),
                          ('topic', ['code', 'code']), ('AI delegation', []),
                          ('student interaction type', 'L2'),
                          ('student interaction type', ['Task automation', 'Guided generation']),
                          ('AI response type', ['Explain', 'Explain'])]:
            row = example()
            row['labels'][axis] = bad
            self.assertTrue(validate(row))

    def test_topic_cooccurrence_is_preserved_without_relabelling(self):
        row = example()
        row['labels']['topic'] = ['code', 'concept', 'tooling or environment', 'conversation management']
        before = copy.deepcopy(row)
        self.assertEqual(validate(row), [])
        self.assertEqual(row, before)

    def test_old_version_is_not_silently_migrated(self):
        row = example()
        row['schema_version'] = 'direct-20260922'
        row['labels'] = {'level': 'L2', 'topic': 'both', 'AI delegation': ['ND'], 'AI response type': ['Explain']}
        before = copy.deepcopy(row)
        self.assertTrue(validate(row))
        self.assertEqual(row, before)

    def test_duplicate_submission_does_not_inflate_count(self):
        row = example()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'records.jsonl'
            p.write_text((json.dumps(row) + '\n') * 2, encoding='utf-8')
            before = p.read_bytes()
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                read_records(p)
            self.assertEqual(p.read_bytes(), before)

    def test_schema_and_container_errors(self):
        row = example()
        row['schema_version'] = 'old-version'
        self.assertTrue(validate(row))
        row['labels'] = None
        self.assertTrue(validate(row))
        self.assertTrue(validate([]))


if __name__ == '__main__':
    unittest.main()
