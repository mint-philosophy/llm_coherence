"""Preserve the same stimulus text on UTF-8 and Windows-default systems."""

import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from llm_coherence.experiments.ladder_statement_pair import experiment_runner_tradeoff as runner


class ComparisonInputEncodingTests(unittest.TestCase):
    def test_utf8_stimuli_survive_a_cp1252_default_encoding(self):
        comparisons = [{
            'outcome_a': {'text': 'A child’s wellbeing improves—at home and at the café.'},
            'outcome_b': {'text': 'The child receives no additional support.'},
        }]
        real_open = builtins.open

        def windows_default_open(file, mode='r', *args, **kwargs):
            if 'b' not in mode and kwargs.get('encoding') is None:
                kwargs['encoding'] = 'cp1252'
            return real_open(file, mode, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'unicode_stimuli_comparisons.json'
            path.write_text(json.dumps({'comparisons': comparisons}, ensure_ascii=False), encoding='utf-8')
            with patch.object(runner, 'open', side_effect=windows_default_open, create=True):
                loaded = runner.load_comparisons(root, 'unicode_stimuli')
            self.assertEqual(loaded, comparisons)
            prompt = runner.build_prompt(loaded[0]['outcome_a']['text'], loaded[0]['outcome_b']['text'])
            self.assertIn(comparisons[0]['outcome_a']['text'], prompt)


if __name__ == '__main__':
    unittest.main()
