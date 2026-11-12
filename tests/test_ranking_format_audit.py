import json
import unittest

from llm_coherence.analysis.audit_full_ladder_rankings import formatting_only, remove_line_comments
from llm_coherence.experiments import full_ladder_ranking as ranking


class FormatAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.task = ranking.prepare(ranking.DEFAULT_INPUT, ["gpt-54-nano"], limit=1)[0][0]

    def test_bare_and_bracketed_labels_preserve_given_order(self):
        order = list("GFEDCBA")
        self.assertEqual(formatting_only('{"ranking":[G,F,E,D,C,B,A]}', self.task), order)
        self.assertEqual(formatting_only(json.dumps({"ranking": [f"[{s}]" for s in order]}), self.task), order)

    def test_comments_removed_without_touching_quoted_urls(self):
        self.assertEqual(remove_line_comments('{"url":"https://example.com"}// ignore'), '{"url":"https://example.com"}')
        raw = '{"ranking":["A",// first\n"B","C","D","E","F","G"]}'
        self.assertEqual(formatting_only(raw, self.task), list("ABCDEFG"))

    def test_statement_labels_require_matching_stimuli(self):
        entries = [f"[{label}] {self.task.statements[index]}" for label,index in zip(self.task.labels,self.task.shuffle_indices)]
        self.assertEqual(formatting_only(json.dumps({"ranking":entries}),self.task),self.task.labels)
        entries[0] = "[A] a different statement"
        self.assertIsNone(formatting_only(json.dumps({"ranking":entries}),self.task))

    def test_no_guessing_duplicates_missing_labels_or_prose(self):
        for raw in ['{"ranking":[A,B,C,D,E,F,F]}','{"ranking":[A,B,C]}',
                    'A seems best, then B then C then D then E then F then G',
                    'I decline to rank these.']:
            self.assertIsNone(formatting_only(raw,self.task))


if __name__ == "__main__":
    unittest.main()
