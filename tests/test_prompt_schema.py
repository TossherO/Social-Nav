"""Static contract tests; dynamic IDs/causality require a deployment validator."""

import json
import re
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((ROOT / "decision/vlm/configs/schemas/social_nav_prompt.json").read_text())


def validator(mode):
    return Draft202012Validator({
        "$schema": SCHEMA["$schema"],
        "$defs": SCHEMA["$defs"],
        "$ref": f"#/$defs/{mode}",
    })


class PromptSchemaTests(unittest.TestCase):
    def test_schema_valid(self):
        Draft202012Validator.check_schema(SCHEMA)

    def test_legacy_appendix_examples_remain_valid(self):
        document = (ROOT / "docs/旧版_基于VLM的长程社交导航.md").read_text().split("## 附录 A：主接口与 Prompt 契约", 1)[1]
        examples = re.findall(r"```json\n(.*?)\n```", document, re.DOTALL)
        self.assertGreaterEqual(len(examples), 7)
        checker = Draft202012Validator(SCHEMA)
        for example in examples:
            with self.subTest(example=example):
                checker.validate(json.loads(example))

    def test_current_document_response_examples(self):
        found = set()
        names = ('当前整体方案.md', '纯公开数据可行性验证方案.md', 'JRDB数据与测试实现.md', '项目框架.md')
        for name in names:
            document = (ROOT / 'docs' / name).read_text(encoding='utf-8')
            for example in re.findall(r'```json\n(.*?)\n```', document, re.DOTALL):
                response = json.loads(example)
                # Dataset snippets, split designs and artificial RLE placeholders
                # are not model responses and have their own documented format.
                if not isinstance(response, dict) or not {'g', 'u', 'p'} & set(response):
                    continue
                mode = 'GOAL' if 'p' in response else 'REL_FOLLOW' if {'id', 's'} & set(response) else 'REL'
                with self.subTest(document=name, example=example):
                    validator(mode).validate(response)
                found.add(mode)
        self.assertEqual(found, {'REL', 'REL_FOLLOW', 'GOAL'})

    def test_mode_mismatch_rejected(self):
        self.assertFalse(validator("REL").is_valid({"p": -1, "s": "u"}))
        self.assertFalse(validator("GOAL").is_valid({"g": [], "u": []}))

    def test_invalid_static_cases(self):
        cases = [
            ("REL", {"g": [{"m": [1], "r": "w", "k": 0}], "u": []}),
            ("REL", {"g": [{"m": [1, 1], "r": "w", "k": 0}], "u": []}),
            ("REL", {"g": [{"m": [1, 2], "r": "t", "k": 1}], "u": []}),
            ("REL", {"g": [], "u": [100]}),
            ("REL", {"g": []}),
            ("REL_FOLLOW", {"g": [], "u": [], "id": -1, "s": "v"}),
            ("REL_FOLLOW", {"g": [], "u": [], "id": 2, "s": "a"}),
            ("GOAL", {"p": 8, "s": "v"}),
            ("GOAL", {"p": -1, "s": "v"}),
            ("GOAL", {"p": 2, "s": "u"}),
            ("GOAL", {"p": -1, "s": "n", "reason": "extra"}),
        ]
        for mode, response in cases:
            with self.subTest(mode=mode, response=response):
                self.assertFalse(validator(mode).is_valid(response))

    def test_twenty_person_structures(self):
        groups = [{"m": [n, n + 1], "r": "w", "k": 0} for n in range(0, 20, 2)]
        validator("REL_FOLLOW").validate({"g": groups, "u": [], "id": 19, "s": "v"})
        validator("REL").validate({"g": [], "u": list(range(20))})

    def test_dynamic_overlap_not_claimed_as_static_validation(self):
        # This is intentionally schema-valid, but the deployment validator MUST
        # reject overlap between g and u. Do not mistake these tests for it.
        response = {"g": [{"m": [1, 2], "r": "q", "k": 1}], "u": [1]}
        self.assertTrue(validator("REL").is_valid(response))


if __name__ == "__main__":
    unittest.main()
