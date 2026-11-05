"""Current documentation and module registration; no models or network."""
import re
import unittest
from decision.vlm.project_config import ROOT,VLM_ROOT,load_config


class ProjectLayoutTests(unittest.TestCase):
    def test_current_document_links_resolve(self):
        for doc in [ROOT/'AGENTS.md',ROOT/'README.md',*(ROOT/'docs').rglob('*.md')]:
            if 'archive' in doc.parts:continue
            for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)',doc.read_text()):
                if target.startswith(('https:','http:','mailto:','#')):continue
                path=(doc.parent/target.split('#',1)[0]).resolve()
                if path==ROOT/'ENVIRONMENT.local.md' or any(path.is_relative_to(ROOT/p) for p in ['outputs','data']):continue
                with self.subTest(document=doc.name,target=target):self.assertTrue(path.exists())

    def test_current_tasks_have_prompts_and_frozen_two_second_policy(self):
        config=load_config()
        self.assertEqual(set(config['tasks']),{'groups','appearance','follow','seating'})
        for task in config['tasks'].values():self.assertTrue((ROOT/task['prompt']).is_file())
        self.assertTrue(config['model']['frozen']);self.assertFalse(config['model']['finetuning'])
        self.assertEqual((config['inference']['period_s'],config['inference']['rate_hz']),(2,.5))


if __name__=='__main__':unittest.main()
