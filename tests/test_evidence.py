"""公开证据文件的完整性检查：每章引用的证据都存在、状态为 PASS、且含必要字段、不含私人路径。"""
import json
import unittest
from pathlib import Path

PUB = Path(__file__).resolve().parents[1] / "evidence" / "public"
REQUIRED = {"chapter_id", "experiment_id", "status", "timestamp", "platform", "model_id",
            "model_revision", "runtime", "runtime_version", "exact_command", "assertions", "git_sha"}


class EvidenceTest(unittest.TestCase):
    def test_all_public_evidence_is_valid(self):
        files = sorted(PUB.glob("ch*.json"))
        self.assertGreaterEqual(len(files), 16)
        for f in files:
            with self.subTest(f=f.name):
                text = f.read_text()
                d = json.loads(text)
                self.assertFalse(REQUIRED - d.keys(), f"{f.name} 缺字段")
                self.assertEqual(d["status"], "PASS")
                self.assertTrue(all(d["assertions"].values()))
                self.assertNotIn("/Users/", text)
                self.assertNotIn("/root/", text)

    def test_every_chapter_page_links_existing_evidence(self):
        import re
        docs = PUB.parents[1] / "docs" / "chapters"
        for page in docs.glob("ch*.md"):
            for name in re.findall(r"evidence/public/([\w\-]+\.json)", page.read_text()):
                self.assertTrue((PUB / name).exists(), f"{page.name} 引用了不存在的 {name}")


if __name__ == "__main__":
    unittest.main()
