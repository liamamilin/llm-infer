"""实验的端到端测试：需要 .venv 已装好 requirements.txt；缺依赖或模型时自动跳过。"""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PY = ROOT / ".venv" / "bin" / "python"
MODEL = ROOT / "models" / "smollm2-135m-instruct" / "model.safetensors"


def run(module: str) -> subprocess.CompletedProcess:
    return subprocess.run([str(VENV_PY), "-m", module], cwd=ROOT, capture_output=True, text=True)


@unittest.skipUnless(VENV_PY.exists(), "未创建 .venv（见 README 的实验环境一节）")
class ExperimentTest(unittest.TestCase):
    def test_ch00_smoke(self):
        r = run("experiments.ch00.smoke")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)

    def test_ch01_assets(self):
        r = run("experiments.ch01.inspect_assets")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("PASS", r.stdout)


class ConfigTest(unittest.TestCase):
    def test_revision_is_full_sha(self):
        sys.path.insert(0, str(ROOT))
        from experiments.common.config import MODEL_REVISION
        self.assertRegex(MODEL_REVISION, r"^[0-9a-f]{40}$")


if __name__ == "__main__":
    unittest.main()
