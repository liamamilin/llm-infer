import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from doctor import detect

class DoctorTest(unittest.TestCase):
    def test_sanitized_preflight(self):
        result = detect()
        self.assertEqual(result['status'], 'PREFLIGHT_ONLY_NOT_INFERENCE_TEST')
        self.assertFalse(result['model_inference_verified'])
        self.assertIn(result['os'], ('Linux', 'Darwin', 'Windows'))
        self.assertTrue(result['python'])

if __name__ == '__main__':
    unittest.main()
