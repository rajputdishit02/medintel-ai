import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class StreamlitSmokeTests(unittest.TestCase):
    def test_overview_renders_without_exception(self) -> None:
        app = AppTest.from_file(
            str(Path(__file__).resolve().parents[1] / "streamlit_app.py"),
            default_timeout=20,
        ).run()
        self.assertEqual(app.exception, [])
        self.assertEqual(app.title[0].value, "MedIntel AI")
        self.assertGreaterEqual(len(app.metric), 4)


if __name__ == "__main__":
    unittest.main()
