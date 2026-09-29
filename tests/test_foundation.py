"""Offline checks of the public command and source catalogue contract."""

import subprocess
import sys
import unittest
from urllib.parse import urlparse

from medintel.catalog import COMPONENTS


class FoundationTests(unittest.TestCase):
    def test_catalogue_has_unique_codes_and_official_links(self):
        codes = [component.code for component in COMPONENTS]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertIn("DEMO_L", codes)
        for component in COMPONENTS:
            url = urlparse(component.documentation_url)
            self.assertEqual(url.scheme, "https")
            self.assertEqual(url.netloc, "wwwn.cdc.gov")
            self.assertTrue(url.path.endswith(f"/{component.code}.htm"))

    def test_sources_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "medintel", "sources"],
            capture_output=True, text=True, check=True,
        )
        self.assertIn("Proposed NHANES sources", result.stdout)
        self.assertIn("DEMO_L", result.stdout)
        self.assertIn("MCQ_L", result.stdout)

    def test_invalid_command_fails(self):
        result = subprocess.run(
            [sys.executable, "-m", "medintel", "predict"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
