import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from medintel.literature import Article, answer_question, retrieve, validate_citations


ARTICLES = [
    Article(
        pmid="1001",
        title="Calibration of cardiovascular prediction models",
        abstract="Calibration compares predicted cardiovascular probabilities with observed outcomes.",
        journal="Methods Journal",
        year="2024",
        doi=None,
        url="https://pubmed.ncbi.nlm.nih.gov/1001/",
    ),
    Article(
        pmid="1002",
        title="Blood pressure measurement in health surveys",
        abstract="NHANES measures blood pressure using standardized examination procedures.",
        journal="Survey Journal",
        year="2023",
        doi=None,
        url="https://pubmed.ncbi.nlm.nih.gov/1002/",
    ),
]


class LiteratureTests(unittest.TestCase):
    def test_retrieval_ranks_relevant_article(self) -> None:
        results = retrieve("How should prediction model calibration be assessed?", ARTICLES)
        self.assertEqual(results[0][0].pmid, "1001")

    def test_citation_validator_rejects_missing_and_out_of_range(self) -> None:
        self.assertTrue(validate_citations("Supported by the study [1].", 2))
        self.assertFalse(validate_citations("No citation.", 2))
        self.assertFalse(validate_citations("Unsupported [3].", 2))

    def test_offline_answer_is_cited_and_research_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(
                "".join(json.dumps(asdict(article)) + "\n" for article in ARTICLES),
                encoding="utf-8",
            )
            result = answer_question("What does calibration assess?", path, use_openai=False)
        self.assertEqual(result["mode"], "extractive")
        self.assertIn("[1]", result["answer"])
        self.assertIn("not medical advice", result["answer"])

    def test_irrelevant_question_abstains(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.jsonl"
            path.write_text(
                "".join(json.dumps(asdict(article)) + "\n" for article in ARTICLES),
                encoding="utf-8",
            )
            result = answer_question("volcanic geology of Mars", path, use_openai=False)
        self.assertEqual(result["mode"], "abstained")
        self.assertEqual(result["sources"], [])


if __name__ == "__main__":
    unittest.main()
