"""Offline entry point: python -m medintel sources."""

import argparse
import json
from pathlib import Path

from medintel import __version__
from medintel.catalog import COMPONENTS, CYCLE


def main() -> None:
    parser = argparse.ArgumentParser(description="MedIntel AI research foundation")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=[
        "sources", "acquire", "verify-data", "build-cohort", "eda",
        "baseline", "compare-models", "interpret", "literature-acquire",
        "literature-ask",
    ])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--question")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    if args.command == "sources":
        print(f"Proposed NHANES sources: {CYCLE}")
        for component in COMPONENTS:
            print(f"{component.code}: {component.description}")
            print(f"  {component.documentation_url}")
    elif args.command == "acquire":
        from medintel.acquisition import acquire_snapshot

        manifest = acquire_snapshot(args.data_dir, overwrite=args.overwrite)
        print(f"Wrote verified source manifest: {manifest}")
    elif args.command == "verify-data":
        from medintel.acquisition import verify_snapshot

        count = verify_snapshot(args.data_dir)
        print(f"Verified {count} local source files against the manifest")
    elif args.command == "build-cohort":
        from medintel.cohort import write_cohort

        cohort, summary = write_cohort(args.data_dir)
        print(f"Wrote local cohort: {cohort}")
        print(f"Wrote aggregate audit: {summary}")
    elif args.command == "eda":
        from medintel.analysis import write_eda

        report, table = write_eda(args.data_dir, Path("reports"))
        print(f"Wrote EDA report: {report}")
        print(f"Wrote prevalence table: {table}")
    elif args.command == "baseline":
        from medintel.modeling import run_baselines

        output = run_baselines(args.data_dir, Path("reports"), Path("models"))
        print(f"Wrote baseline metrics: {output}")
    elif args.command == "compare-models":
        from medintel.comparison import run_comparison

        output = run_comparison(args.data_dir, Path("reports"), Path("models"))
        print(f"Wrote model comparison: {output}")
    elif args.command == "interpret":
        from medintel.interpretation import run_interpretation

        output = run_interpretation(args.data_dir, Path("reports"), Path("models"))
        print(f"Wrote interpretation analysis: {output}")
    elif args.command == "literature-acquire":
        from medintel.literature import acquire_pubmed

        corpus = args.data_dir / "literature" / "pubmed-corpus.jsonl"
        summary = args.data_dir / "metadata" / "literature-summary.json"
        articles = acquire_pubmed(corpus, summary)
        print(f"Wrote {len(articles)} PubMed records: {corpus}")
    elif args.command == "literature-ask":
        from medintel.literature import answer_question

        if not args.question:
            parser.error("literature-ask requires --question")
        corpus = args.data_dir / "literature" / "pubmed-corpus.jsonl"
        result = answer_question(args.question, corpus, use_openai=not args.offline)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
