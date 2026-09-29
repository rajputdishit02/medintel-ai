"""Offline entry point: python -m medintel sources."""

import argparse

from medintel import __version__
from medintel.catalog import COMPONENTS, CYCLE


def main() -> None:
    parser = argparse.ArgumentParser(description="MedIntel AI research foundation")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("command", choices=["sources"])
    args = parser.parse_args()
    if args.command == "sources":
        print(f"Proposed NHANES sources: {CYCLE}")
        for component in COMPONENTS:
            print(f"{component.code}: {component.description}")
            print(f"  {component.documentation_url}")


if __name__ == "__main__":
    main()
