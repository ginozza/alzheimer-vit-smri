"""Run E2 on an explicitly partitioned manifest of registered derivatives."""

import argparse
import json
import sys

from src.data.preprocessing import PreprocessingConfig
from src.data.preprocessing_batch import run_preprocessing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True, help="New directory; existing results are never overwritten")
    parser.add_argument("--config", default="configs/e2.json")
    args = parser.parse_args()
    try:
        summary = run_preprocessing(args.manifest, args.data_root, args.output_dir, PreprocessingConfig.load(args.config))
    except (OSError, ValueError, TypeError) as error:
        print(f"E2 failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2))
    return 1 if summary["rejected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
