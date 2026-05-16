import argparse
import sys

from train_lightgbm import main as train_lightgbm
from train_xgboost import main as train_xgboost


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper. Prefer train_xgboost.py or "
            "train_lightgbm.py when training one boosting model."
        )
    )
    parser.add_argument("--params", default=None)
    parser.add_argument("--target-recall", default=None)
    args = parser.parse_args()

    forwarded_args = [sys.argv[0]]
    if args.params is not None:
        forwarded_args.extend(["--params", args.params])
    if args.target_recall is not None:
        forwarded_args.extend(["--target-recall", args.target_recall])

    original_argv = sys.argv
    try:
        sys.argv = forwarded_args
        train_xgboost()
        sys.argv = forwarded_args
        train_lightgbm()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    main()
