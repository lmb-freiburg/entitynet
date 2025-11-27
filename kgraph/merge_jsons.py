import argparse
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, nargs="+")
    return parser.parse_args()


def run(args: argparse.Namespace):
    all = {}
    for file in args.files:
        with open(file, "r") as f:
            all.update(json.load(f))

    json.dump(all, sys.stdout, indent=2)


if __name__ == "__main__":
    run(parse_args())
