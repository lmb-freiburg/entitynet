"""
Can be used to filter the taxons e.g. remove all taxons of rank "kingdom"
"""

import argparse
from typing import Iterator

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    parser.add_argument("--ranks", nargs="+", type=str, default=None)
    parser.add_argument("--parents", nargs="+", type=str, default=None)
    parser.add_argument("--parent-ranks", nargs="+", type=str, default=None)
    return parser.parse_args()


def normalize(s: str) -> str:
    return s.lower()


def split_iter(s: str, sep: str = ";") -> Iterator[str]:
    for item in s.split(sep):
        item = item.strip()
        if not item:
            continue
        yield item


def has_parent(
    taxons: dict[str, list],
    columns: dict[str, int],
    taxon: str,
    parents: set[str],
    parent_ranks: set[str],
) -> bool:
    info = taxons.get(taxon, None)
    if info is None:
        return False

    for parent in split_iter(info[columns["parents"]]):
        parent_info = taxons.get(parent, None)
        if parent_info is None:
            continue

        if parent_ranks:
            ranks = set(normalize(r) for r in split_iter(parent_info[columns["rank_labels"]]))
            if ranks.intersection(parent_ranks):
                return True

        if parents:
            name = normalize(parent_info[columns["taxon_name"]])
            if name in parents:
                return True

        if has_parent(
            taxons,
            columns,
            parent,
            parents,
            parent_ranks,
        ):
            return True

    return False


def filter_taxons(args: argparse.Namespace):
    taxons: dict[str, list] = {}
    with open(args.input, "r") as inf:
        # skip header
        columns = {col[1:]: i for i, col in enumerate(next(inf).rstrip("\r\n").split("\t")[1:])}

        for line in inf:
            line = line.rstrip("\r\n")
            taxon, *info = line.split("\t")
            assert taxon not in taxons, f"duplicate taxon {taxon}"
            taxons[taxon] = info

    # prepare ranks and parent
    if args.ranks is not None:
        ranks = set(normalize(r) for r in args.ranks)
    else:
        ranks = set()

    if args.parents is not None:
        parents = set(normalize(p) for p in args.parents)
    else:
        parents = set()

    if args.parent_ranks is not None:
        parent_ranks = set(normalize(r) for r in args.parent_ranks)
    else:
        parent_ranks = set()

    with open(args.output, "w") as of:
        for taxon, info in tqdm(taxons.items(), desc="filtering", leave=False):
            if args.ranks:
                taxon_ranks = set(normalize(r) for r in split_iter(info[columns["rank_labels"]]))
                if not taxon_ranks.intersection(ranks):
                    continue

            if args.parents:
                if not has_parent(
                    taxons,
                    columns,
                    taxon,
                    parents,
                    parent_ranks,
                ):
                    continue

            info_str = "\t".join(info)
            of.write(f"{taxon}\t{info_str}\n")


if __name__ == "__main__":
    filter_taxons(parse_args())
