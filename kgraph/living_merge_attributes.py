import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    return parser.parse_args()


def add_to_map(
    file: str,
    entity_map: dict[str, tuple[list[int], dict[tuple[str, str], str]]],
    stats: dict[str, int],
) -> None:
    print(f"Reading attributes from {file}", file=sys.stderr)
    with open(file, "r") as f:
        for rank, line in enumerate(f):
            entity, *attributes = line.rstrip("\r\n").split("\t")
            if entity not in entity_map:
                # initialize
                entity_map[entity] = ([], {})

            ranks, attribute_map = entity_map[entity]
            ranks.append(rank)

            for i in range(0, len(attributes), 3):
                category, attribute, query = attributes[i : i + 3]
                key = (category, attribute.lower())
                # for a given key (category, attribute),
                # only keep the first query
                # this implies that the attribute files should
                # be specified from best to worst model
                if key not in attribute_map:
                    attribute_map[key] = query
                    stats["merged"] += 1
                else:
                    stats["removed"] += 1


def merge(args: argparse.Namespace) -> None:
    entity_map = {}
    stats = {"merged": 0, "removed": 0}
    print(f"Merging attributes from {len(args.files)} files", file=sys.stderr)
    for file in args.files:
        add_to_map(file, entity_map, stats)

    # sort ascending by average rank
    print(f"Aggregated {len(entity_map)} unique entities", file=sys.stderr)
    sorted_entities = sorted(
        entity_map.items(),
        key=lambda x: sum(x[1][0]) / len(x[1][0]),
    )
    print("Writing merged attributes", file=sys.stderr)
    print(
        f"Merged attributes kept: {stats['merged']}; duplicates removed: {stats['removed']}",
        file=sys.stderr,
    )
    for entity, (_, attribute_map) in sorted_entities:
        s = entity
        for (category, attribute), query in attribute_map.items():
            s += f"\t{category}\t{attribute}\t{query}"
        print(s, flush=True)


if __name__ == "__main__":
    merge(parse_args())
