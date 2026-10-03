import argparse
import json
import random
import sys
from collections import Counter, defaultdict

from entitynet.datasets.wikidata.wikidata_utils import strip_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("entities", type=str)
    parser.add_argument("types_and_classes", type=str)
    parser.add_argument("searched_queries", type=str, nargs="*", default=[])
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--entity-queries", action="store_true")
    parser.add_argument("--alias-queries", action="store_true")
    return parser.parse_args()


def _load_tsv(path: str) -> list[list[str]]:
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def _load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _load_jsonl(path: str) -> list:
    with open(path, "r") as f:
        return [json.loads(line.rstrip("\r\n")) for line in f]


def build(args: argparse.Namespace):
    entities = _load_tsv(args.entities)
    types_and_classes = _load_json(args.types_and_classes)
    already_searched = set(
        query["entity"] for path in args.searched_queries for query in _load_jsonl(path)
    )

    ignored = 0
    missing = 0
    searched = 0
    total_aliases = 0
    types = {}
    keep = {}
    for id, name, *info in entities:
        if id in already_searched:
            searched += 1
            continue

        if id not in types_and_classes:
            missing += 1
            continue

        type_and_class = types_and_classes[id]
        if type_and_class["visuality_class"]["class"] != "visual":
            ignored += 1
            continue

        score = int(info[1])
        if score < args.min_score:
            ignored += 1
            continue

        name = strip_label(name)
        typ_id = type_and_class["natural_type"]["id"]
        typ_name = type_and_class["natural_type"]["name"]
        types[typ_id] = typ_name

        aliases = [alias for alias in info[2][1:-1].split(";;;") if alias]
        total_aliases += len(aliases)

        keep[id] = (name, typ_id, score, aliases)

    print(f"{len(entities):,} entities in total", file=sys.stderr)
    print(
        f"{total_aliases:,} aliases in total "
        f"({total_aliases / max(1, len(entities)):.2f} on avg. per entity)",
        file=sys.stderr,
    )
    print(f"Already searched {searched:,} entities", file=sys.stderr)
    print(f"Ignored {ignored:,} non-visual or unpopular entities", file=sys.stderr)
    print(f"Missing {missing:,} entities", file=sys.stderr)

    # get average entities per visual type
    print(file=sys.stderr)
    print(f"{len(types):,} visual types", file=sys.stderr)
    count_per_type = Counter()
    score_per_type = Counter()
    names_per_type = defaultdict(list)
    for name, typ_id, score, _ in keep.values():
        count_per_type[typ_id] += 1
        score_per_type[typ_id] += score
        names_per_type[typ_id].append(name)

    print("The 10 most common types:", file=sys.stderr)
    for typ_id, count in count_per_type.most_common(10):
        avg_score = score_per_type[typ_id] / count
        perc = count / max(1, count_per_type.total())
        samples = random.sample(names_per_type[typ_id], min(5, count))
        print(
            f"{types[typ_id]}: {count:,} entities {perc:.2%}, "
            f"avg_score={avg_score:.2f}: {samples}",
            file=sys.stderr,
        )

    print(file=sys.stderr)
    alias_to_id = {}
    for id, (*_, aliases) in keep.items():
        for alias in aliases:
            if alias in alias_to_id:
                alias_to_id[alias].append(id)
            else:
                alias_to_id[alias] = [id]

    print(f"Unique aliases: {len(alias_to_id):,}", file=sys.stderr)
    print(
        f"Avg. entities per alias: {sum(len(ids) for ids in alias_to_id.values()) / max(1, len(alias_to_id)):.2f}",
        file=sys.stderr,
    )
    print(file=sys.stderr)

    if args.entity_queries:
        print(f"Building {len(keep):,} entity queries", file=sys.stderr)
        for id, (name, _, _, _) in keep.items():
            print(json.dumps({"entity": id, "query": name, "type": "Photo"}))

    elif args.alias_queries:
        total = sum(len(ids) for ids in alias_to_id.values())
        print(f"Building {total:,} entity alias queries", file=sys.stderr)

        for id, (*_, aliases) in keep.items():
            for alias in aliases:
                print(json.dumps({"entity": id, "query": alias, "type": "Photo"}))


if __name__ == "__main__":
    build(parse_args())
