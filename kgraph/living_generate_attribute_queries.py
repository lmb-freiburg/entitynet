import argparse
import json
from collections import Counter

EXCLUDE_TERMS = "drawing clipart illustration cartoon vector painting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["animal", "plant"])
    parser.add_argument("attributes")
    parser.add_argument("natural_types")
    parser.add_argument("--search-api", choices=["google", "bing"], default="google")
    return parser.parse_args()


def generate(args: argparse.Namespace):
    entity_types = {}
    with open(args.natural_types, "r") as f:
        for line in f:
            entity, typ_id, typ_name = line.rstrip("\r\n").split("\t")
            entity_types[entity] = (typ_id, typ_name)

    counts = Counter()
    with open(args.attributes, "r") as f:
        for line in f:
            line = line.rstrip("\r\n")
            entity, *attributes = line.split("\t")
            for i in range(0, len(attributes), 3):
                _, attribute, _ = attributes[i : i + 3]
                attribute = attribute.lower()
                entity_type = entity_types.get(entity, args.type)

                key = (attribute, entity_type)
                counts[key] += 1  # type: ignore

    # sort by frequency
    for (attribute, (typ_id, typ_name)), count in counts.most_common():
        query_params = {
            "attribute": attribute,
            "count": count,
            "entity_type": typ_id,
            "entity_type_name": typ_name,
            "query": f"{attribute} {typ_name}",
        }

        if args.search_api == "bing":
            query_params["type"] = "Photo"
        else:
            query_params["type"] = "photo"
            query_params["exclude"] = EXCLUDE_TERMS

        print(json.dumps(query_params), flush=True)


if __name__ == "__main__":
    generate(parse_args())
