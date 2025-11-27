import argparse
import json

# import sys

EXCLUDE_TERMS = "drawing clipart illustration cartoon vector painting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["animal", "plant"])
    parser.add_argument("entities")
    parser.add_argument("natural_types")
    parser.add_argument("--search-api", choices=["google", "bing"], default="google")
    return parser.parse_args()


def generate(args: argparse.Namespace):
    entity_types = {}
    with open(args.natural_types, "r") as f:
        for line in f:
            entity, _, typ_name = line.rstrip("\r\n").split("\t")
            entity_types[entity] = typ_name

    all_labels = {}
    n_dups = 0
    with open(args.entities, "r") as f:
        for line in f:
            entity = line.rstrip("\r\n").split("\t")
            label = entity[1][1:-4]
            natural_type = entity_types.get(entity[0], args.type)

            query_params: dict = {
                "entity": entity[0],
                "query": label,
            }

            if args.search_api == "bing":
                query_params["type"] = "Photo"
                query_params["or_terms"] = natural_type

            else:
                or_terms = [
                    alias.lower()
                    for alias in entity[4][1:-1].split(";")
                    + entity[5][1:-1].split(";")
                    + entity[6][1:-1].split(";")
                    if alias != "" and alias.lower() != label.lower()
                ]
                or_terms.append(natural_type)

                # uniquify without changing order
                or_terms = list({k: None for k in or_terms}.keys())
                or_terms_masked = []
                for or_term in or_terms:
                    # # putting multi-word terms in quotes seems to put too much weight on them
                    # if len(or_term.split(" ")) > 1:
                    #     or_term = f'({or_term})'
                    or_terms_masked.append(or_term)

                query_params["type"] = "photo"
                query_params["exclude"] = EXCLUDE_TERMS
                query_params["or_terms"] = " ".join(or_terms_masked)

            if label in all_labels:
                # print(
                #     f"WARN: Duplicate query {label} OLD {all_labels[label]} NEW {query_params}",
                #     file=sys.stderr,
                # )
                n_dups += 1
            all_labels[label] = query_params

            print(json.dumps(query_params), flush=True)
    # print(
    #     f"{n_dups=} duplicate queries detected (but possibly different or_terms)", file=sys.stderr
    # )


if __name__ == "__main__":
    generate(parse_args())
