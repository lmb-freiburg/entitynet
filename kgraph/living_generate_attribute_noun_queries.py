import argparse
import json

EXCLUDE_TERMS = "drawing clipart illustration cartoon vector painting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("attributes")
    parser.add_argument("entities")
    parser.add_argument("natural_types")
    parser.add_argument("--search-api", choices=["google", "bing"], default="google")
    return parser.parse_args()


def generate(args: argparse.Namespace):
    entity_infos = {}
    with open(args.entities, "r") as f:
        for line in f:
            entity = line.rstrip("\r\n").split("\t")
            label = entity[1][1:-4]
            or_terms = [
                alias.lower()
                for alias in entity[4][1:-1].split(";")
                + entity[5][1:-1].split(";")
                + entity[6][1:-1].split(";")
                if alias != ""
            ]
            or_terms.append(label.lower())
            or_terms = list({k: None for k in or_terms}.keys())
            entity_infos[entity[0]] = or_terms

    entity_types = {}
    with open(args.natural_types, "r") as f:
        for line in f:
            entity, _, typ_name = line.rstrip("\r\n").split("\t")
            entity_types[entity] = typ_name

    with open(args.attributes, "r") as f:
        for line in f:
            line = line.rstrip("\r\n")
            entity, *attributes = line.split("\t")
            for i in range(0, len(attributes), 3):
                category, attribute, query = attributes[i : i + 3]
                assert (
                    entity in entity_infos and entity in entity_types
                ), f"expected entity {entity} in infos and natural types"

                query_params: dict = {
                    "entity": entity,
                    "attribute_category": category,
                    "attribute": attribute,
                    "query": query,
                }

                if args.search_api == "bing":
                    query_params["type"] = "Photo"
                    query_params["or_terms"] = entity_types[entity]

                else:
                    # filter out all or_terms that are already in the query
                    # TODO: maybe wrap multi-word terms in quotes??
                    # see bit.ly/AllTheOperators
                    or_terms = [
                        term
                        for term in entity_infos[entity]  # + [entity_types[entity]]
                        if term not in query.lower()
                    ]

                    # uniquify without changing order, put multi-word terms in quotes
                    or_terms = list({k: None for k in or_terms}.keys())
                    # consider put multi-word terms in quotes
                    or_terms_masked = []
                    for or_term in or_terms:
                        # if len(or_term.split(" ")) > 1:
                        #     or_term = f'"{or_term}"'
                        or_terms_masked.append(or_term)

                    query_params["type"] = "photo"
                    query_params["exclude"] = EXCLUDE_TERMS
                    query_params["or_terms"] = " ".join(or_terms_masked)

                print(json.dumps(query_params), flush=True)


if __name__ == "__main__":
    generate(parse_args())
