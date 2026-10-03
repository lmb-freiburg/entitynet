import argparse
import os
import sys

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("types", type=str)
    parser.add_argument("subclass_query", type=str)
    parser.add_argument("output_dir", type=str)
    parser.add_argument("-i", "--instance-query", type=str)
    parser.add_argument("-ms", "--min-subclass-sitelinks", type=int, default=10)
    parser.add_argument("-mi", "--min-instance-sitelinks", type=int, default=50)
    return parser.parse_args()


def get_type_list(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    data = []
    with open(args.types) as inf:
        for line in inf.read().splitlines()[1:]:
            line = line.split(",")
            if line[-1] != "1":
                continue
            data.append(line[:-1])

    queries = {}
    with open(args.subclass_query) as inf:
        queries["subclass"] = inf.read()

    if args.instance_query:
        with open(args.instance_query) as inf:
            queries["instance"] = inf.read()

    stats = []
    all = {}
    for query_type in queries:
        print(f"Querying {query_type}", file=sys.stderr)
        print("-" * 40, file=sys.stderr)
        for ent, label in data:
            if query_type == "instance" and ent == "Q729":
                add = "MINUS { ?ent wdt:P31 wd:Q5. }"
            else:
                add = ""
            query = (
                queries[query_type]
                .replace("%TYPE%", f"wd:{ent}")
                .replace(
                    "%LINKS%",
                    str(
                        args.min_instance_sitelinks
                        if query_type == "instance"
                        else args.min_subclass_sitelinks
                    ),
                )
                .replace("%ADDITIONAL%", add)
            )
            response = requests.post(
                "https://qlever.cs.uni-freiburg.de/api/wikidata",
                headers={
                    "Content-Type": "application/sparql-query",
                    "Accept": "text/tab-separated-values",
                },
                data=query,
            )
            if response.status_code != 200:
                print(response.json()["exception"], file=sys.stderr)

            lines = response.iter_lines()
            next(lines)  # skip header
            count = 0
            filename = f"world-entities.{label.replace(' ', '_')}.tsv"
            with open(os.path.join(args.output_dir, filename), "w") as outf:
                for line in lines:
                    line = line.decode()
                    outf.write(line + "\n")
                    count += 1
                    values = line.split("\t")
                    if values[0] in all:
                        continue
                    all[values[0]] = values[1:]

            stats.append((ent, label, count))
            print(
                f"{len(all):,} entities after adding {label}",
                file=sys.stderr,
            )
        print(file=sys.stderr)

    total_count = max(1, sum(count for *_, count in stats))
    stats = "\n".join(f"{count/total_count:.2%}\t{ent}\t{label}" for ent, label, count in stats)
    print(f"Statistics:\n{stats}", file=sys.stderr)

    with open(os.path.join(args.output_dir, "world-entities.tsv"), "w") as outf:
        for ent, infos in sorted(all.items(), key=lambda item: int(item[1][2]), reverse=True):
            info = "\t".join(infos)
            outf.write(f"{ent}\t{info}" + "\n")


if __name__ == "__main__":
    get_type_list(parse_args())
