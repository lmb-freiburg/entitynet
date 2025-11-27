"""
Create the clean living_thing hierarchy, with max 1 parent per node.

See WordNet license at src/entitynet/datasets/wordnet/LICENSE
"""

from collections import Counter
import os
from pathlib import Path

import pandas as pd
from attr import define
from loguru import logger

from packg.iotools import dump_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.datasets.wordnet import load_wordnet_nouns
from entitynet.datasets.wordnet.wordnet_queries import load_wordnet_noun_query2synnames_list
from entitynet.paths import get_entitynet_annotations_dir


@define
class Args(VerboseQuietArgs):
    overwrite: bool = add_argument(
        shortcut="-o", action="store_true", help="Overwrite existing files."
    )
    output_dir: Path = add_argument(
        shortcut="-d", type=str, default="output", help="Output directory for the created files."
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    queries2synnames = load_wordnet_noun_query2synnames_list("livingthingsandparents")
    print(f"Got {len(queries2synnames)} queries")

    synnames = list(sorted(set([a for b in queries2synnames.values() for a in b])))
    print(f"Got {len(synnames)} synnames")
    synnames_set = set(synnames)

    # figure out which of the 82k nouns in wordnet is child of living_thing
    i = 0
    for noun_key, noun_value in nouns.items():
        noun_value["is_living_thing"] = False
    set_is_living_things("living_thing.n.01")

    # use that information to prioritize living_things as parents in case of multiple parents.
    print(Counter([nouns[synname]["is_living_thing"] for synname in nouns.keys()]))
    new_hierarchy = {}
    for synname in synnames:
        noun = nouns[synname]
        parents = noun["parents"]
        n_found_parents = []
        for parent in parents:
            if parent in synnames_set:
                n_found_parents += [parent]
        if len(n_found_parents) == 0:
            assert (
                synname == "living_thing.n.01"
            ), f"Synname: {synname} has no parents and is not root."
            new_hierarchy[synname] = None
            continue
        elif len(n_found_parents) == 1:
            pass
        else:
            i += 1
            print(f"Synname: {synname} has {len(n_found_parents)} parents: {n_found_parents}")
        # the final tie breaker is the order of what is returned by nltk.synset.hypernyms()
        new_hierarchy[synname] = n_found_parents[0]

    inverted_hierarchy = dict()
    for synname, parent_synname in new_hierarchy.items():
        if parent_synname is None:
            continue
        if parent_synname not in inverted_hierarchy:
            inverted_hierarchy[parent_synname] = []
        if synname not in inverted_hierarchy:
            inverted_hierarchy[synname] = []
        inverted_hierarchy[parent_synname].append(synname)

    # calculate depth for each synset, in the new hierarchy
    depths = {}

    def calc_depth(c_synname, d=0):
        depths[c_synname] = d
        for child_synname in inverted_hierarchy[c_synname]:
            calc_depth(child_synname, d + 1)

    calc_depth("living_thing.n.01")

    new_hierarchy = dict(sorted(new_hierarchy.items(), key=lambda x: (depths[x[0]], x[0])))
    print(f"Got {i} synnames with multiple parents")
    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    outf = base_dir / "hierarchy_livingthings.json"
    if outf.is_file():
        print(f"File {outf} exists")
    else:
        dump_json(new_hierarchy, outf, indent=2)

    # next problem: 1 query, multiple synnames (homonymy). around 5% of queries have multiple synnames
    i = 0
    for query, synnames in queries2synnames.items():
        if len(synnames) > 1:
            print(f"Query {query} has {len(synnames)} synnames: {synnames}")
            i += 1
    print(f"Got {i} queries with multiple synnames out of {len(queries2synnames)}")

    print(pd.Series(list(depths.values())).describe())

    # count    8861.000000
    # mean        7.792348
    # std         2.270379
    # min         0.000000
    # 25%         6.000000
    # 50%         8.000000
    # 75%         9.000000
    # max        14.000000
    # dtype: float64


def set_is_living_things(synname):
    nouns[synname]["is_living_thing"] = True
    for children_synname in nouns[synname]["children"]:
        set_is_living_things(children_synname)


if __name__ == "__main__":
    nouns = load_wordnet_nouns()
    main()
