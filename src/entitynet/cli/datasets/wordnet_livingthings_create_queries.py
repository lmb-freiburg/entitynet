"""
Create JSONS for WordNet livingthings

Notes:
- Removed the duplicate "balm of gilead" "balm of Gilead" with different casing

See WordNet license at src/entitynet/datasets/wordnet/LICENSE
"""

from collections import defaultdict
import os
from pathlib import Path

import pandas as pd
from attrs import define
from loguru import logger

from packg.iotools import dump_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs, add_argument
from typedparser.objects import invert_dict_of_dict

from entitynet.datasets.wordnet import display_synset_from_synname, load_wordnet_nouns


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

    print(f"Start at: ", end="")
    print(display_synset_from_synname(start))
    collected_synnames = get_children(start)
    print(len(collected_synnames))
    collected_synnames = sorted(set(collected_synnames))
    print(f"Got unique nodes:", len(collected_synnames))
    print()
    for i, wnid in enumerate(collected_synnames):
        if i >= 5:
            break
        print(display_synset_from_synname(wnid))
    print()
    wordnet_nouns_lt = {synname: wordnet_nouns[synname] for synname in collected_synnames}
    dflt = pd.DataFrame(invert_dict_of_dict(wordnet_nouns_lt))
    print(dflt.head(3))

    # Remove some instances, and internal nodes
    # some sort of name of award winning dogs (?)
    exclude = []
    dft = dflt[(dflt["node_type"] == "instance") & (dflt["parent"] == "thoroughbred.n.02")]
    for t in dft.index:
        exclude.append(t)
        print(t)
    print(f"Excludes: {len(exclude)}")

    # remove all internal nodes - they are good for filtering, but bad for querying
    parent_synnames = list(dflt[dflt["node_type"] == "internal"].index)
    print(f"Internal nodes: {len(parent_synnames)}")

    run_exclusion(f"livingthings", exclude + parent_synnames, collected_synnames, args)
    run_exclusion(f"livingthingsandparents", exclude, collected_synnames, args)


def run_exclusion(query_name, exclude_list, candidate_synnames, args: Args):
    # run the exclusion and save the output
    print(
        f"---------- Creating queries '{query_name}' with {len(candidate_synnames)} candidates, "
        f"{len(exclude_list)} exclusions."
    )
    exclude_set = set(exclude_list)
    new_synnames = []
    for synname in sorted(candidate_synnames):
        if synname in exclude_set:
            continue
        new_synnames.append(synname)
    # wordnet_nouns_ltv3 = {synname: wordnet_nouns[synname] for synname in new_synnames}
    print(f"synsets {len(new_synnames)}")
    queries2synnamelist = defaultdict(list)

    # there is one query that has different casing, leading to problems later with lowercase search
    # goal is to keep the default casing, but not add two mixedcase versions of the same thing
    lower2upper = {}
    for i, wnid in enumerate(new_synnames):
        wndata = wordnet_nouns[wnid]
        lemmas = wndata["lemmas"]
        # print(wndata["definition"])
        for l in lemmas:
            ll = l.lower()
            if ll in lower2upper:
                lu = lower2upper[ll]
            else:
                lu = l
                lower2upper[ll] = lu
            queries2synnamelist[lu].append(wnid)

    print(f"queries {len(queries2synnamelist)}")
    queries_keys = list(queries2synnamelist.keys())
    print(queries_keys[:10])

    base_dir = args.output_dir
    os.makedirs(base_dir, exist_ok=True)
    dump_json(
        queries_keys,
        base_dir / f"{query_name}.json",
        indent=2,
        custom_format=False,
        overwrite=args.overwrite,
    )
    dump_json(
        new_synnames,
        base_dir / f"{query_name}_synnames.json",
        indent=2,
        custom_format=False,
        overwrite=args.overwrite,
    )
    dump_json(
        queries2synnamelist,
        base_dir / f"{query_name}_query2synnamelist.json",
        indent=2,
        overwrite=args.overwrite,
    )
    return new_synnames


# -----------------------------

wordnet_nouns = load_wordnet_nouns()
start = "living_thing.n.01"
blacklist = {
    "person.n.01",
    "parent.n.02",
    "individual.n.02",
    "mutant.n.01",
    "stander.n.01",
    "utterer.n.01",
    # update for v2
    "microorganism.n.01",
    "cell.n.02",
    "wildlife.n.01",  # leaf
    "biont.n.01",  # leaf
    "dihybrid.n.01",  # leaf
    "world.n.08",  # leaf, all of the living human inhabitants of the earth
    "monster.n.05",
    "abortus.n.01",
    # remove some more imaginary characters
    "church_mouse.n.01",
    "donald_duck.n.01",
    "easter_bunny.n.01",
    "lucy.n.01",
    "mickey_mouse.n.01",
    "mighty_mouse.n.01",
    "minnie_mouse.n.01",
    "yggdrasil.n.01",
}


def get_children(start_synname, current_depth=0, max_depth=-1):
    if current_depth > max_depth > -1:
        return []
    if start_synname in blacklist:
        return []
    # print(f"----"*current_depth, end="")
    # display_id(start_wnid)
    children = [start_synname]
    wndata = wordnet_nouns[start_synname]
    for wnid in wndata["children"]:
        children += get_children(wnid, current_depth=current_depth + 1, max_depth=max_depth)
    return children


if __name__ == "__main__":
    main()
