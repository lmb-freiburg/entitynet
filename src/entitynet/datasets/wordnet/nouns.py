"""
https://www.nltk.org/howto/wordnet.html
https://wordnet.princeton.edu/documentation/wndb5wn

Load WordNet noun synsets.

Fixes:
- Either type_parents or inst_parents is set but never both
  so the node type defines whether the node is an instance or not
  and we can just set "parents"
- We are guaranteed one of the 5 node types and their properties

'entity.n.01': {
    'children': ['abstraction.n.06', 'physical_entity.n.01', 'thing.n.08'],
    'definition': 'that which is perceived or known or inferred to have its own '
                'distinct existence (living or nonliving)',
    'inst_children': [],
    'inst_parents': [],
    'lemmas': ['entity'],
    'max_depth': 0,
    'min_depth': 0,
    'synname': 'entity.n.01',
    'node_type': 'root',
    'parent': None,
    'parents': [],
    'type_children': ['abstraction.n.06', 'physical_entity.n.01', 'thing.n.08'],
    'type_parents': [],
    'wnid': 'n00001740'
}

"""

from __future__ import annotations

from collections import defaultdict
from pprint import pprint
from typing import Optional

import pandas as pd
from attrs import define
from loguru import logger
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset  # noqa

from packg.iotools import dump_json, load_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.tqdmext import tqdm_max_ncols
from typedparser import NamedTupleMixin, TypedParser, VerboseQuietArgs
from typedparser.objects import invert_dict_of_dict
from visiontext.distutils import get_global_rank

from entitynet.datasets.wordnet.utils import (
    NodeTypes,
    convert_synname_to_synset,
    convert_synset_to_synname,
    convert_synset_to_wnid,
    display_synset_from_synname,
    ensure_wordnet_is_downloaded,
)
from entitynet.paths import get_entitynet_cache_dir

_wordnet_nouns: Optional[dict] = None


@define
class WordnetNoun(NamedTupleMixin):
    wnid: str
    synname: str
    definition: str
    node_type: str
    lemmas: list[str]
    min_depth: int
    max_depth: int
    parent: str | None
    parents: list[str]
    type_parents: list[str]
    inst_parents: list[str]
    children: list[str]
    type_children: list[str]
    inst_children: list[str]


def load_wordnet_nouns_as_namedtuples(use_cache: bool = True) -> dict[str, WordnetNoun]:
    nouns = load_wordnet_nouns(use_cache=use_cache)
    return {k: WordnetNoun(**v) for k, v in nouns.items()}


def load_wordnet_nouns(use_cache=True):
    """
    Load all noun synsets from wordnet into a custom structure.

    Args:
        use_cache: store results in cache for load_wordnet_nouns
    """
    global _wordnet_nouns
    if _wordnet_nouns is not None:
        return _wordnet_nouns
    if use_cache:
        cache_file = get_entitynet_cache_dir() / f"wordnet/wordnet_nouns_singleparent.json"
        if cache_file.is_file():
            _wordnet_nouns = load_json(cache_file)
            return _wordnet_nouns

    ensure_wordnet_is_downloaded()
    all_nouns = list(wn.all_synsets("n"))
    logger.info(f"Total number of nouns: {len(all_nouns)}")
    _root: Synset = wn.synset("entity.n.01")

    wordnet_data = {}
    synname_to_type_children = defaultdict(list)
    synname_to_inst_children = defaultdict(list)
    n_mixed_parents = 0
    for synset in tqdm_max_ncols(all_nouns, desc="Read WordNet, pass 1/2"):
        wnid = convert_synset_to_wnid(synset)
        synname = convert_synset_to_synname(synset)

        # collect "type of" parents and "instance of" parents
        type_parents = synset.hypernyms()
        inst_parents = synset.instance_hypernyms()

        if len(type_parents) > 0 and len(inst_parents) > 0:
            # 5 nodes are a "type of" something but also an "instance of" something else
            # seems that they should be defined as instances instead.
            logger.debug(
                f"{synname} has both type_parents {type_parents} and inst_parents {inst_parents}"
            )
            n_mixed_parents += 1
            new_inst_parents = type_parents + inst_parents
            type_parents = []
            inst_parents = new_inst_parents
            logger.debug(f"New parents {type_parents} and inst parents {inst_parents}")

        is_instance = False
        if len(type_parents) == 0 and len(inst_parents) > 0:
            is_instance = True

        type_parents_synnames = [convert_synset_to_synname(p) for p in type_parents]
        inst_parents_synnames = [convert_synset_to_synname(p) for p in inst_parents]

        for p_synname in type_parents_synnames:
            synname_to_type_children[p_synname].append(synname)
        for p_synname in inst_parents_synnames:
            synname_to_inst_children[p_synname].append(synname)

        lemmas = synset.lemma_names()
        name = synset.name()
        definition = synset.definition()
        lemmas = [l.replace("_", " ") for l in lemmas]

        wordnet_data[synname] = {
            "type_parents": type_parents_synnames,
            "inst_parents": inst_parents_synnames,
            "lemmas": lemmas,
            "synname": name,
            "definition": definition,
            "is_instance": is_instance,
            "wnid": wnid,
            "min_depth": synset.min_depth(),
            "max_depth": synset.max_depth(),
        }

    synname_to_type_children = dict(synname_to_type_children)
    synname_to_inst_children = dict(synname_to_inst_children)

    n_instance_but_not_leaf, n_multi_parents = 0, 0
    for synname, wndata in tqdm_max_ncols(
        wordnet_data.items(), total=len(wordnet_data), desc="Read WordNet nouns, pass 2/2"
    ):
        _synset = convert_synname_to_synset(synname)
        type_parents_synnames = wndata["type_parents"]
        inst_parent_synnames = wndata["inst_parents"]
        all_parents_synnames = type_parents_synnames + inst_parent_synnames
        children_synnames = sorted(synname_to_type_children.get(synname, []))
        inst_children_synnames = sorted(synname_to_inst_children.get(synname, []))
        all_children_synnames = children_synnames + inst_children_synnames

        if len(all_parents_synnames) == 0 and len(all_children_synnames) == 0:
            node_type = NodeTypes.STANDALONE
        elif len(all_parents_synnames) == 0 and len(all_children_synnames) > 0:
            node_type = NodeTypes.ROOT
        elif len(all_parents_synnames) > 0 and len(all_children_synnames) == 0:
            if wndata["is_instance"]:
                node_type = NodeTypes.INSTANCE
            else:
                node_type = NodeTypes.LEAF
        else:
            node_type = NodeTypes.INTERNAL

        if len(all_parents_synnames) == 0:
            parent = None
        elif len(all_parents_synnames) == 1:
            parent = all_parents_synnames[0]
        else:
            # in case of multiparents, define the single parent as lowest depth parent
            # break ties by taking hypernyms over instance hypernyms, and otherwise using the
            # first thing that wordnet returned
            p_d = defaultdict(list)
            for p in all_parents_synnames:
                parent_depth = wordnet_data[p]["min_depth"]
                p_d[parent_depth].append(p)
            min_d = min(p_d.keys())
            parent = list(p_d[min_d])[0]

        if isinstance(parent, list):
            raise ValueError(f"Multiple parents: {parent}")
        wndata.update(
            {
                "parent": parent,
                "type_children": children_synnames,
                "inst_children": inst_children_synnames,
                "node_type": node_type,
                "parents": all_parents_synnames,
                "children": all_children_synnames,
            }
        )
        if wndata["is_instance"] and len(children_synnames) > 0:
            # according to this doc, instances should always be terminal nodes
            # https://globalwordnet.github.io/gwadoc/#instance_hyponym
            # there are 11 of 7686 such "wrong" instances in the nouns
            # with the current logic those end up as "internal" nodes and thats fine.
            logger.debug(f"{synname} is instance but not leaf: {wndata}")
            n_instance_but_not_leaf += 1

        if len(all_parents_synnames) > 1:
            n_multi_parents += 1

        del wndata["is_instance"]

    if n_instance_but_not_leaf > 0:
        logger.warning(f"{n_instance_but_not_leaf} synsets are instance but not leaf")
    logger.info(f"Nodes with multiple parents: {n_multi_parents}")
    logger.info(f"Nodes with mixed parents: {n_mixed_parents}")

    if use_cache and get_global_rank() == 0:
        dump_json(wordnet_data, cache_file, create_parent=True)  # noqa

    _wordnet_nouns = wordnet_data
    return _wordnet_nouns


def display_noun_hierarchy(synname: str, multi_parents: bool = True, depth: int = 0) -> str:
    """

    Args:
        synname:
        multi_parents: true: display all parents, false: display shortest tree
        depth: current depth

    Returns:

    """
    nouns = load_wordnet_nouns()
    if synname not in nouns:
        raise ValueError(f"Synset {synname} not in wordnet")
    noun = nouns[synname]
    out = [f"[{depth:2d}] ", display_synset_from_synname(synname), "\n"]
    if multi_parents:
        parents = noun["parents"]
    else:
        parents = [noun["parent"]]
    for parent in parents:
        out.append(display_noun_hierarchy(parent, multi_parents=multi_parents, depth=depth + 1))
    return "".join(out)


@define
class Args(VerboseQuietArgs):
    pass


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    wordnet_nouns = load_wordnet_nouns(use_cache=False)
    df = pd.DataFrame(invert_dict_of_dict(wordnet_nouns))
    print(df)
    print(f"Done")
    print(df["node_type"].value_counts().to_dict())
    # {'leaf': 57267, 'internal': 17156, 'instance': 7691, 'root': 1}
    print(df["parents"].apply(len).value_counts().to_dict())
    # {1: 72962, 0: 7731, 2: 1388, 3: 30, 4: 3, 5: 1}
    print(df["inst_parents"].apply(len).value_counts().to_dict())
    # {0: 74385, 1: 6939, 2: 746, 3: 33, 4: 9, 5: 2, 6: 1}
    pprint(next(iter(wordnet_nouns.items())))

    wordnet_nouns_namedtuples = load_wordnet_nouns_as_namedtuples()
    print(next(iter(wordnet_nouns_namedtuples.items())))
    print(f"Done")


if __name__ == "__main__":
    main()
