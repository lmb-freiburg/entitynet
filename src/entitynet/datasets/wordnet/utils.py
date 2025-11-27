"""
Field names:
    - wnid: n00001740
    - synname: entity.n.01
    - synset: the actual synset object Synset('entity.n.01')

"""

from __future__ import annotations

import os

import nltk
import pandas as pd
from loguru import logger
from nltk.corpus import wordnet as wn
from nltk.corpus.reader import Synset  # noqa

from packg import Const
from packg.iotools import dump_json, load_json
from packg.tqdmext import tqdm_max_ncols
from typedparser.objects import invert_list_of_dict
from visiontext.nlp import preprocess_text_simple

from entitynet.paths import get_entitynet_cache_dir

POS_DICT = {
    "n": "noun",
    "v": "verb",
    "a": "adjective",
    "s": "satellite adjective",
    "r": "adverb",
}


class NodeTypes(Const):
    ROOT = "root"  # no parents, has children
    LEAF = "leaf"  # no children, has "type of" parents
    INSTANCE = "instance"  # no children, has "instance of" parents
    INTERNAL = "internal"  # has children, has parents
    STANDALONE = "standalone"  # no children, no parents


def ensure_wordnet_is_downloaded():
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download("wordnet")


def get_all_synsets(pos: str | list[str] | None = None) -> list[Synset]:
    if pos is None or isinstance(pos, str):
        return list(wn.all_synsets(pos=pos))
    # pos is a list, but some synsets may be duplicate, have to deduplicate them
    synset_dict = {}
    for p in pos:
        for synset in wn.all_synsets(pos=p):
            synset_dict[synset.name()] = synset
    return list(synset_dict.values())


cache_dict = {}


def ensure_mapping_from_wnid_to_synname(use_cache=True):
    """
    Load mapping from wordnet id to synset id

    Returns:
        dictionary of entries like
        {
            "n00001740": "entity.n.01",
            ...
        }
    """
    fn_cache_key = "ensure_mapping_from_wnid_to_synname"
    if use_cache and fn_cache_key in cache_dict:
        return cache_dict[fn_cache_key]

    ensure_wordnet_is_downloaded()
    cache_file = get_entitynet_cache_dir() / "wordnet/wnid_synname.json"
    if use_cache and cache_file.is_file():
        wnid_to_synname = load_json(cache_file)
    else:
        mapping = {}
        synnames = set()
        for synset in tqdm_max_ncols(wn.all_synsets(), desc="Create wordnet mapping"):
            wnid = convert_synset_to_wnid(synset)
            synname = synset.name()
            if synname in synnames:
                logger.error(f"Duplicate synname: {synname}")
            synnames.add(synname)
            mapping[wnid] = synname
        dump_json(mapping, cache_file, create_parent=True)
        wnid_to_synname = mapping

    synname_to_wnid = {v: k for k, v in wnid_to_synname.items()}
    assert len(wnid_to_synname) == len(synname_to_wnid), (
        f"Mapping is not bijective. Length of wnid_to_synname: {len(wnid_to_synname)}, "
        f"length of synname_to_wnid: {len(synname_to_wnid)}"
    )

    cache_dict[fn_cache_key] = wnid_to_synname, synname_to_wnid
    return wnid_to_synname, synname_to_wnid


def convert_wnid_to_synname(wnid):
    wnid_to_synname, synname_to_wnid = ensure_mapping_from_wnid_to_synname()
    return wnid_to_synname[wnid]


def convert_synname_to_wnid(synname):
    wnid_to_synname, synname_to_wnid = ensure_mapping_from_wnid_to_synname()
    return synname_to_wnid[synname]


def convert_synset_to_wnid(synset: Synset):
    pos = synset.pos()
    offset = synset.offset()
    wnid = f"{pos}{offset:08d}"
    return wnid


def convert_wnid_to_synset(wnid):
    pos = wnid[0]
    offset = int(wnid[1:])
    synset = wn.synset_from_pos_and_offset(pos, offset)
    return synset


def convert_synname_to_synset(synname):
    return wn.synset(synname)


def convert_synset_to_synname(synset):
    return synset.name()


def display_synset(synset: Synset) -> str:
    synname = convert_synset_to_synname(synset)
    wnid = convert_synname_to_wnid(synname)
    out = f"{synname} ({wnid}): {' | '.join(synset.lemma_names())} - {synset.definition()}"
    return out


def display_synset_from_synname(synname: str) -> str:
    synset = convert_synname_to_synset(synname)
    return display_synset(synset)


def display_synset_from_wnid(wnid: str) -> str:
    synset = convert_wnid_to_synset(wnid)
    return display_synset(synset)


def load_wordnet_as_dataframe(pos: str | list[str] | None = None, use_cache: bool = True):
    if pos is None:
        pos_str = "all"
    elif isinstance(pos, str):
        pos_str = pos
    else:
        pos_str = "_".join(pos)

    fn_cache_key = f"load_wordnet_as_dataframe_{pos_str}"
    if use_cache and fn_cache_key in cache_dict:
        return cache_dict[fn_cache_key]

    cache_file = get_entitynet_cache_dir() / f"wordnet/wordnet-{pos_str}.parquet"
    if use_cache and cache_file.is_file():
        df = pd.read_parquet(cache_file)
        return df

    synsets = get_all_synsets(pos=pos)
    rows = []
    for synset in synsets:
        lemmas, lemmas_clean = [], []
        for lemma in synset.lemma_names():
            lemma = lemma.replace("_", " ").strip()
            assert "|" not in lemma, f"Lemma {lemma} contains |"
            lemma_clean = preprocess_text_simple(lemma, lowercase=False)
            lemmas.append(lemma)
            lemmas_clean.append(lemma_clean)
        lemmas = [""] + lemmas + [""]
        lemmas_clean = [""] + lemmas_clean + [""]
        lemma_str = " | ".join(lemmas).strip()
        lemmas_clean_str = " | ".join(lemmas_clean).strip()
        row = {
            "synname": convert_synset_to_synname(synset),
            "wnid": convert_synset_to_wnid(synset),
            "pos": synset.pos(),
            "definition": synset.definition(),
            "lemmas": lemma_str,
            "lemmas_clean": lemmas_clean_str,
        }
        rows.append(row)
    df = pd.DataFrame(invert_list_of_dict(rows))
    df = df.set_index("synname")
    os.makedirs(cache_file.parent, exist_ok=True)
    df.to_parquet(cache_file)

    cache_dict[fn_cache_key] = df
    return df


def search_wordnet_column(term, case=False, regex=False, column_name="lemmas") -> pd.Series:
    df = load_wordnet_as_dataframe()
    return df[df[column_name].str.contains(term, case=case, regex=regex)].index


def search_wordnet_synnames(term, case=False, regex=False) -> pd.Series:
    df = load_wordnet_as_dataframe()
    return df[df.index.str.contains(term, case=case, regex=regex)].index


RELATIONS = [
    "hypernyms",
    "hyponyms",
    "also_sees",
    "attributes",
    "causes",
    "entailments",
    "in_region_domains",
    "in_topic_domains",
    "in_usage_domains",
    "instance_hypernyms",
    "instance_hyponyms",
    "member_holonyms",
    "member_meronyms",
    "part_holonyms",
    "part_meronyms",
    "region_domains",
    "similar_tos",
    "substance_holonyms",
    "substance_meronyms",
    "topic_domains",
    "usage_domains",
    "verb_groups",
]


def check_hyponyms_hypernyms_are_equivalent():
    """
    This function checks if all parent-child (hypernym-hyponym) relations are equivalent.
    2 errors total
        inhibit.v.04 is hyponyms of restrain.v.01 but hypernyms of child are {'suppress.v.04'}
        suppress.v.04 is hypernyms of inhibit.v.04 but hyponyms of parent are {'swallow.v.06'}
    so if we ever build a verb tree just add this manually.

    """
    n_errors = 0
    for parent_fn, child_fn in (
        ("hypernyms", "hyponyms"),
        ("instance_hypernyms", "instance_hyponyms"),
    ):
        for synset in get_all_synsets():
            synname = convert_synset_to_synname(synset)
            children = getattr(synset, child_fn)()
            for child_synset in children:
                child_synname = convert_synset_to_synname(child_synset)
                child_parent_synsets = getattr(child_synset, parent_fn)()
                child_parent_synnames = set(
                    [convert_synset_to_synname(p) for p in child_parent_synsets]
                )
                if synname not in child_parent_synnames:
                    logger.error(
                        f"{child_synname} is {child_fn} of {synname} but "
                        f"{parent_fn} of child are {child_parent_synnames}"
                    )
                    n_errors += 1
            parents = getattr(synset, parent_fn)()
            for parent_synset in parents:
                parent_synname = convert_synset_to_synname(parent_synset)
                parent_child_synsets = getattr(parent_synset, child_fn)()
                parent_child_synnames = set(
                    [convert_synset_to_synname(p) for p in parent_child_synsets]
                )
                if synname not in parent_child_synnames:
                    logger.error(
                        f"{parent_synname} is {parent_fn} of {synname} but "
                        f"{child_fn} of parent are {parent_child_synnames}"
                    )
                    n_errors += 1
    logger.error(f"Number of errors in parent-child equivalence: {n_errors}")


def main():
    check_hyponyms_hypernyms_are_equivalent()


if __name__ == "__main__":
    main()
