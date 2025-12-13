"""
Utilities for wikidata hierarchy
"""

import math

import pandas as pd
from loguru import logger

from entitynet.datasets.wikidata.wikidata_utils import (
    strip_label_no_quotes,
    strip_label_no_quotes_allow_multilang,
)
from entitynet.paths import get_entitynet_repo_root
from packg.iotools import dump_json, load_json
from visiontext.nlp import preprocess_text_simple


wrong_links = {
    "Q181537": set(["Q134683"]),
    # Q181537 --- Amniota | amniote | amniotes (71, small) group of tetrapods
    # Q134683 --- Reptiliomorpha (30, small) order or subclass of reptile-like amphibians
    # Reptile-like amphibians is not a parent of tetrapods (which are just animals with 4 legs)
    "Q19159": set(["Q7460384"]),
    # Q19159 --- Tetrapoda | tetrapod | tetrapods | tetrapod animals | tetrapod animal (88, small) superclass of the first four-limbed vertebrates and their descendants
    # Q7460384 --- Stegocephalia (23, small) Ancient amphibians
    # technically every animal stems from fish billions of years ago so this is not wrong
    # but we also don't want to link from tetrapods back to amphibians
    # since we want a "a parrot is bird is animal" hierarchy
    # and not a "a parrot is bird is a fish" hierarchy
}
manual_fixes = {
    "Q1047998": {  # https://www.wikidata.org/wiki/Q1047998
        "label": "Hidrobiontas",
    },
    "Q1385549": {  # Rassegeflügel https://www.wikidata.org/wiki/Q1385549
        "label": "purebred poultry"
    },
    "Q13594007": {  # Positurtümmler https://www.wikidata.org/wiki/Q13594007
        "label": "Posture tumbler pigeon",
        "desc": "Group of domestic pigeon breeds",
    },
    "Q15848080": {  # Spanische Kropftauben https://www.wikidata.org/wiki/Q15848080
        "label": "Spanish cropper pigeons",
        "desc": "Subgroup of cropper pigeons",
    },
    "Q17294326": {  # Spielflugtauben https://www.wikidata.org/wiki/Q17294326
        "label": "Performing tumbler pigeons",
    },
    "Q12160912": {  # Терофіти (ukrainian) https://www.wikidata.org/wiki/Q12160912
        "label": "therophytes",
        "desc": "These are plants in which the renewal buds are contained within the seed, from which a new plant organism develops.",
    },
    "Q25394204": {  # pied (botanique) (fr) https://www.wikidata.org/wiki/Q25394204
        "label": "individual plant",
    },
    "Q65950322": {  # Haftwurzelkletterer https://www.wikidata.org/wiki/Q65950322
        "label": "Adhesive root climbers",
        "desc": "These plants use adhesive roots to climb and attach themselves to surfaces.",
    },
    "Q65953851": {  # Blattranker https://www.wikidata.org/wiki/Q65953851
        "label": "Leaf tendrils",
        "desc": " These plants use modified leaves or leaf parts as tendrils to climb and support themselves.",
    },
    "Q65953900": {  # Blattstielranker https://www.wikidata.org/wiki/Q65953900
        "label": "Petiole tendrils",
        "desc": "These plants use their petioles (the stalk that attaches the leaf blade to the stem) as tendrils to climb and support themselves.",
    },
    "Q65953976": {  # Sproßranker https://www.wikidata.org/wiki/Q65953976
        "label": "Stem tendrils",
        "desc": "These plants use modified stems as tendrils to climb and support themselves.",
    },
}

KGRAPH_DIR = get_entitynet_repo_root() / "kgraph"
entity_types = ["animals", "plants_and_fruits"]


def check_empty(field):
    if field is None or field == "" or (isinstance(field, float) and math.isnan(field)):
        return True
    return False


def read_entity_tsv(tsv_file):
    columns = [
        "entity_id",
        "label",
        "desc",
        "links",
        "aliases",
        "common_names",
        "taxon_names",
        "images",
    ]
    logger.info(f"Reading {tsv_file=}")
    df = pd.read_csv(tsv_file, sep="\t", header=None, names=columns)
    logger.info(f"Got shape {df.shape}")
    return df


def simplify_id(entity_id):
    return entity_id.split("/")[-1].rstrip(">")


def display_raw_entity(t_entity_id, entity_item, prefix="", splitter=" --- "):
    simple_entity_id = simplify_id(t_entity_id)
    source = "large"
    if "source" in entity_item:
        source = entity_item["source"]
    name = (
        " | ".join(list(entity_item["names"].keys()))
        + f" ({entity_item['sitelinks']}, {source}) {entity_item['desc']}"
    )
    print(f"{prefix}{simple_entity_id}{splitter}{name} ")


def display_entity_and_parents(entity_key, entities_large, hierarchy_links, min_sitelinks: int = 0):
    # display_entity_new(entity_key, entities_small[entity_key])
    display_raw_entity(entity_key, entities_large[entity_key])
    t_parent_dict = resolve_parents(entity_key, hierarchy_links)
    for t_parent_id, t_depth in t_parent_dict.items():
        t_parent_item = entities_large[t_parent_id]
        sitelinks = t_parent_item["sitelinks"]
        if sitelinks < min_sitelinks:
            continue
        display_raw_entity(t_parent_id, t_parent_item, prefix="| " * t_depth)
    print()


def check_link_is_valid(child_id, parent_id):
    if child_id in wrong_links and parent_id in wrong_links[child_id]:
        return False
    return True


def resolve_parents(entity_id, hierarchy_links):
    """
    Note that this is not recursive
    """
    if entity_id not in hierarchy_links:
        return {}
    # build the initial depth 1 dict of parents
    # the dict popitem will create a LIFO stack
    # the first entry in the parents list is the most important one,
    # so we add the parents in reverse order.
    parents_here = hierarchy_links[entity_id]
    current_parent_dict = {}
    for parent_id in parents_here[::-1]:
        if check_link_is_valid(entity_id, parent_id):
            current_parent_dict[parent_id] = 1
    parent_dict = {}
    while True:
        if len(current_parent_dict) == 0:  # everything resolved
            break
        # pop the last added parent from the stack
        parent_id, t_depth = current_parent_dict.popitem()
        if parent_id in parent_dict:
            # parent was already checked, skip to avoid loops and duplicate parents
            continue
        # write the parent back to the parent collector
        parent_dict[parent_id] = t_depth
        # now select the parent's parents and add them to the stack, if they are valid connections
        parents_here = hierarchy_links.get(parent_id, [])
        for new_parent_id in parents_here[::-1]:
            if check_link_is_valid(parent_id, new_parent_id):
                current_parent_dict[new_parent_id] = t_depth + 1
    return parent_dict


def get_small_entity_set():
    # produce small entity set: the ones with downloaded images (56k, 28k each animals and plants)
    entities_small_output_file = KGRAPH_DIR / "hierarchy_wikidata_living/entities_small.json"
    if entities_small_output_file.is_file():
        entities_small = load_json(entities_small_output_file)
    else:
        logger.info(f"********** Create {entities_small_output_file.name}")
        entities_small = {}
        for entity_type in entity_types:
            entity_file = KGRAPH_DIR / f"wikidata_living/{entity_type}.tsv"
            entity_df = read_entity_tsv(entity_file)
            for index, series in entity_df.iterrows():
                if index >= 28000:
                    break
                entity_id, item = convert_entity_tuple(series, return_sitelinks=True)
                entity_id_simple = simplify_id(entity_id)
                item["type"] = entity_type
                entities_small[entity_id_simple] = item
        dump_json(
            entities_small, entities_small_output_file, create_parent=True, verbose=True, indent=2
        )
    logger.info(f"{len(entities_small)=}")
    return entities_small


def get_medium_entity_set():
    # produce medium entity set: all downloadable ones (290k, 205k animals and 85k plants)
    entities_medium_output_file = KGRAPH_DIR / "hierarchy_wikidata_living/entities_medium.json"
    if entities_medium_output_file.is_file():
        entities_medium = load_json(entities_medium_output_file)
    else:
        logger.info(f"********** Create {entities_medium_output_file.name}")
        entities_medium = {}
        for entity_type in entity_types:
            entity_file = KGRAPH_DIR / f"wikidata_living/{entity_type}.tsv"
            entity_df = read_entity_tsv(entity_file)
            for index, series in entity_df.iterrows():
                entity_id, item = convert_entity_tuple(series, return_sitelinks=True)
                entity_id_simple = simplify_id(entity_id)
                item["type"] = entity_type
                entities_medium[entity_id_simple] = item
        dump_json(
            entities_medium, entities_medium_output_file, create_parent=True, verbose=True, indent=2
        )
    logger.info(f"{len(entities_medium)=}")
    return entities_medium


def get_large_entity_set(add_missing=True):
    # produce large entity set: all the ones that appear in the hierarchy
    mst = "_nomissing" if not add_missing else ""
    entities_large_output_file = KGRAPH_DIR / f"hierarchy_wikidata_living/entities_large{mst}.json"
    if entities_large_output_file.is_file():
        entities_large = load_json(entities_large_output_file)
    else:
        logger.info(f"********** Create {entities_large_output_file.name}")
        entities_large = {}
        for entity_type in entity_types:
            entity_file = KGRAPH_DIR / f"wikidata_living/{entity_type}.hierarchy.tsv"
            entity_df = read_entity_tsv(entity_file)
            for index, series in entity_df.iterrows():
                entity_id, item = convert_entity_tuple(series, return_sitelinks=True)
                entity_id_simple = simplify_id(entity_id)
                item["type"] = entity_type
                entities_large[entity_id_simple] = item
            if add_missing:
                entity_missing_file = KGRAPH_DIR / f"wikidata_living/{entity_type}.missing.tsv"
                if not entity_missing_file.is_file():
                    raise FileNotFoundError(
                        f"{entity_missing_file} not found, either set add_missing=False or run "
                        f"wikidata_find_missing.py to create the file."
                    )
                entity_missing_df = read_entity_tsv(entity_missing_file)
                for index, series in entity_missing_df.iterrows():
                    entity_id, item = convert_entity_tuple(series, return_sitelinks=True)
                    entity_id_simple = simplify_id(entity_id)
                    item["type"] = entity_type
                    entities_large[entity_id_simple] = item
        dump_json(
            entities_large, entities_large_output_file, create_parent=True, verbose=True, indent=2
        )
    logger.info(f"{len(entities_large)=}")
    return entities_large


def convert_entity_tuple(series, return_sitelinks=False, return_images=False):
    entity_id, label, desc, n_links, aliases, common_names, taxon_names, images = series
    simple_entity_id = simplify_id(entity_id)

    possible_names_and_sources = [
        (aliases, "aliases"),
        (common_names, "common_names"),
        (taxon_names, "taxon_names"),
    ]

    # first apply the manual fixes from below
    if simple_entity_id in manual_fixes:
        logger.debug(f"Manually fixing {simple_entity_id}!")
        fix_data = manual_fixes[simple_entity_id]
        if "label" in fix_data:
            label = fix_data["label"] + "@en"
        if "desc" in fix_data:
            desc = fix_data["desc"] + "@en"

    if check_empty(label):
        # very rare special case that there is no label.
        pseudo_label = None
        # try to convert the first thing we find into a label.
        for names_str, source in possible_names_and_sources:
            if check_empty(names_str):
                continue
            names_str = strip_label_no_quotes_allow_multilang(names_str, assert_has_lang=False)
            names = names_str.split(";")
            for name in names:
                stripped_name = preprocess_text_simple(name)
                if stripped_name == "":
                    # some names are only utf-8 smileys, skip them
                    continue
                pseudo_label = name
                logger.debug(f"{entity_id}: got label {pseudo_label} from field {source}")
                break
            if pseudo_label is not None:
                break
        if pseudo_label is None:
            raise RuntimeError(
                f"{entity_id} has no label and no other fields and also was not manually fixed. "
                f"Full data: {series}"
            )
        label = pseudo_label
    else:
        # regular case, we have a label
        # allow multilang hiere since for some parts of the hierarchy we are now downloading
        # whatever we can find in case there is no english label
        label = strip_label_no_quotes_allow_multilang(label, assert_has_lang=True)

    # parse all other fields. skip any duplicate names
    if check_empty(desc):
        desc = None
    else:
        desc = strip_label_no_quotes(desc, assert_has_lang=True)
    if check_empty(n_links):
        n_links = 0
    n_links = int(n_links)
    output_names_and_source = {label: "label"}
    for names_str, source in possible_names_and_sources:
        if check_empty(names_str):
            continue
        names_str = strip_label_no_quotes_allow_multilang(names_str, assert_has_lang=False)
        names = names_str.split(";")
        for name in names:
            if name in output_names_and_source:
                continue
            stripped_name = preprocess_text_simple(name)
            if stripped_name in output_names_and_source:
                continue
            if stripped_name == "":
                # some names are only utf-8 smileys, skip them
                continue
            output_names_and_source[name] = source
    # print(images)

    item = {
        "names": output_names_and_source,
        "desc": desc,
    }
    if return_sitelinks:
        item["sitelinks"] = n_links
    if return_images:
        item["images"] = images
    return entity_id, item
