"""
Loader class for wikidata hierarchy
"""

from collections import defaultdict

import numpy as np
import pandas as pd
from natsort import natsorted

from packg.iotools import load_json

from entitynet.config.task_config import EntityNetTextAugCfg
from entitynet.datasets.wikidata.wikidata_hierarchy import KGRAPH_DIR

ParentTuple = tuple[str, int]  # parent_id like Q756, depth (0 is the entity itself)


class WikidataLoader:
    """
    hyperparameters:
        - pruning parents with less than min_sitelinks, default 40
        - log rebalance probabilities, to give rare parents a higher chance, default 0.5
        - how often to replace the entity with one of its parents, TBD, default 0.2

    hierarchy format: dict child_id: list of tuple (parent_id, depth)
        this contains all the leaves as keys (entities we downloaded images for)
    entity format: entity_id, names: {value -> source}, desc: value, sitelinks: int,
        type: animals or plants_and_fruits, source: small for the 56k downloaded set, medium, large

    to balance out the sampling of the parents:
        N: the total sum of all occurrences of all parents
        k: the number of unique parents
        n_i: number of occurrences in all hierarchies of parent i
        p_i: n_i / N, the base probability of selecting parent i, on average for any
            datapoint's hierarchy.
        p_{target}: 1 / k, the target probability of selecting any parent, such that they are all selected equally
        m_i: multiplication factor for each parent to achieve the same probability:
        m_i = p_{target} / p_i = (1 / k) / (n_i / N) = N / (k * n_i)

        however, this yet ignores the fact that entities have different lengths of parents
        we can also calculate
        q_i: the actual probability of selecting parent i, taking into account selecting a random
            entity first and just checking the probabilities for all entities
            with q_i = qu_i / qn_i  # without normalization and the normalization factor
        r_i: the actual multiplication factor
            p_{target} / q_i = (1 / k) / (qu_i / qn_i) = qn_i / (k * qu_i)
    """

    def __init__(self, text_aug: EntityNetTextAugCfg | None = None):
        if text_aug is None:
            text_aug = EntityNetTextAugCfg()

        ann_dir = KGRAPH_DIR / "hierarchy_wikidata_living"
        entities = load_json(ann_dir / "entities_relevant.json")
        hierarchy: dict[str, list[tuple[str, int]]] = load_json(ann_dir / "hierarchy.json")
        # logger.info(f"Create wikidata hierarchy with {len(hierarchy)} leafs")  # ~56k
        # logger.debug("no parents:", sum(len(parents) == 0 for parents in hierarchy.values()))  # 85

        # first, prune away everything in the hierarchy that has not enough sitelinks
        # modify the hierarchy in place
        if text_aug.min_sitelinks > 0:
            for leaf_id in list(hierarchy.keys()):
                new_hierarchy_row = []
                for parent_id, depth in hierarchy[leaf_id]:
                    parent_item = entities[parent_id]
                    sitelinks = parent_item["sitelinks"]
                    if sitelinks < text_aug.min_sitelinks:
                        continue
                    new_hierarchy_row.append((parent_id, depth))
                hierarchy[leaf_id] = new_hierarchy_row

        # second, count how often the parents appear in the hierarchy
        parent_counter = defaultdict(int)
        parent_probs_unnorm = defaultdict(float)
        for leaf_id, parent_list in hierarchy.items():
            n_parents = len(parent_list)
            for parent_id, depth in parent_list:
                parent_counter[parent_id] += 1
                # if we would uniform sample once for each entity:
                parent_probs_unnorm[parent_id] += 1 / n_parents
        # print(f"{sum(parent_probs_unnorm.values() )} probs")  # sums to ~num_leafs
        parent_probs_sum = sum(parent_probs_unnorm.values())

        n_unique_parents = len(parent_counter)
        # print(f"{n_unique_parents} unique parents")  # ~1.8k at min_sitelinks=40
        n_total_occurrences = sum(parent_counter.values())
        # print(f"{n_total_occurrences} total occurrences")  # ~598k

        # calculate the multiplication factors that would completely balance the parent sampling
        parent_prob_correction_fator = {}
        for parent_id, n_occurrences in parent_counter.items():
            parent_prob_correction_fator[parent_id] = parent_probs_sum / (
                n_unique_parents * parent_probs_unnorm[parent_id]
            )

        # precompute final probability values for each entity
        prob_values: dict[str, np.ndarray | None] = {}
        for entity_id, parent_list in hierarchy.items():
            n_parents = len(parent_list)
            if n_parents == 0:
                # ~0.1% of the entities have no parents, ignore and return the entity itself
                prob_values[entity_id] = None
                continue
            base_probs = np.ones(n_parents) / n_parents
            mult_factors = []
            for parent_id, depth in parent_list:
                parent_corr_factor = parent_prob_correction_fator[parent_id]
                mult_factors.append(parent_corr_factor)
            mult_factors = np.array(mult_factors)
            lambd = text_aug.hierarchy_balancing_strength
            final_probs = np.exp((1 - lambd) * np.log(base_probs) + lambd * np.log(mult_factors))
            final_probs = final_probs / final_probs.sum()
            prob_values[entity_id] = final_probs

        # in order to find all children for a given hierarchy, build an inverted list
        parent2children = defaultdict(list)
        for entity_id, parent_list in hierarchy.items():
            for parent_tuple in parent_list:
                parent_id = parent_tuple[0]
                parent2children[parent_id].append(entity_id)
        for parent_id in list(parent2children.keys()):
            parent2children[parent_id] = natsorted(set(parent2children[parent_id]))

        self.text_aug = text_aug
        self.hierarchy = hierarchy
        self.hierarchy_probs: dict[str, np.ndarray] = prob_values
        self.entities = entities
        self.parent2children = parent2children

    def randomly_select_parent(self, entity_id, rng=np.random):
        parent_probs = self.hierarchy_probs[entity_id]
        if parent_probs is None:
            return entity_id, 0
        parent_list = self.hierarchy[entity_id]

        random_parent_idx = rng.choice(len(parent_list), p=parent_probs)
        # random_parent_idx = 0
        parent_entity_id, parent_depth = parent_list[random_parent_idx]
        return parent_entity_id, parent_depth

    def display_entity(self, entity_id, prefix="", splitter=" --- ", do_print=True):
        entity_item = self.entities[entity_id]
        display_entity(entity_id, entity_item, prefix=prefix, splitter=splitter, do_print=do_print)

    # TODO add global entities


def display_entity(entity_id, entity_item, prefix="", splitter=" --- ", do_print=True):
    # source = entity_item["source"]
    etype = entity_item["type"]
    sitelinks = entity_item["sitelinks"]
    desc = entity_item["desc"]
    all_names = list(entity_item["names"].keys())
    print_name = " | ".join(all_names) + f" ({sitelinks}x, {desc})"
    out_str = f"{prefix}{entity_id}{splitter}{print_name}"
    if do_print:
        print(out_str)
    return out_str


def display_entity_and_parent_list(entity_id, entities, hierarchy, do_print=True, prefix="| "):
    entity_item = entities[entity_id]
    out_strs = [display_entity(entity_id, entity_item, do_print=do_print)]
    for parent_tuple in hierarchy[entity_id]:
        parent_id, parent_depth = parent_tuple[:2]
        parent_item = entities[parent_id]
        out_strs.append(
            display_entity(parent_id, parent_item, prefix=prefix * parent_depth, do_print=do_print)
        )
    return out_strs


def simplify_wikidata_id(entity_id):
    return entity_id.split("/")[-1].rstrip(">")


def load_tsv_simple(path: str) -> list[list[str]]:
    """does not automatically support quoted strings: it will return the quotes, too."""
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def load_entity_tsv_to_df(path: str):
    """load the raw entity tsvs"""
    wd_raw = load_tsv_simple(path)
    new_data = []
    for wdr in wd_raw:
        if len(wdr) == 5:
            entity_full, name_raw, defi_raw, sitelinks, synonyms_raw = wdr
            imageurls_raw = ""
        else:
            entity_full, name_raw, defi_raw, sitelinks, synonyms_raw, imageurls_raw = wdr
        entity = simplify_wikidata_id(entity_full)
        name = name_raw[1:-4]
        sitelinks = int(sitelinks)
        if defi_raw is None or len(defi_raw) == 0:
            description = None
        else:
            description = defi_raw[1:-4]
        if synonyms_raw is None or len(synonyms_raw) == 0:
            synonyms = []
        else:
            synonyms = synonyms_raw[1:-1].split(";;;")
        if imageurls_raw is None or len(imageurls_raw) == 0:
            imageurls = []
        else:
            imageurls = imageurls_raw.split(";;;")
        new_data.append((entity, name, description, sitelinks, synonyms, imageurls))
    wd_pd = pd.DataFrame(
        new_data,
        columns=["entity", "entity_name", "description", "sitelinks", "synonyms", "images"],
    )
    return wd_pd
