"""
TODO fix

TODO if reloading old large entities, it was missing 65k, now it's missing only like 200.

Stats 2024:
    before adding missing:
    20240718 12:21:37 INFO len(entities_small)=55999
    20240718 12:21:38 INFO len(entities_medium)=289528
    20240718 12:21:52 INFO len(entities_large)=3325975
    20240718 12:21:52 INFO Total len(entities_large)=3326313
    20240718 12:21:54 INFO After simplifying ids: 3326313
    afterwards, 43 more: len(entities_large)=3326356
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from attrs import define
from loguru import logger

from packg.iotools import load_json
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args, logger
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.datasets.wikidata.qleverutils import (
    query_living_entities_batched,
    query_qlever_robust,
)
from entitynet.datasets.wikidata.wikidata_hierarchy import (
    KGRAPH_DIR,
    entity_types,
    get_large_entity_set,
    get_medium_entity_set,
    get_small_entity_set,
    simplify_id,
)


@define
class Args(VerboseQuietArgs):
    base_dir: Optional[Path] = add_argument(
        shortcut="-b", type=str, help="Source base dir", default=None
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    entities_small = get_small_entity_set()
    entities_medium = get_medium_entity_set()
    entities_large = get_large_entity_set(add_missing=False)
    entities_large = entities_large
    for g_entity_id, g_item in entities_medium.items():
        g_item["source"] = "medium"
        entities_large[g_entity_id] = g_item
    for g_entity_id, g_item in entities_small.items():
        g_item["source"] = "small"
        entities_large[g_entity_id] = g_item
    logger.info(f"Total {len(entities_large)=}")
    logger.info(f"After simplifying ids: {len(set(simplify_id(a) for a in entities_large.keys()))}")

    # integrity check
    for g_entity_id, g_item in entities_large.items():
        label, source = list(g_item["names"].items())[0]
        assert source == "label", f"{source=} {label=} {g_item=} {g_entity_id=}"
        assert isinstance(label, str) and len(label) > 0, f"{label=} {g_item=} {g_entity_id=}"

    # load the hierarchy(ies)
    hierarchy_links = {}
    g_all_ids = []
    miss_ids_per_type = {}
    for g_entity_type in entity_types:
        g_hierarchy_links_file = f"kgraph/wikidata_living/{g_entity_type}.hierarchy.json"
        g_hierarchy_links_ent = load_json(g_hierarchy_links_file)
        miss_ids = []
        for g_entity_id, g_parent_ids in g_hierarchy_links_ent.items():
            g_entity_id = simplify_id(g_entity_id)
            g_parent_ids = [simplify_id(p) for p in g_parent_ids]
            hierarchy_links[g_entity_id] = g_parent_ids

            # check for missing entities
            g_ids_here = [g_entity_id] + g_parent_ids
            g_all_ids += g_ids_here
            for g_entity_id_to_check in g_ids_here:
                if g_entity_id_to_check not in entities_large:
                    miss_ids.append(g_entity_id_to_check)
        miss_ids_per_type[g_entity_type] = miss_ids
        logger.warning(
            f"Entity type {g_entity_type} missing entities: {len(miss_ids)} unique {len(set(miss_ids))}"
        )
    g_all_ids = sorted(set(g_all_ids))
    logger.info(
        f"Got {len(hierarchy_links)} links from 1 child to N parents. "
        f"Total parents {sum(len(v) for v in hierarchy_links.values())}"
    )
    logger.info(f"Got {len(g_all_ids)} unique entities in the hierarchy")
    logger.info(f"Got information about {len(entities_large)} entities in the large set")

    # find missing
    g_miss_ids_all = {}
    for g_entity_type, g_miss_ids_here in miss_ids_per_type.items():
        for g_miss_id in sorted(set(g_miss_ids_here)):
            # assert (
            #     g_miss_id not in g_miss_ids_all
            # ), f"Duplicate {g_miss_id} missing in multiple entity_types."
            g_miss_ids_all[g_miss_id] = g_entity_type
    print(f"Total missing entities: {len(g_miss_ids_all)}")

    # request missing
    requests = defaultdict(list)
    for entity_id, supertype in g_miss_ids_all.items():
        requests[supertype].append(f"wd:{entity_id}")
    requests = dict(requests)

    for supertype, entity_ids in requests.items():
        logger.info(f"Supertype {supertype} has {len(entity_ids)} missing entities")
        all_results = query_living_entities_batched(entity_ids)
        output_file = KGRAPH_DIR / f"wikidata_living/{supertype}.missing.tsv"
        logger.info(f"Writing {output_file} with {len(all_results)} results")
        if output_file.is_file():
            logger.error(f"File exists. Not writing {output_file}")
        else:
            with open(output_file, "w", encoding="utf-8") as f:
                for line in all_results:
                    f.write("\t".join(line) + "\n")


def print_sitelink_statistics(entities_small, entities_medium, entities_large):
    for entity_dict, entity_dict_name in [
        [entities_small, "small"],
        [entities_medium, "medium"],
        [entities_large, "large"],
    ]:
        for g_entity_type in [None] + entity_types:
            g_entity_type_str = "all" if g_entity_type is None else g_entity_type
            print(
                f"---------- entity group: {entity_dict_name}, total size: {len(entity_dict)}, "
                f"subset: {g_entity_type_str}"
            )

            g_sitelinks_coll = []
            for g_entity_id, g_item in entity_dict.items():
                if g_entity_type is None or g_entity_type == g_item["type"]:
                    g_sitelinks = g_item["sitelinks"]
                    g_sitelinks_coll.append(g_sitelinks)
            # print(pd.Series(g_sitelinks_coll).describe())
            print(f"count {len(g_sitelinks_coll):9_d}")
            arr = np.array(g_sitelinks_coll)
            print(f"mean {arr.mean():14_.3f}")
            print(f"std  {arr.std():14_.3f}")
            print(f"min   {arr.min():9_.0f}")
            arr_sorted = np.sort(arr)
            percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]
            for p in percentiles:
                perc_index = int(p * len(arr_sorted))
                print(f"{p * 100:3.0f}%  {arr_sorted[perc_index]:9_.0f}")
            print(f"max   {arr.max():9_.0f}")

            # plt.figure(figsize=(4,4))
            # plt.plot(sorted(g_sitelinks_coll)[::-1])
            # plt.semilogy()
            # plt.xlabel("Entity number")
            # plt.ylabel("Sitelinks")
            # plt.title("Sitelinks per entity")
            # plt.show()


if __name__ == "__main__":
    main()
