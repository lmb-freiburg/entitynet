"""
Run the WikidataLoader and analyze the hierarchy distributions with different balancing strengths.
"""

from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from attrs import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.tqdmext import tqdm_max_ncols
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.config.task_config import EntityNetTextAugCfg
from entitynet.datasets.wikidata.wikidata_hierarchies_loader import WikidataLoader


@define
class Args(VerboseQuietArgs):
    check_hierarchy_distribution: bool = add_argument(shortcut="-c", action="store_true")


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    wdl = WikidataLoader(EntityNetTextAugCfg(min_sitelinks=40))
    logger.info(f"Loading default wikidataloader was successful, got {len(wdl.hierarchy)} leafs")

    # take an entity and sample it's parents in the hierarchy
    entity_id = "Q38280"
    wdl.display_entity(entity_id)
    for _ in range(5):
        parent_id, parent_depth = wdl.randomly_select_parent(entity_id)
        wdl.display_entity(parent_id, prefix=f"  Depth {parent_depth} --- ")

    if args.check_hierarchy_distribution:
        check_hierarchy_distribution()


def check_hierarchy_distribution():
    logger.info(f"Check hierarchy distributions for different balancing strengths")
    parent_ids_sorted = None
    for lambd in [0.0, 0.5, 1.0]:
        print(f"lambd={lambd}")
        wdl = WikidataLoader(
            EntityNetTextAugCfg(min_sitelinks=40, hierarchy_balancing_strength=lambd)
        )
        parent_counter = defaultdict(int)
        for entity_id in tqdm_max_ncols(list(wdl.hierarchy.keys())):
            size = 1000
            # for this experiment it's faster to batch sample
            parent_probs = wdl.hierarchy_probs[entity_id]
            if parent_probs is None:
                continue
            parent_list = wdl.hierarchy[entity_id]
            random_parent_idx_set = np.random.choice(len(parent_list), p=parent_probs, size=size)
            for random_parent_idx in random_parent_idx_set:
                parent_entity_id, parent_depth = parent_list[random_parent_idx]
                # for _ in range(size):
                #     parent_entity_id, parent_depth = wdl.randomly_select_parent(entity_id)
                parent_counter[parent_entity_id] += 1
        parent_counter_sorted = Counter(parent_counter).most_common()
        # ----- print
        n_parents = len(parent_counter)  # here it's always 1841 parents
        if parent_ids_sorted is None:
            # sort only the first time with lambda=0, so the x axis stays constant
            # then we can really see how the probs shift
            parent_ids_sorted = [x[0] for x in parent_counter_sorted]
            new_parent_ids_sorted = parent_ids_sorted
            ranks_to_show = list(range(10)) + list(range(n_parents - 10, n_parents))
        else:
            new_parent_ids_sorted = [x[0] for x in parent_counter_sorted]
            # show both the old and new top and bottom 10
            ids_to_show = sorted(
                set(
                    parent_ids_sorted[:10]
                    + parent_ids_sorted[-10:]
                    + new_parent_ids_sorted[:10]
                    + new_parent_ids_sorted[-10:]
                )
            )
            id2rank = {parent_id: i for i, parent_id in enumerate(new_parent_ids_sorted)}
            ranks_to_show = sorted([id2rank[parent_id] for parent_id in ids_to_show])

        for parent_idx in ranks_to_show:
            parent_id = new_parent_ids_sorted[parent_idx]
            parent_count = parent_counter[parent_id]
            parent_item = wdl.entities[parent_id]
            parent_name = next(iter(parent_item["names"].keys()))
            print(f"    {parent_idx:4d} C={parent_count:7d} {parent_id:10s} {parent_name}")

        plt.plot([parent_counter[i] for i in parent_ids_sorted], label=f"lambd={lambd}", alpha=0.7)
    plt.xlabel("parents")
    plt.ylabel("occurrences")
    plt.legend()
    plt.semilogy()
    plt.show()
    print("done")

    # # TODO look at depths
    # depths = []
    # depth_counter = Counter(depths)
    # print(f"    {depth_counter.most_common(10)}")
    # print(pd.Series(depths).describe())


if __name__ == "__main__":
    main()
