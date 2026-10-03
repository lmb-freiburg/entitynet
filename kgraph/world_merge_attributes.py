"""
merge llm-generated attributes, skip some invalid ones, and split into include/exclude files
given some max limit of queries.

use -v to view invalid attributes.

python misc/scripts_world/merge_attributes_v3.py \
    misc/wikidata_world/world-entities.attributes.gpt-4o.json \
    misc/wikidata_world/world-entities.attributes.qwen2.5-7b.json
"""

import json
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy

from attrs import define
from crossm.paths import get_crossm_repo_root
from loguru import logger

from packg.iotools import dump_json, dump_jsonl
from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.strings import dict_to_str_comma_equals
from typedparser import TypedParser, VerboseQuietArgs, add_argument

_world_dir = get_crossm_repo_root() / "misc" / "wikidata_world"


@define
class Args(VerboseQuietArgs):
    files: list[str] = add_argument(
        positional=True,
        nargs="+",
        help="Attribute files to merge, best to worst",
    )
    max_queries: int = add_argument(
        type=int, help="Maximum number of queries to process", default=92000
    )
    output_include: str = add_argument(
        shortcut="-oi",
        type=str,
        help="Output file for queries to include",
        required=True,
    )
    output_exclude: str = add_argument(
        shortcut="-oe",
        type=str,
        help="Output file for queries to exclude",
        required=True,
    )


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    # merge multiple llm outputs
    merged = {}
    total = 0
    n_skip_invalid = 0
    for file in args.files:
        with open(file, "r") as inf:
            data = json.load(inf)
        for entity, attributes in data.items():
            if entity not in merged:
                merged[entity] = {}
            for cat, att_list in attributes.items():
                if cat not in merged[entity]:
                    merged[entity][cat] = []
                for att in att_list:
                    if is_none(att):
                        logger.debug(f"SKIP {dict_to_str_comma_equals(att)}")
                        n_skip_invalid += 1
                        continue
                    if contains(merged[entity][cat], att):
                        continue
                    merged[entity][cat].append(att)
        n = num_atts(data)
        logger.info(f"Processed {file}: {n:,} attributes for {len(data):_} entities")
        total += n
    logger.info(f"{n_skip_invalid=}", file=sys.stderr)
    logger.info(
        f"After merge, before deleting empties: {num_atts(merged):_} attrs for {len(merged):_} ents"
    )
    delete_empty(merged)
    logger.info(f"After deleting empties: {num_atts(merged):_} attrs for {len(merged):_} ents")
    n = num_atts(merged)
    logger.info(f"Reduced {total:_} attributes to {n:,} ({total - n:_} duplicates removed)")
    logger.info(f"Average attributes per entity: {n / len(merged):.2f}")
    n_ents_cats = sum(len(attributes) for attributes in merged.values())
    logger.info(f"Total entity-category combinations: {n_ents_cats:_}")
    att_per_cat = n / n_ents_cats
    logger.info(f"Average attributes per category: {att_per_cat:.2f}")
    logger.info(f"----------")

    # limit number of attributes per entity-category combination to stay under max queries, step 1
    keep_per_ent_cat = math.ceil(args.max_queries / n_ents_cats)
    logger.info(f"Keeping {keep_per_ent_cat:_} attributes per entity-category combination")
    inc, exc = defaultdict(dict), defaultdict(dict)
    random.seed(42)
    if n > args.max_queries:
        logger.info(f"Subsampling step 1 to {args.max_queries:,} queries")
        for entity, attributes in merged.items():
            for cat, att_list in attributes.items():
                att_list_shuffled = deepcopy(att_list)
                random.shuffle(att_list_shuffled)
                inc[entity][cat] = att_list_shuffled[:keep_per_ent_cat]
                rem = att_list_shuffled[keep_per_ent_cat:]
                if len(rem) > 0:
                    exc[entity][cat] = rem
    logger.info(f"Include: {num_atts(inc):_} attributes for {len(inc):_} entities")
    logger.info(f"Exclude: {num_atts(exc):_} attributes for {len(exc):_} entities")
    logger.info(f"Delete empties...")
    delete_empty(inc)
    delete_empty(exc)
    logger.info(f"Include: {num_atts(inc):_} attributes for {len(inc):_} entities")
    logger.info(f"Exclude: {num_atts(exc):_} attributes for {len(exc):_} entities")
    logger.info(f"----------")

    # throw away more attributes to meet the limit exactly
    n = num_atts(inc)
    if n > args.max_queries:
        logger.info(f"Subsampling step 2 to {args.max_queries:,} queries")
        to_remove = n - args.max_queries
        for entity in list(inc.keys())[::-1]:
            attributes = inc[entity]
            for cat, att_list in attributes.items():
                # print(att_list)
                inc[entity][cat] = att_list[:-1]
                if cat not in exc[entity]:
                    exc[entity][cat] = []
                exc[entity][cat].append(att_list[-1])
                to_remove -= 1
                if to_remove == 0:
                    break
            if to_remove == 0:
                break
    logger.info(f"Step 2 Include: {num_atts(inc):_} attributes for {len(inc):_} entities")
    logger.info(f"Step 2 Exclude: {num_atts(exc):_} attributes for {len(exc):_} entities")
    inc, exc = dict(inc), dict(exc)
    # print(json.dumps(merged, indent=2))
    dump_json(inc, args.output_include, indent=2, verbose=False)
    dump_json(exc, args.output_exclude, indent=2, verbose=False)
    logger.info(f"Wrote {args.output_include} and {args.output_exclude}")


def contains(att_list: list[dict[str, str]], att: dict[str, str]) -> bool:
    for attr in att_list:
        # skip same attribute, even if query is different
        for field in ["attribute", "search_query"]:
            if att[field].lower() == attr[field].lower():
                return True
    return False


def num_atts(data: dict[str, dict[str, list[dict[str, str]]]]) -> int:
    return sum(len(att_list) for attributes in data.values() for att_list in attributes.values())


def delete_empty(data: dict[str, dict[str, list[dict[str, str]]]]):
    for entity in list(data.keys()):
        attributes = data[entity]
        for cat in list(attributes.keys()):
            att_list = attributes[cat]
            if len(att_list) == 0:
                del data[entity][cat]
                logger.debug(f"DELETE EMPTY entity-cat {entity} {cat}")
        if len(data[entity]) == 0:
            del data[entity]
            logger.debug(f"DELETE EMPTY entity {entity}")


def is_none(att: dict[str, str]) -> bool:
    if att["attribute"] == "Note":  # qwen generated some notes
        return True
    for check_str in [att["attribute"].lower(), att["search_query"].lower()]:
        if check_str == "not applicable":
            return True
        if check_str == "none":
            return True
        if check_str == "note":
            return True
        if check_str.startswith("none ") and len(check_str) > 16:
            return True
        if len(check_str) > 100:
            return True
    return False


if __name__ == "__main__":
    main()
