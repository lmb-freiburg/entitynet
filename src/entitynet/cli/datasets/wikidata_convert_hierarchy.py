"""
Columns:
0: entity id
1: name@language
2: description@language
3: number of sitelinks
4: "also known as" / aliases
5: common names
6: taxon names
7: image(s) link

entity_id   : <http://www.wikidata.org/entity/Q144>
label       : dog@en
desc        : domestic animal@en
links       : 306
aliases     : domestic dog;🐕;dogs;🐶;Canis lupus familiaris;Canis familiaris
common_names: nan
taxon_names : Canis lupus familiaris
images      : http://commons.wikimedia.org/wiki/Special:FilePath/Greenland%20467%20%2835130903436%29%20%28cropped%29.jpg


python -m entitynet.cli.datasets.wikidata_print_hierarchy_tree \
kgraph/wikidata_living/animals.hierarchy.tsv kgraph/wikidata_living/animals.hierarchy.json \
Q38280

German Shepherd
+-- dog
    |-- domesticated mammal
    |   +-- domesticated animal
    |       |-- animal
    |       +-- non-human animal
    |           |-- animal
    |           +-- non-human
    |               +-- animal
    +-- pet
        +-- domesticated animal
            |-- animal
            +-- non-human animal
                |-- animal
                +-- non-human
                    +-- animal

<http://www.wikidata.org/entity/Q38280>_0_1 German Shepherd
+-- <http://www.wikidata.org/entity/Q144>_1_1 dog
    |-- <http://www.wikidata.org/entity/Q39201>_2_1 pet
    |   +-- <http://www.wikidata.org/entity/Q622852>_3_1 domesticated animal
    |       |-- <http://www.wikidata.org/entity/Q24249370>_4_1 non-human animal
    |       |   |-- <http://www.wikidata.org/entity/Q13023682>_5_1 non-human
    |       |   |   +-- <http://www.wikidata.org/entity/Q729>_6_1 animal
    |       |   +-- <http://www.wikidata.org/entity/Q729>_5_1 animal
    |       +-- <http://www.wikidata.org/entity/Q729>_4_1 animal
    +-- <http://www.wikidata.org/entity/Q57814795>_2_1 domesticated mammal
        +-- <http://www.wikidata.org/entity/Q622852>_3_2 domesticated animal
            |-- <http://www.wikidata.org/entity/Q24249370>_4_2 non-human animal
            |   |-- <http://www.wikidata.org/entity/Q13023682>_5_2 non-human
            |   |   +-- <http://www.wikidata.org/entity/Q729>_6_2 animal
            |   +-- <http://www.wikidata.org/entity/Q729>_5_2 animal
            +-- <http://www.wikidata.org/entity/Q729>_4_2 animal

Leftover todos from 2024:
    entity_large was downloaded later and therefore sometimes has more information
        so we could try and merge that also to the medium and small entity set
        however probably too much effort for a few description
"""

from collections import defaultdict

from loguru import logger
from natsort import natsorted

from entitynet.datasets.wikidata.wikidata_hierarchy import (
    get_large_entity_set,
    get_medium_entity_set,
    get_small_entity_set,
    simplify_id,
    entity_types,
    resolve_parents,
    display_entity_and_parents,
    KGRAPH_DIR,
)
from packg.iotools import dump_json, load_json


def main():
    # load entities. this creates the jsons if not exists: entities_{size}-{version}.json
    # these use simplified keys like "Q144" without all the url stuff
    entities_small = get_small_entity_set()
    entities_medium = get_medium_entity_set()
    entities_large = get_large_entity_set()
    for entity_id, item in entities_medium.items():
        item["source"] = "medium"
        entities_large[entity_id] = item
    for entity_id, item in entities_small.items():
        item["source"] = "small"
        entities_large[entity_id] = item
    logger.info(f"Total {len(entities_large)=}")
    logger.info(f"After simplifying ids: {len(set(simplify_id(a) for a in entities_large.keys()))}")
    # logger.info(f"After simplifying ids: {len(entities_large)}")

    # load the full hierarchies for all entity_types, with simplified ids everywhere
    missing_entities = {}
    hierarchy_links = {}
    for entity_type in entity_types:
        hierarchy_links_file = KGRAPH_DIR / f"wikidata_living/{entity_type}.hierarchy.json"
        hierarchy_links_ent = load_json(hierarchy_links_file)
        for entity_id, parent_ids in hierarchy_links_ent.items():
            entity_id = simplify_id(entity_id)
            parent_ids = [simplify_id(p) for p in parent_ids]
            hierarchy_links[entity_id] = parent_ids
            for check_entity in [entity_id] + parent_ids:
                if check_entity not in entities_large:
                    missing_entities[check_entity] = None
    logger.info(f"Got {len(hierarchy_links)} links from 1 child to 1-N parents")
    if len(missing_entities) > 0:
        raise RuntimeError(f"Missing entities: {natsorted(missing_entities.keys())}")

    # for each entity in the small set, build the full hierarchy and save hierarchy.json
    full_hierarchy = {}
    entities_appearing = {}
    not_founds = defaultdict(int)
    for e_num, (entity_id, item) in enumerate(entities_small.items()):
        entities_appearing[entity_id] = None
        parent_dict = resolve_parents(entity_id, hierarchy_links)
        if e_num < 10:
            display_entity_and_parents(entity_id, entities_large, hierarchy_links, min_sitelinks=40)
        # fill the hierarchy
        full_hierarchy[entity_id] = []
        for parent_id, depth in parent_dict.items():
            parent_item = entities_large.get(parent_id)
            if parent_item is None:
                raise RuntimeError(f"Item not found: {parent_id}")
                # not_founds[parent_id] += 1
                # continue
            full_hierarchy[entity_id].append((parent_id, depth))
            entities_appearing[parent_id] = None
    hierarchy_out_file = KGRAPH_DIR / "hierarchy_wikidata_living/hierarchy.json"
    dump_json(full_hierarchy, hierarchy_out_file, verbose=True, indent=2)
    logger.info(f"Entities appearing in the hierarchy: {len(entities_appearing)}")
    logger.info(f"{len(not_founds)} entities were not found, {sum(not_founds.values())} times")

    # since there are significantly less than 3.3M entities it's worth to only safe those
    # currently it's only around 58k (2k more than the input)
    # create subset of relevant entities: entities_relevant.json
    entity_data_file = KGRAPH_DIR / "hierarchy_wikidata_living/entities_relevant.json"
    entity_data = {}
    for entity_id in entities_appearing.keys():
        item = entities_large[entity_id]
        entity_data[entity_id] = item
    dump_json(entity_data, entity_data_file, indent=2, create_parent=True)


if __name__ == "__main__":
    main()
