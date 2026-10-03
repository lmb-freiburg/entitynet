"""
Convert the entity tsvs into a simpler dictionary

Columns:
0: entity id
1: name@language
2: description@language
3: number of sitelinks
4: "also known as" / aliases
5: common names
6: taxon names
7: image(s) link
"""

import math
from pprint import pprint

import pandas as pd
from loguru import logger

from packg.iotools import dump_json
from visiontext.nlp import preprocess_text_simple
from visiontext.pandatools import pd_print_series

from entitynet.datasets.wikidata.wikidata_hierarchy import KGRAPH_DIR
from entitynet.datasets.wikidata.wikidata_utils import strip_label_no_quotes


def check_empty(field):
    if field is None or field == "" or (isinstance(field, float) and math.isnan(field)):
        return True
    return False


def main():
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
    out_dict = {}
    for tsv, n_rows in [
        ("animals.tsv", 28000),
        ("plants_and_fruits.tsv", 28000),
    ]:
        tsv_file = KGRAPH_DIR / "wikidata_living" / tsv
        df = pd.read_csv(
            tsv_file,
            sep="\t",
            header=None,
            names=columns,
        )
        print(df.head())
        pd_print_series(df.iloc[0])
        for index, series in df.iterrows():
            entity_id, label, desc, links, aliases, common_names, taxon_names, images = series

            assert not check_empty(label)
            label = strip_label_no_quotes(label, assert_has_lang=True)
            if check_empty(desc):
                desc = None
            else:
                desc = strip_label_no_quotes(desc, assert_has_lang=True)
            if check_empty(links):
                links = 0
            links = int(links)

            output_names_and_source = {label: "label"}
            for names_str, source in [
                (aliases, "aliases"),
                (common_names, "common_names"),
                (taxon_names, "taxon_names"),
            ]:
                if check_empty(names_str):
                    continue
                names = names_str.split(";")
                for name in names:
                    if name in output_names_and_source:
                        continue
                    stripped_name = preprocess_text_simple(name)
                    if stripped_name == "":
                        # some names are only utf-8 smileys, skip them
                        continue
                    output_names_and_source[name] = source
            if index == 0:
                pprint(output_names_and_source)
            item = {
                "names": output_names_and_source,
                "desc": desc,
                # "sitelinks": links,
            }

            if entity_id in out_dict:
                logger.warning(f"Skipping duplicate\nOld: {out_dict[entity_id]}\nNew: {item}")
            else:
                out_dict[entity_id] = item

    logger.info(f"Total entities: {len(out_dict)=}")
    dump_json(
        out_dict,
        KGRAPH_DIR / "wikidata_living" / "entities-simple.json",
        create_parent=True,
        verbose=True,
        indent=2,
    )


if __name__ == "__main__":
    main()
