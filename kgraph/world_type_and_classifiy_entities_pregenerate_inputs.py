import argparse
import json
import os
import random
import sys
from multiprocessing import Pool
from pathlib import Path
from re import escape

import requests
from tqdm import tqdm

from packg.iotools import dump_json, load_json

from entitynet.datasets.wikidata.qleverutils import query_qlever
from entitynet.datasets.wikidata.wikidata_hierarchies_loader import simplify_wikidata_id
from entitynet.datasets.wikidata.wikidata_utils import strip_label


def get_super_types(entity: str) -> list[str]:
    query = f"""
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT DISTINCT ?typ WHERE {{
    {entity} wdt:P279|wdt:P279/wdt:P279|wdt:P279/wdt:P279/wdt:P279 ?typ .
    MINUS {{ VALUES ?typ {{ wd:Q223557 wd:Q488383 wd:Q35120 wd:Q4406616 }} }}
    ?typ rdfs:label ?label .
    FILTER(LANG(?label) = "en")
}}
"""
    return [res[0] for res in query_qlever(query)]


def clip(s: str, n: int = 64) -> str:
    return s[:n] + "..." * (len(s) > n)


def get_infos(entities: list[str], top_k: int = 10) -> list[tuple[str, str, str, list[str]]]:
    query = f"""
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT
?ent
(SAMPLE(?label) AS ?l)
(SAMPLE(?desc) AS ?d)
(GROUP_CONCAT(DISTINCT ?alt; SEPARATOR=";;;") AS ?a)
(MAX(?score) AS ?s)
WHERE {{
   VALUES ?ent {{ {" ".join(entities)} }}
   ?ent rdfs:label ?label .
   FILTER(LANG(?label) = "en")
   ?ent ^schema:about/wikibase:sitelinks ?score .
   OPTIONAL {{ ?ent schema:description ?desc . FILTER(LANG(?desc) = "en") }}
   OPTIONAL {{ ?ent skos:altLabel ?alt . FILTER(LANG(?alt) = "en") }}
}}
GROUP BY ?ent
ORDER BY DESC(?s)
LIMIT {top_k}
"""
    output = []
    for ent, label, desc, aliases, score in query_qlever(query):
        ent = "wd:" + ent.split("/")[-1][:-1]  # strip > at the end
        label = strip_label(label)
        desc = strip_label(desc)
        desc = clip(desc)
        aliases = [clip(alias) for alias in aliases[1:-1].split(";;;") if alias]
        if len(aliases) > 1:
            aliases = random.sample(aliases, 1)
        output.append((ent, label, desc, aliases))
    return output


def format_entity(entity: str, label: str, desc: str, aliases: list[str]) -> str:
    if aliases:
        label += ", also known as " + ", ".join(aliases)
    label += f" ({entity})"
    if desc:
        label += f": {desc}"
    return label


PROMPT = """\
We are looking for the natural type and visuality class of the given entity.
The natural type of an entity is the superclass \
of the entity that a human would most likely associate with it. \
The natural type is neither too general, nor too specific, and can be used to \
disambiguate the entity from other entities with the same name.
The visuality class of an entity can be either 'visual' or 'non-visual'. \
For example, abstract concepts like 'justice' or non-physical entities \
are typically non-visual. Concrete physical entities like 'cat' are visual.
Format your answer as follows:
{
    "natural_type": {
        "explanation": "(short explanation of max. 20 words)",
        "name": "(natural type)",
        "id": "(natural type id)"
    },
    "visuality_class": {
        "explanation": "(short explanation of max. 20 words)",
        "class": "(visual|non-visual)"
    }
}

"""


def format_sample(entity: str, top_k: int = 10) -> tuple[str, list[tuple[str, str]]]:
    entity_info = get_infos([entity])[0]
    super_types = get_super_types(entity)
    if len(super_types) == 0:
        super_types.append(entity_info)

    type_infos = get_infos(super_types, top_k)

    selectable_types = []
    types_formatted = []
    for typ, label, desc, aliases in type_infos:
        selectable_types.append((label, typ))
        types_formatted.append(format_entity(typ, label, desc, aliases))

    types_formatted = "\n".join(types_formatted)
    s = f"""Entity information:
{format_entity(*entity_info)}

List of possible natural types:
{types_formatted}

"""
    return s, selectable_types


def get_prompt_and_regex(entity: str, top_k: int = 10) -> tuple[str, str]:
    formatted, selectable = format_sample(entity, top_k=top_k)
    prompt = PROMPT + formatted
    regex = f"""\
\{{
    "natural_type": \{{
        "explanation": "([a-zA-Z0-9]+ ?){{1,20}}",
        "name": "({"|".join(escape(s) for s, _ in selectable)})",
        "id": "({"|".join(escape(s) for _, s in selectable)})"
    \}},
    "visuality_class": \{{
        "explanation": "([a-zA-Z0-9]+ ?){{1,20}}",
        "class": "(visual|non-visual)"
    \}}
\}}"""
    return prompt, regex


def load_tsv(path: str) -> list[list[str]]:
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # parser.add_argument("model")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input file, if not provided, read from stdin",
    )
    parser.add_argument(
        "-n",
        "--num-processes",
        type=int,
        default=None,
        help="Number of processes to use for preparing inputs",
    )
    parser.add_argument("-k", "--top-k", type=int, default=10, help="Select from the top k types")
    parser.add_argument(
        "-s",
        "--skip",
        type=int,
        default=0,
        help="Skip the first n entities in the input",
    )
    parser.add_argument(
        "-t",
        "--take",
        type=int,
        default=None,
        help="Take only the first n entities in the input (after skipping)",
    )
    parser.add_argument("--clear_cache", action="store_true")
    return parser.parse_args()


def get_inputs_cached(inputs):
    entity, top_k, entity_file = inputs
    if entity_file.is_file():
        inp = load_json(entity_file)
    else:
        inp = get_prompt_and_regex(entity, top_k)
        dump_json(inp, entity_file, verbose=False)
    return inp


def run(args: argparse.Namespace):
    if args.num_processes is None:
        args.num_processes = len(os.sched_getaffinity(0))

    if args.input is not None:
        entities = load_tsv(args.input)
    else:
        entities = [line.rstrip("\n\r").split("\t") for line in sys.stdin]

    entities = entities[args.skip : args.skip + (args.take or len(entities))]
    print(f"{len(entities):,} remaining")

    entity_cache_dir = Path(__file__).parent / "entity_cache"
    os.makedirs(entity_cache_dir, exist_ok=True)

    inputs_for_imap = []
    for entity, *_ in tqdm(entities, desc="Preparing inputs for imap"):
        entity_simple = simplify_wikidata_id(entity)
        entity_file = entity_cache_dir / f"{entity_simple}.json"
        if args.clear_cache and entity_file.is_file():
            entity_file.unlink()
        inputs_for_imap.append((entity, args.top_k, entity_file))
    with Pool(args.num_processes) as pool:
        inputs = list(
            pool.imap(
                get_inputs_cached,
                tqdm(
                    (inp for inp in inputs_for_imap),
                    desc="Generating inputs",
                    total=len(entities),
                ),
            )
        )


if __name__ == "__main__":
    run(parse_args())
