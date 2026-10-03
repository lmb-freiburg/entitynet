import argparse
import os
import random
import sys
from multiprocessing import Pool
from re import escape

import requests
from llm_text_generation import TextGenerator
from tqdm import tqdm

QLEVER_URL = "https://qlever.cs.uni-freiburg.de/api/wikidata"


def query_qlever(query: str) -> dict:
    response = requests.post(
        QLEVER_URL, headers={"Content-type": "application/sparql-query"}, data=query
    )
    return response.json()


def get_infos(entity: str) -> tuple[str, str | None, list[str]]:
    query = f"""
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT
?label
(SAMPLE(?desc) AS ?d)
# (GROUP_CONCAT(?alias; SEPARATOR=";") AS ?a)
WHERE {{
   ?x @en@rdfs:label ?label .
   OPTIONAL {{ ?x @en@schema:description ?desc . }}
   OPTIONAL {{ ?x @en@skos:altLabel ?alias . }}
   VALUES ?x {{ {entity} }}
}}
GROUP BY ?label
"""
    # uncomment aliases selection in the query above to use aliases
    result = query_qlever(query)
    bindings = result["results"]["bindings"]
    if len(bindings) == 0:
        raise RuntimeError(f"No bindings found for entitiy {entity}")
    binding = bindings[0]
    label = binding["label"]["value"]
    desc = binding.get("d", {}).get("value", None)
    aliases = binding.get("a", {}).get("value", None)
    if aliases is None:
        aliases = []
    else:
        aliases = aliases.split(";")
    return label, desc, aliases


# Animals
# dog
# https://qlever.cs.uni-freiburg.de/wikidata/uizEyZ
# wolf
# https://qlever.cs.uni-freiburg.de/wikidata/LGebK4

# Plants
# tree
# https://qlever.cs.uni-freiburg.de/wikidata/2jEWqA
# alium cepa (onion)
# https://qlever.cs.uni-freiburg.de/wikidata/g6HUu8


def get_type_list(entity: str, entity_typ: str) -> list[tuple[str, str, str | None, list[str]]]:
    typ_entity = "wd:Q729" if entity_typ == "animal" else "wd:Q756"
    query = f"""
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX schema: <http://schema.org/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT
?typ
(SAMPLE(?label) AS ?l)
(SAMPLE(?desc) AS ?d)
(MAX(?score) AS ?s)
# (GROUP_CONCAT(?alias; SEPARATOR=";") AS ?a)
WHERE {{
   {entity} (wdt:P31/wdt:P279*)|wdt:P279+ ?typ .
   ?typ wdt:P279* {typ_entity} .
   ?typ @en@rdfs:label ?label .
   OPTIONAL {{ ?typ @en@schema:description ?desc . }}
   # OPTIONAL {{ ?typ @en@skos:altLabel ?alias . }}
   OPTIONAL {{ ?typ ^schema:about/wikibase:sitelinks ?score . }}
}}
GROUP BY ?typ
ORDER BY DESC(?s)
"""
    result = query_qlever(query)
    types = []
    for binding in result["results"]["bindings"]:
        typ = binding["typ"]["value"]
        if not typ.startswith("http://www.wikidata.org/entity/Q"):
            continue
        typ = typ.split("/")[-1]
        label = binding["l"]["value"]
        desc = binding.get("d", {}).get("value", None)

        # uncomment this and the corresponding parts in the query
        # to use aliases within the type list
        # aliases = binding.get("a", {}).get("value", None)
        # if aliases is None:
        #     aliases = []
        # else:
        #     aliases = aliases.split(";")

        types.append((typ, label, desc, []))

    # defaults
    if len(types) == 0:
        if entity_typ == "animal":
            types.append(("Q729", "animal", "kingdom of multicellular eukaryotic organisms", []))
        else:
            types.append(("Q756", "plant", "kingdom of photosynthetic eukaryotes", []))

    assert len(types) > 0, f"no types found for {entity}"
    return types


def _format(label: str, desc: str | None, entity: str | None, aliases: list[str]) -> str:
    if desc is not None:
        label += f" ({desc})"
    if len(aliases) > 0:
        label += ", also known as " + ", ".join(aliases)
    if entity is not None:
        label += f" {entity}"
    return label


PROMPT = """The natural type of an entity is the superclass \
of the entity that a human would most likely associate with it. \
The natural type is neither too general, nor too specific, and can be used to \
disambiguate the entity from other entities with the same name.


"""


def format_entity(
    entity: str,
    entity_typ: str,
    natural_types: list[str] | None = None,
) -> tuple[str, list[str]]:
    label, desc, aliases = get_infos(entity)
    types = get_type_list(entity, entity_typ)
    random.shuffle(types)
    formatted = _format(label, desc, None, aliases)
    selectable_types = []
    types_formatted = []
    for typ, label, desc, aliases in types:
        selectable_types.append(f"{label} {typ}")
        types_formatted.append(_format(label, desc, typ, aliases))
    types_formatted = "\n".join(types_formatted)
    s = f"""Entity:
{formatted}

List of types:
{types_formatted}

Natural type:
"""
    if natural_types is not None:
        types_formatted = []
        for entity in natural_types:
            label, desc, _ = get_infos(entity)
            types_formatted.append(f"{label} {entity.split('/')[-1][:-1]}")
        formatted = "\n".join(types_formatted)
        s += formatted + "\n"
    return s, selectable_types


def get_prompt_and_regex(
    entity: str,
    entity_typ: str,
    examples: list[str],
) -> tuple[str, str]:
    formatted, selectable = format_entity(entity, entity_typ)

    prompt = PROMPT
    if len(examples) > 0:
        prompt += "\n\n".join(examples) + "\n\n"
    prompt += formatted

    regex = "(" + "|".join(escape(s) for s in selectable) + ")"
    return prompt, regex


def load_samples(path: str) -> list[tuple[str, list[str]]]:
    samples = []
    with open(path) as inf:
        for line in inf:
            entity, types = line.rstrip("\r\n").split("\t")
            samples.append((entity, types.split()))
    return samples


def load_tsv(path: str) -> list[list[str]]:
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["plant", "animal"])
    parser.add_argument("model")
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-e", "--examples", type=str, default=None)
    parser.add_argument("-ne", "--num-examples", type=int, default=5)
    parser.add_argument("-n", "--num-processes", type=int, default=None)
    parser.add_argument("-l", "--label", action="store_true")
    parser.add_argument("-b", "--batch-size", type=int, default=8)
    return parser.parse_args()


def get_inputs(inputs):
    return get_prompt_and_regex(*inputs)


def run(args: argparse.Namespace):
    if args.num_processes is None:
        args.num_processes = len(os.sched_getaffinity(0))

    examples = []
    if args.examples is not None:
        for entity, types in tqdm(load_samples(args.examples), "preparing examples", leave=False):
            example, _ = format_entity(entity, args.type, types)
            examples.append(example)

    args.num_examples = min(args.num_examples, len(examples))

    if args.input is not None:
        entities = load_tsv(args.input)
    else:
        entities = [line.rstrip("\n\r").split("\t") for line in sys.stdin]

    # # generate inputs, single thread version for easier debugging
    # inputs = []
    # for entity, *_ in tqdm(entities, desc="Generating inputs"):
    #     examples_here = random.sample(examples, args.num_examples)
    #     prompt, regex = get_prompt_and_regex(entity, args.type, examples_here)
    #     inputs.append((prompt, regex))

    # generate inputs, multi thread version
    with Pool(args.num_processes) as pool:
        inputs = list(
            pool.imap(
                get_inputs,
                tqdm(
                    (
                        (entity, args.type, random.sample(examples, args.num_examples))
                        for entity, *_ in entities
                    ),
                    desc="generating inputs dynamically",
                    total=len(entities),
                ),
            )
        )

    batch_size = args.batch_size

    def _generate_batches():
        for i in range(0, len(entities), batch_size):
            batch_entities = entities[i : i + batch_size]
            batch_inputs = inputs[i : i + batch_size]
            yield batch_entities, batch_inputs

    print(f"Setup model {args.model}", file=sys.stderr)
    llm = TextGenerator.from_experiment(args.model, device="auto")
    print(f"Generating types for {len(entities)} entities", file=sys.stderr)

    pbar = tqdm(
        total=len(entities),
        desc=f"Generate types {args.type} {args.model.split('/')[-1]}",
        ncols=90,
    )
    for batch_entities, batch_inputs in _generate_batches():
        for entity, input, output in zip(
            batch_entities,
            batch_inputs,
            llm.generate(batch_inputs, batch_size),
        ):
            result = []
            for typ in output.split("\n"):
                if typ == "":
                    continue
                splits = typ.strip().split(" ")
                result.append((" ".join(splits[:-1]), splits[-1]))

            s = f"{entity[0]}\t"
            if len(result) == 0:
                if args.label:
                    s += "\t"
                print(s, flush=True)
                continue

            assert len(result) == 1
            label, qid = result[0]
            s += f"<http://www.wikidata.org/entity/{qid}>"
            if args.label:
                s += f"\t{label}"
            print(s, flush=True)
            pbar.update()
    pbar.close()


if __name__ == "__main__":
    run(parse_args())
