import argparse
import json
import logging
import os
import re
import sys
from enum import Enum
from typing import Iterable, Iterator

from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from entitynet.datasets.wikidata.wikidata_utils import strip_label
from packg import format_exception
from packg.web.robust_request import send_robust_post_request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--llm-text-gen-model",
        type=str,
        default=None,
        help="Path to experiment directory",
    )
    model_group.add_argument(
        "--openai-model",
        type=str,
        default=None,
        help="OpenAI model",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input file, if not provided, read from stdin",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="Batch size",
    )
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
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite entries in output file even if they already exist",
    )
    return parser.parse_args()


# build pydantic models for attributes
class Category(str, Enum):
    color = "Color"
    pattern_and_texture = "Pattern and texture"
    parts = "Parts"
    shape_and_size = "Shape and size"
    environment = "Environment"
    other = "Other"


class Attribute(BaseModel):
    attribute: str
    search_query: str


class Output(BaseModel):
    attributes: dict[Category, list[Attribute]]


class OpenAIOutput(BaseModel):
    color_attributes: list[Attribute]
    pattern_and_texture_attributes: list[Attribute]
    parts_attributes: list[Attribute]
    shape_and_size_attributes: list[Attribute]
    environment_attributes: list[Attribute]
    other_attributes: list[Attribute]

    def to_output(self) -> Output:
        return Output(
            attributes={
                Category.color: self.color_attributes,
                Category.pattern_and_texture: self.pattern_and_texture_attributes,
                Category.parts: self.parts_attributes,
                Category.shape_and_size: self.shape_and_size_attributes,
                Category.environment: self.environment_attributes,
                Category.other: self.other_attributes,
            }
        )


def get_attribute_regex() -> str:
    att_query_pairs = r"[a-zA-Z0-9\x20]{1,64}: [a-zA-Z0-9\x20]{1,128}"
    regex = ""
    for i, category in enumerate(Category):
        regex += f"{i+1}. {re.escape(category.value)}\n"
        regex += f"(?:{att_query_pairs}\n){{0,10}}\n"
    return regex


def parse_output(s: str) -> Output:
    lines = s.splitlines()
    attributes = {}
    att_list = []
    cat = None
    for line in lines:
        if not line.strip():
            if cat:
                attributes[cat] = att_list
                att_list = []
            cat = None
            continue

        if line[0].isdigit() and cat is None:
            cat_val = line.split(maxsplit=1)[1]
            for cat in Category:
                if cat.value == cat_val:
                    break
        else:
            att, query = line.split(": ")
            att_list.append(Attribute(attribute=att, search_query=query))

    assert all(cat in attributes for cat in Category), "Missing categories"

    return Output(attributes=attributes)


ATTRIBUTE_PROMPT = """\
For a given object, we want to generate visual attributes \
for different attribute categories. A visual attribute is a characteristic or \
additional context that can be observed in images of the object.
We are given the object's name and description, and also aliases and \
attribute hints if available. Importantly, the hints can but do not have to be \
used for generating attributes. They may be incomplete or \
unsuitable for the task, and should primarily serve as additional information.
The attribute categories are predefined and should be \
used to structure the generated attributes. For each attribute category, \
provide at most ten attributes depending on how many diverse and fitting \
attributes there are for the object in that category.
For each attribute, also provide a search query that can be used to find images \
of the object where the attribute is visible in a search engine. Examples \
are provided below before the actual target object.


Attribute categories:
1. Color
2. Pattern and texture
3. Parts
4. Shape and size
5. Environment
6. Other


Example 1:
Object:
apple, also known as "apple fruit": fruit of the apple tree

Attribute hints:
{}

Attributes:
1. Color
red: red apple
green: green apple
yellow: yellow apple

2. Pattern and texture
smooth: smooth apple
wrinkled: wrinkled apple
hole: hole in an apple

3. Parts
skin: apple skin
flesh: apple flesh
seeds: apple seeds

4. Shape and size
big: big apple
small: small apple
round: round apple

5. Environment
tree: apple on a tree
orchard: apple orchard
ground: apple on the ground

6. Other
ripe: ripe apple
rotten: rotten apple
peeled: peeled apple
worm: worm in an apple


Example 2:
Object:
elephant: large terrestrial mammals with trunks from Africa and Asia

Attribute hints:
{}

Attributes:
1. Color
gray: gray elephant
brown: brown elephant

2. Pattern and texture
wrinkled: elephant with wrinkled skin

3. Parts
tusks: elephant tusks
trunk: elephant trunk
ears: elephant ears
tail: elephant tail
legs: elephant legs

4. Shape and size
small: small elephant

5. Environment
desert: elephant in the desert
herd: herd of elephants

6. Other
walking: walking elephant
eating: elephant eating something
swimming: swimming elephant
baby: baby elephant


Example 3:
Object:
laptop, also known as "laptop computer", "notebook", "notebook computer", "notebook PC", \
"laptop PC", "notebook personal computer": personal computer that is foldable and portable

Attribute hints:
{}

Attributes:
1. Color
black: black laptop
silver: silver laptop
white: white laptop

2. Pattern and texture
shiny: shiny laptop
carbon fiber: laptop with carbon fiber texture
sticker: laptop with stickers

3. Parts
keyboard: laptop keyboard
screen: laptop screen
trackpad: laptop trackpad
hinge: laptop hinge

4. Shape and size
thin: thin laptop
mini: mini laptop

5. Environment
desk: laptop on a desk
lap: laptop on a lap
stand: laptop on a stand

6. Other
open: open laptop
closed: closed laptop
charging: laptop charging
gaming: gaming laptop
"""

QLEVER_URL = "https://qlever.cs.uni-freiburg.de/api/wikidata-sebastian"
DIRECT_ATTRIBUTES = [
    "P462",  # color
    "P1672",  # taxon source of
    "P1552",  # has characteristic
    "P1535",  # used by
    "P4733",  # produced sound
    "P1034",  # main food source
    "P3095",  # practiced by
    "P361",  # part of
    "P527",  # has part
    "P1542",  # has effect
    "P1963",  # properties for this type
    "P136",  # genre
    "P793",  # significant event
    "P1056",  # product, material, or service produced or provided
    "P186",  # made from material
    "P2079",  # fabrication method
    "P137",  # operator
]
QUALIFIER_ATTRIBUTES = [
    ("P2283", "P5102"),  # uses (+ nature of statement)
    ("P366", "P805"),  # has use (+ subject)
    ("P5869", "P11527"),  # model item (+ applies to use with)
    # ("P5135", "P1013"),  # greater than (+ criterion used)
    # ("P5136", "P1013"),  # less than (+ criterion used)
]


def get_attribute_hints(entity: str) -> str:
    direct_att_list = " ".join("wdt:" + a for a in DIRECT_ATTRIBUTES)

    qual_att_list = "\n".join(
        f"""UNION {{
        {entity} p:{p} ?s .
        ?s ps:{p} ?v .
        wd:{p} rdfs:label ?attLabel .
        FILTER(LANG(?attLabel) = "en")
        ?v rdfs:label ?vLabel .
        FILTER(LANG(?vLabel) = "en")
        OPTIONAL {{
            ?s pq:{q} ?q .
            ?q rdfs:label ?qLabel .
            FILTER(LANG(?qLabel) = "en")
        }}
        BIND(
            IF(
                BOUND(?qLabel),
                CONCAT(?vLabel, " (", ?qLabel, ")"),
                ?vLabel
            ) as ?valLabel
        )
    }}
"""
        for p, q in QUALIFIER_ATTRIBUTES
    )

    query = f"""\
PREFIX p: <http://www.wikidata.org/prop/>
PREFIX ps: <http://www.wikidata.org/prop/statement/>
PREFIX pq: <http://www.wikidata.org/prop/qualifier/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
SELECT
    ?attLabel
    (GROUP_CONCAT(DISTINCT ?valLabel; SEPARATOR=";;;") AS ?vals)
WHERE {{
    {{
        VALUES ?att {{ {direct_att_list} }}
        {entity} ?att ?val .
        ?val rdfs:label ?valLabel .
        ?attEnt wikibase:directClaim ?att .
        ?attEnt rdfs:label ?attLabel .
        FILTER(LANG(?attLabel) = "en")
        FILTER(LANG(?valLabel) = "en")
    }}
    {qual_att_list}
}}
GROUP BY ?attLabel"""

    response = send_robust_post_request(
        QLEVER_URL,
        headers={
            "Content-type": "application/sparql-query",
            "Accept": "text/tab-separated-values",
        },
        data=query,
        n_trials=10,
        n_retry_sleep=30,
    )

    result_iter = response.iter_lines()
    hints = []
    # skip header
    next(result_iter)
    for line in result_iter:
        att, vals = line.decode().split("\t")
        att = strip_label(att)
        vals = [v for v in vals[1:-1].split(";;;") if v]
        if not vals:
            continue

        hints.append(f"{att}: {', '.join(vals)}")

    return "\n".join(hints) if hints else "None"


_PROMPT: str | None = None


def get_base_prompt() -> str:
    global _PROMPT
    if _PROMPT is not None:
        return _PROMPT

    prompt = ATTRIBUTE_PROMPT

    apple_hints = get_attribute_hints("wd:Q89")
    elephant_hints = get_attribute_hints("wd:Q7378")
    laptop_hints = get_attribute_hints("wd:Q3962")

    _PROMPT = prompt.format(
        apple_hints,
        elephant_hints,
        laptop_hints,
    )
    return _PROMPT


def get_prompt(entity: str, label: str, desc: str, aliases: list[str]) -> str:
    prompt = get_base_prompt()

    hints = get_attribute_hints(entity)

    object_string = label
    if aliases:
        object_string += ", also known as "
        object_string += ", ".join(f'"{alias}"' for alias in aliases)

    if desc:
        object_string += f": {desc}"

    prompt += f"""

Object:
{object_string}

Attribute hints:
{hints}"""

    return prompt


Entity = tuple[str, str, str, list[str]]


class AttributeGenerator:
    def generate(self, inputs: Iterable[Entity]) -> Iterator[Output]:
        raise NotImplementedError


class OpenAIAttributeGenerator(AttributeGenerator):
    def __init__(self, model: str, logger: logging.Logger):
        from openai import OpenAI

        self.logger = logger
        self.client = OpenAI()
        self.model = model

    def generate(self, inputs: Iterable[Entity]) -> Iterator[Output]:
        for input in inputs:
            prompt = get_prompt(*input)
            self.logger.debug(f"Entity:\n{input}")
            self.logger.debug(f"Prompt:\n{prompt}")
            n_trials, response = 5, None
            for n_trial in range(n_trials):
                try:
                    response = self.client.beta.chat.completions.parse(
                        messages=[{"role": "user", "content": prompt}],
                        model=self.model,
                        top_p=0.9,
                        response_format=OpenAIOutput,
                    )
                except ValidationError as e:
                    err_msg = (
                        f"OpenAI generated invalid JSON in trial {n_trial}/{n_trials}: "
                        f"{format_exception(e)}"
                    )
                    if n_trial >= n_trials - 1:
                        raise e
                    self.logger.error(err_msg)

            output = response.choices[0].message.parsed
            self.logger.debug(f"Output:\n{json.dumps(output.dict(), indent=2)}")
            usage = response.usage
            self.logger.debug(
                f"{usage.prompt_tokens:,} input tokens "
                f"({usage.prompt_tokens_details.cached_tokens:,} cached)"
            )
            self.logger.debug(f"{usage.completion_tokens:,} completion tokens")
            yield output.to_output()


class OwnAttributeGenerator(AttributeGenerator):
    def __init__(self, model: str, logger: logging.Logger, batch_size: int = 1):
        from llm_text_generation import TextGenerator

        self.logger = logger
        self.model = TextGenerator.from_experiment(model)
        self.model.set_inference_options(
            sample=True,
            min_p=0.2,
            repeat_penalty=1.05,
            temperature=0.7,
            constraint=get_attribute_regex(),
        )
        self.batch_size = batch_size

    def generate(self, inputs: Iterable[Entity]) -> Iterator[Output]:
        entities = list(inputs)

        def generate_batches() -> tuple[list[Entity], list[str], list]:
            for i in range(0, len(entities), self.batch_size):
                batch_entities = entities[i : i + self.batch_size]
                prompts = []
                inputs = []
                for entity in batch_entities:
                    prompt = get_prompt(*entity)
                    prompts.append(prompt)
                    inputs.append(
                        (
                            [
                                {"role": "user", "text": prompt},
                                {
                                    "role": "assistant",
                                    "text": "Attributes:\n",
                                    "partial": True,
                                },
                            ],
                            None,
                        )
                    )

                yield batch_entities, prompts, inputs

        for batch_entities, batch_prompts, batch in generate_batches():
            for entity, prompt, output in zip(
                batch_entities,
                batch_prompts,
                self.model.generate(batch, self.batch_size),
            ):
                self.logger.debug(f"Entity:\n{entity}")
                self.logger.debug(f"Prompt:\n{prompt}")
                self.logger.debug(f"Output:\n{output}")
                parsed = parse_output(output)
                self.logger.debug(f"Parsed:\n{json.dumps(parsed.dict(), indent=2)}")
                yield parsed


def load_tsv(path: str) -> list[list[str]]:
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def main(args: argparse.Namespace) -> None:
    logger = logging.getLogger("ATTRBITUE_GENERATION")
    logger.setLevel(args.log_level)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)

    if args.input:
        entities = load_tsv(args.input)
    else:
        entities = [line.rstrip("\n\r").split("\t") for line in sys.stdin]

    entities = entities[args.skip : args.skip + (args.take or len(entities))]

    if os.path.exists(args.output) and not args.overwrite:
        with open(args.output, "r") as f:
            outputs = json.load(f)
    else:
        outputs = {}

    entities = [entity for entity in entities if entity[0] not in outputs]
    logger.info(f"Already processed {len(outputs):,} entities, {len(entities):,} remaining")

    def format_entities() -> Iterator[Entity]:
        # prepare and yield formatted entities
        for entity, label, desc, _, aliases, *_ in entities:
            if len(label) > 0:
                label = strip_label(label)
            if len(desc) > 0:
                desc = strip_label(desc)
            if len(aliases) > 0:
                aliases = strip_label(aliases)
            aliases = {alias for alias in aliases.split(";;;") if alias and alias != label}
            yield entity, label, desc, aliases

    entities = list(format_entities())

    if args.llm_text_gen_model:
        generator = OwnAttributeGenerator(
            args.llm_text_gen_model,
            logger,
            args.batch_size,
        )
    elif args.openai_model:
        generator = OpenAIAttributeGenerator(args.openai_model, logger)
    else:
        raise ValueError("No model provided")

    for entity, output in tqdm(
        zip(entities, generator.generate(entities)),
        desc="Generating attributes",
        total=len(entities),
    ):
        outputs[entity[0]] = output.dict()["attributes"]
        # write after every entity
        with open(args.output, "w") as f:
            json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    main(parse_args())
