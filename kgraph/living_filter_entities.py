import argparse
import math
import sys

from tqdm import tqdm

from llm_text_generation import TextGenerator

PLANT_PROMPT = """\
Given the name and description of an entity, one can decide whether the \
entity is a plant/fruit or not. Species, breeds, and similar entities are \
considered plants. Specific named plants like famous trees are not considered \
plants. Also products like oils or other entities only loosely related to \
plants are not considered plants.


Entity:
banana (elongated, edible fruit produced by several kinds of large herbaceous \
flowering plants in the genus Musa), also known as "bananas", "banana fruit"

Plant/fruit:
true


Entity:
cottonseed oil (cooking oil extracted from the seeds of cotton plants of \
various species, mainly Gossypium hirsutum and Gossypium herbaceum)

Plant/fruit:
false


Entity:
Quercus ilex (Oak tree species native to the Mediterranean), also known as \
"Holm Oak"

Plant/fruit:
true


Entity:
bouquet garni (herb mixture used in cooking)

Plant/fruit:
false


Entity:
Moon Tree (tree grown from one of the 500 seeds taken into orbit around the \
Moon by Stuart Roosa during the Apollo 14 mission in 1971)

Plant/fruit:
false
"""

ANIMAL_PROMPT = """\
Given the name and description of an entity, one can decide whether the \
entity is an animal or not. Species, breeds, and similar entities are \
considered animals. Specific named animals like famous horses are not \
considered animals. Also products like leather or other entities only \
loosely related to animals are not considered animals.


Entity:
house cat (domesticated feline), also known as "cat", "domestic cat", \
"housecat", "puss", "cats"

Animal:
true


Entity:
black tiger (color of tiger)

Animal:
false


Entity:
alligator leather (leather from alligator)

Animal:
false


Entity:
Pomeranian Show Crest Highflyer (pigeon breed)

Animal:
true


Entity:
Tuna (House cat and Internet celebrity from #BlackCatsofTikTok), also known \
as "Katherina Chapa"

Animal:
false
"""


def load_tsv(path: str) -> list[list[str]]:
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["plant", "animal"])
    parser.add_argument("model")
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-b", "--batch_size", type=int, default=8)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.input is not None:
        entities = load_tsv(args.input)
    else:
        entities = [line.rstrip("\n\r").split("\t") for line in sys.stdin]

    print(f"Setup model {args.model}", file=sys.stderr)
    llm = TextGenerator.from_experiment(args.model, device="auto")
    llm.set_inference_options(constraint="true|false")
    print(f"Filtering {len(entities)} entities", file=sys.stderr)

    batch_size = args.batch_size

    def _generate_batches():
        for i in range(0, len(entities), batch_size):
            batch_entities = entities[i : i + batch_size]
            batch_prompts_regexes = []
            for entity in batch_entities:
                label = entity[1][1:-4]
                desc = entity[2][1:-4]
                aliases = {
                    alias
                    for alias in entity[4][1:-1].split(";")
                    + entity[5][1:-1].split(";")
                    + entity[6][1:-1].split(";")
                    if alias != "" and alias != label
                }

                if args.type == "plant":
                    prompt = PLANT_PROMPT
                else:
                    prompt = ANIMAL_PROMPT

                prompt += f"\n\nEntity:\n{label} ({desc})"
                if aliases:
                    prompt += ", also known as " + ", ".join(f'"{alias}"' for alias in aliases)

                typ_prompt = "Plant/fruit" if args.type == "plant" else "Animal"
                prompt += f"\n\n{typ_prompt}:\n"

                batch_prompts_regexes.append((prompt, None))

            yield batch_entities, batch_prompts_regexes

    pbar = tqdm(
        total=len(entities), desc=f"Filter {args.type} {args.model.split('/')[-1]}", ncols=90
    )
    for batch_entities, batch_prompts_regexes in _generate_batches():
        for entity, prompt, output in zip(
            batch_entities,
            batch_prompts_regexes,
            llm.generate(batch_prompts_regexes, batch_size),
        ):
            if output == "false":
                continue
            print("\t".join(entity), flush=True)
            pbar.update(1)
    pbar.close()


if __name__ == "__main__":
    main(parse_args())
