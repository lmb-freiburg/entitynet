import math

import sys
import argparse
from tqdm import tqdm

from llm_text_generation import TextGenerator


PLANT_PROMPT = """\
From the following list of visual plant attributes, one can predict attribute \
instances of each attribute for a particular plant. An attribute instance is \
a specific manifestation of an attribute, such as green, red, or yellow for \
the attribute color. The attribute instances are predicted based on the \
plant's name, description, and aliases if available. It is important that \
the attribute instances are visual and can be observed in images of the \
plant. For each attribute there should be at least one and at most ten \
attribute instances, depending on how diverse and how fitting the attribute \
instances are for the plant. After each attribute instance, also provide \
a search query that can be used to find images of the attribute instance \
in a search engine. Make sure that the name of the plant or one of its \
aliases is mentioned in each search query.

1. Color (e.g. green, red, yellow, etc.)
2. Pattern and texture (e.g. stripes, spots, smooth, rough, etc.)
3. Plant parts (e.g. leaves, bark, stem, roots, seeds, etc.)
4. Shape and size (e.g. big, small, tall, round, flat, etc.)
5. Habitat and environment (e.g. tree, bush, forest, garden, etc.)
6. Other (e.g. age, flowering, peeled, etc.)


Plant:
apple (fruit of the apple tree), also known as "apple fruit"

Attribute instances:
1. Color
red: red apple
green: green apple
yellow: yellow apple

2. Pattern and texture
smooth: smooth apple
wrinkled: wrinkled apple
hole: hole in an apple

3. Plant parts
skin: apple skin
flesh: apple flesh
seeds: apple seeds

4. Shape and size
big: big apple
small: small apple
round: round apple

5. Habitat and environment
tree: apple on a tree
orchard: apple orchard
ground: apple on the ground

6. Other
ripe: ripe apple
rotten: rotten apple
peeled: peeled apple
worm: worm in an apple


Plant:
Castanea (genus of plants), also known as "chestnut", "chestnut genus", \
"chestnuts"

Attribute instances:
1. Color
brown: brown chestnut
green: green chestnut
yellow: yellow chestnut

2. Pattern and texture
spiny: spiny chestnut shell
smooth: smooth chestnut

3. Plant parts
leaves: chestnut leaves
shell: chestnut shell
bark: chestnut tree bark
flower: chestnut flower

4. Shape and size
round: round chestnut
large: large chestnut
elongated: elongated chestnut leaf

5. Habitat and environment
hillside: chestnut tree on hillside
garden: chestnut tree in garden
fall: chestnut tree in fall
tree: chestnut on a tree

6. Other
flowering: flowering chestnut tree
fallen: fallen chestnut
roasted: roasted chestnut


Plant:
Pinaceae (family of plants), also known as "spruces", "firs", \
"cedars", "pines", "pine family", "larches"

Attribute instances:
1. Color
green: green pine tree
blue-green: blue-green spruce

2. Pattern and texture
scaly: scaly bark of a cedar
smooth: smooth fir bark
spiky: spiky pine needles

3. Plant parts
needles: pine tree needles
cone: fir tree cone
trunk: cedar tree trunk

4. Shape and size
tall: tall spruce tree
columnar: columnar fir tree
voluptuous: voluptuous cedar tree

5. Habitat and environment
mountain: pine tree on a mountain
forest: spruce forest
alpine: alpine larch tree

6. Other
plastic: artificial cedar tree
pot: potted pine tree
sap: sap from a fir tree
"""

PLANT_REGEX = """\
ATTRIBUTE ([a-zA-Z\\x20]{1,64}:\\x20[a-zA-Z\\x20]{1,128}\\n){1,10}
%%
1. Color
{ATTRIBUTE}
2. Pattern and texture
{ATTRIBUTE}
3. Plant parts
{ATTRIBUTE}
4. Shape and size
{ATTRIBUTE}
5. Habitat and environment
{ATTRIBUTE}
6. Other
{ATTRIBUTE}
"""


ANIMAL_PROMPT = """\
From the following list of visual animal attributes, one can predict \
attribute instances of each attribute for a particular animal. An attribute \
instance is a specific manifestation of an attribute, such as brown, black, \
or white for the attribute color. The attribute instances are predicted based \
on the animal's name, description, and aliases if available. It is important \
that the attribute instances are visual and can be observed in images of the \
animal. For each attribute there should be at least one and at most ten \
attribute instances, depending on how diverse and how fitting the attribute \
instances are for the animal. After each attribute instance, also provide a \
search query that can be used to find images of the attribute instance in a \
search engine. Make sure that the name of the animal or one of its aliases \
is mentioned in each search query.


1. Color (e.g. brown, black, white, etc.)
2. Pattern and texture (e.g. stripes, spots, long fur, short fur, etc.)
3. Body parts (e.g. wings, legs, tail, tusks, etc.)
4. Behavior and movement (e.g. flying, swimming, running, sleeping, etc.)
5. Shape and size (e.g. big, small, short, round, flat, etc.)
6. Habitat and environment (e.g. tree, desert, herd, pack, nest, etc.)
7. Other (e.g. age, sex, etc.)


Animal:
elephant (large terrestrial mammals with trunks from Africa and Asia)

Attribute instances:
1. Color
gray: gray elephant
brown: brown elephant

2. Pattern and texture
wrinkled: elephant with wrinkled skin

3. Body parts
tusks: elephant tusks
trunk: elephant trunk
ears: elephant ears
tail: elephant tail
legs: elephant legs

4. Behavior and movement
walking: walking elephant
eating: elephant eating something
sleeping: sleeping elephant
swimming: swimming elephant

5. Shape and size
small: small elephant

6. Habitat and environment
desert: elephant in the desert
herd: herd of elephants

7. Other
baby: baby elephant
old: old elephant
young: young elephant


Animal:
Thunnus (genus of fishes), also known as "tuna", "tunafish", \
"tuna fish", "tunas", "tuna fishes", "tunafishes"

Attribute instances:
1. Color
blue: blue tuna
silver: silver tuna

2. Pattern and texture
striped: striped tuna
shiny: shiny skin of a tuna

3. Body parts
fins: tuna fins
tail: tuna tail
gills: tuna gills
eyes: tuna eyes

4. Behavior and movement
swimming: swimming tuna
jumping: jumping tuna

5. Shape and size
big: big tuna
small: small tuna

6. Habitat and environment
ocean: tuna in the ocean
school: tuna school

7. Other
fresh: fresh tuna
canned: canned tuna
juvenile: juvenile tuna


Animal:
Eurasian Eagle-owl (species of bird), also known as "Bubo bubo", \
"Northern Eagle Owl", "Eurasian Eagle-Owl", "eurasian eagle owl", "Eurasian \
Eagle Owl"

Attribute instances:
1. Color
brown: brown eagle-owl
gray: gray eagle-owl

2. Pattern and texture
mottled: mottled eagle-owl

3. Body parts
beak: eagle-owl beak
eyes: eagle-owl eyes
ear tufts: eagle-owl ear tufts
talons: eagle-owl talons
wing: eagle-owl wing

4. Behavior and movement
flying: flying eagle-owl
hunting: hunting eagle-owl
perching: perching eagle-owl

5. Shape and size
large: large eagle-owl
small: small eagle-owl

6. Habitat and environment
woodland: eagle-owl in woodland
nest: eagle-owl nest

7. Other
young: young eagle-owl
adult: adult eagle-owl
"""

ANIMAL_REGEX = """\
ATTRIBUTE ([a-zA-Z\\x20]{1,64}:\\x20[a-zA-Z\\x20]{1,128}\\n){1,10}
%%
1. Color
{ATTRIBUTE}
2. Pattern and texture
{ATTRIBUTE}
3. Body parts
{ATTRIBUTE}
4. Behavior and movement
{ATTRIBUTE}
5. Shape and size
{ATTRIBUTE}
6. Habitat and environment
{ATTRIBUTE}
7. Other
{ATTRIBUTE}
"""


def load_tsv(path: str) -> list[list[str]]:
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["plant", "animal"])
    parser.add_argument("model", type=str)
    parser.add_argument("-i", "--input", type=str, default=None)
    parser.add_argument("-v", "--version", choices=["v1", "v2"], default="v2")
    parser.add_argument("-b", "--batch_size", type=int, default=1)
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    if args.input is not None:
        entities = load_tsv(args.input)
    else:
        entities = [line.rstrip("\n\r").split("\t") for line in sys.stdin]

    if args.type == "plant":
        regex = PLANT_REGEX
    elif args.type == "animal":
        regex = ANIMAL_REGEX
    else:
        raise ValueError(f"Unknown type {args.type}")

    llm = TextGenerator.from_experiment(args.model, device="auto")
    llm.set_inference_options(sample=True, top_p=0.9, constraint=regex)

    n_total = math.ceil(len(entities) / args.batch_size)
    pbar = tqdm(
        total=n_total, desc=f"Generate attributes {args.type} {args.model.split('/')[-1]}", ncols=90
    )
    for i in range(0, len(entities), args.batch_size):
        batch_entities = entities[i : i + args.batch_size]

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
                prompt = f"{PLANT_PROMPT}\n\nPlant:\n{label} ({desc})"
            else:
                prompt = f"{ANIMAL_PROMPT}\n\nAnimal:\n{label} ({desc})"

            if aliases:
                prompt += ", also known as " + ", ".join(f'"{alias}"' for alias in aliases)

            prompt += "\n\nAttribute instances:\n"
            batch_prompts_regexes.append((prompt, None))

        outputs = llm.generate(batch_prompts_regexes, batch_size=len(batch_prompts_regexes))

        for entity, output in zip(batch_entities, outputs):
            instances = []
            attribute = ""
            for opt in output.split("\n"):
                opt = opt.strip()
                if opt == "":
                    continue
                if opt[0].isdigit():
                    attribute = opt.split(maxsplit=1)[1]
                else:
                    instance, query = opt.split(": ")
                    instances.append(attribute + "\t" + instance + "\t" + query)

            line = entity[0] + "\t" + "\t".join(instances)
            print(line, flush=True)
        pbar.update()
    pbar.close()


if __name__ == "__main__":
    main(parse_args())
