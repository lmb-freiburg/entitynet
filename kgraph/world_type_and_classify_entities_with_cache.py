import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

from crx.datasets.livingthings.wikidata_loader import simplify_wikidata_id
from llm_text_generation import TextGenerator
from packg.iotools import load_json


def load_tsv(path: str) -> list[list[str]]:
    with open(path, "r") as f:
        return [line.rstrip("\r\n").split("\t") for line in f]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=None,
        help="Input file, if not provided, read from stdin",
    )
    parser.add_argument("-o", "--output", type=str, default="types_and_classes.json")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="Batch size")
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


def run(args: argparse.Namespace):
    if args.input is not None:
        entities = load_tsv(args.input)
    else:
        entities = [line.rstrip("\n\r").split("\t") for line in sys.stdin]

    entities = entities[args.skip : args.skip + (args.take or len(entities))]

    if os.path.exists(args.output):
        with open(args.output, "r") as f:
            outputs = json.load(f)
    else:
        outputs = {}

    entities = [entity for entity in entities if entity[0] not in outputs]
    print(f"Already processed {len(outputs):,} entities, {len(entities):,} remaining")

    entity_cache_dir = Path(__file__).parent / "entity_cache"

    # # generate inputs, multi process version
    # with Pool(args.num_processes) as pool:
    #     inputs = list(
    #         pool.imap(
    #             get_inputs,
    #             tqdm(
    #                 ((entity, args.top_k) for entity, *_ in entities),
    #                 desc="Generating inputs",
    #                 total=len(entities),
    #             ),
    #         )
    #     )

    # cached single process version
    inputs = []
    for entity, *_ in tqdm(entities):
        entity_simple = simplify_wikidata_id(entity)
        entity_file = entity_cache_dir / f"{entity_simple}.json"
        if not entity_file.is_file():
            raise FileNotFoundError(
                f"Entity file {entity_file} does not exist, run pregenerate_inputs.py first"
            )
        inp = load_json(entity_file)
        inputs.append(inp)

    llm = TextGenerator.from_experiment(args.model, device="auto")
    # llm.set_inference_options(sampling_strategy="top_p", top_p=0.95)

    for entity, output in tqdm(
        zip(entities, llm.generate(iter(inputs), batch_size=args.batch_size, sort=False)),
        total=len(entities),
        desc="Generating types",
    ):
        try:
            output = json.loads(output)
        except json.JSONDecodeError as e:
            print(
                f"Failed to decode output for entity {entity}:\n{output}\n{e}",
            )
            continue

        if entity[0] in outputs:
            print(f"Entity {entity[0]} already processed")
            continue

        outputs[entity[0]] = output
        if len(outputs) % 100 == 0:
            # save intermediate results
            with open(args.output, "w") as f:
                json.dump(outputs, f, indent=2)

    with open(args.output, "w") as f:
        json.dump(outputs, f, indent=2)


if __name__ == "__main__":
    run(parse_args())
