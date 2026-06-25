"""
Load WordNet data and display some statistics about the different splits.

See WordNet license at src/entitynet/datasets/wordnet/LICENSE
"""

import random
from copy import deepcopy

from packg.iotools import load_json

from entitynet.datasets.wordnet import WordnetNoun, load_wordnet_nouns_as_namedtuples
from entitynet.paths import get_entitynet_annotations_dir


def main():
    # load wordnet, only needed for displaying the info for the nodes
    wordnet_nouns: dict[str, WordnetNoun] = load_wordnet_nouns_as_namedtuples(use_cache=True)
    wn_anno_dir = get_entitynet_annotations_dir() / "wordnet"
    random.seed(37)
    splits = "living-animalsplantsandparents", "living-noanimalsplants", "nonliving"
    synonym_overlap = None
    for split in splits:
        node2synonyms = load_json(wn_anno_dir / f"wordnet-{split}.json")
        synonyms = sorted(set(syn for syns in node2synonyms.values() for syn in syns))
        if synonym_overlap is None:
            synonym_overlap = deepcopy(synonyms)
        else:
            synonym_overlap = set(synonym_overlap) & set(synonyms)

        print(f"\n---------- {split}: {len(node2synonyms)=} {len(synonyms)=}\n")
        nodes = list(node2synonyms.keys())
        random.shuffle(nodes)
        for node in nodes[:25]:
            # synset = convert_synname_to_synset(node)
            wn_data: WordnetNoun = wordnet_nouns[node]
            print(f"{' | '.join(wn_data.lemmas)} - {wn_data.definition} ({wn_data.synname})")
    print()
    print(f"{len(synonym_overlap)=}")
    print()


if __name__ == "__main__":
    main()
