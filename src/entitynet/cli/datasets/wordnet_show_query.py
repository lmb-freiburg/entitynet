"""
Show wordnet query
"""

from pprint import pprint

import pandas as pd
from attrs import asdict, define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from typedparser import TypedParser, VerboseQuietArgs

from entitynet.datasets.wordnet import (
    WordnetNoun,
    display_synset_from_synname,
    load_wordnet_nouns_as_namedtuples,
)
from entitynet.datasets.wordnet.nouns import display_noun_hierarchy
from entitynet.datasets.wordnet.wordnet_queries import (
    load_wordnet_noun_queries,
    load_wordnet_noun_query2synnames_list,
)


@define
class Args(VerboseQuietArgs):
    pass


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    # load wordnet
    wordnet_nouns = load_wordnet_nouns_as_namedtuples()

    # load queries
    queries = load_wordnet_noun_queries()
    logger.info(f"{len(queries)} queries")
    query2synnames = load_wordnet_noun_query2synnames_list()

    # example query to check
    query = queries[64]
    logger.info(f"Checking query: {query}")

    # get all synsets that are sources of this query. only ~4% queries have more than one source.
    synsets = query2synnames[query]
    logger.info(f"Synsets: {synsets}")

    for si, synname in enumerate(synsets):
        # display synset data full
        logger.info(f"Synset {si}: {synname}")
        if synname not in wordnet_nouns:
            raise ValueError(f"Synset name {synname} not in wordnet")
        noun: WordnetNoun = wordnet_nouns[synname]
        pprint(asdict(noun))

        # display compact
        print(display_synset_from_synname(synname))

        # display tree
        print(display_noun_hierarchy(synname))

    # statistics
    nsyns = [len(query2synnames[q]) for q in queries]
    pds = pd.Series(nsyns)
    print(pds.value_counts() / len(pds))
    print(pds.describe())


if __name__ == "__main__":
    main()
