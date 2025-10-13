"""
For EntityNet WebDS dataset.

Ability to filter out some of the queries during training, given some settings.
"""

from collections import defaultdict

from pyparsing import Any

from entitynet.datasets.entitynet_load_metadata import load_entitynet_metadata_cached
from entitynet.datasets.entitynet_textloader import QUERY_TYPES
from packg.iotools import dump_json, load_json
from packg.log import logger

from entitynet.paths import get_entitynet_annotations_dir, get_entitynet_data_dir


class EntityNetQueryFilter:
    """Ability to skip queries"""

    def __init__(
        self,
        filter_op: str,
        filter_dict: dict[str, dict[str, Any]],
        queries: dict[str, dict[str, Any]],
    ):
        # create a unique filename for this setting
        skip_query_types = filter_dict.get("skip_query_types", [])
        skip_api_types = filter_dict.get("skip_apis", [])
        skip_alt_text = filter_dict.get("skip_alt_text", [])
        skip_list = filter_dict.get("skip_list", [])
        assert all(a in QUERY_TYPES for a in skip_query_types), f"{skip_query_types=}"
        assert all(a in ["bing", "google"] for a in skip_api_types), f"{skip_api_types=}"
        assert all(a in ["alt", "noalt"] for a in skip_alt_text), f"{skip_alt_text=}"
        self.filter_dict_hashed = "-".join(
            [
                f"op-{filter_op}",
                (
                    f'skiptypes-{",".join(sorted(skip_query_types))}'
                    if len(skip_query_types) > 0
                    else ""
                ),
                f'skipapis-{",".join(sorted(skip_api_types))}' if len(skip_api_types) > 0 else "",
                f'skipalt-{",".join(sorted(skip_alt_text))}' if len(skip_alt_text) > 0 else "",
                f'skiplist-{",".join(sorted(skip_list))}' if len(skip_list) > 0 else "",
            ]
        ).rstrip("-")

        # compute per query whether it is allowed or not
        if len(skip_query_types) == 0 and len(skip_api_types) == 0:
            query2allowflag = {query_hash: True for query_hash in queries.keys()}
        else:
            skip_query_types_set = set(skip_query_types)
            skip_api_types_set = set(skip_api_types)
            query2allowflag = {}
            for query_hash, query_data in queries.items():
                allowflag = True
                if query_data["query_type"] in skip_query_types_set:
                    allowflag = False
                elif query_data["api"] in skip_api_types_set:
                    allowflag = False
                query2allowflag[query_hash] = allowflag
        for list_name in skip_list:
            list_file = get_entitynet_annotations_dir() / f"entitynet/query_list-{list_name}.json"
            list_data = load_json(list_file)
            logger.warning(f"Skipping {len(list_data)} queries from {list_file}")
            for hsh in list_data:
                query2allowflag[hsh] = False

        self.filter_dict = filter_dict
        self.query2allowflag = query2allowflag
        self.key2allowflag = None
        self.skip_alt_text = skip_alt_text
        self.filter_op = filter_op

    def get_filter_dict_result_file(self, split: str):
        return (
            get_entitynet_data_dir()
            / f"entitynet/cache/precomputed_filters_per_tar"
            / f"filter-{split}-{self.filter_dict_hashed}.json"
        )

    def load_precomputed_filtering_results(self, split: str):
        file = self.get_filter_dict_result_file(split)
        if file.is_file():
            return load_json(file)
        metadata = load_entitynet_metadata_cached(split)
        self.precompute_filtering_results(split, metadata)
        return load_json(file)

    def precompute_filtering_results(
        self,
        split: str,
        metadata: dict[str, Any],
    ):
        filter_dict = self.filter_dict
        query2allowflag = self.query2allowflag
        file = self.get_filter_dict_result_file(split)
        n_queries_allowed = sum(1 for v in query2allowflag.values() if v)
        logger.info(
            f"Process filter dict {filter_dict} for {split=}, OP {self.filter_op}, "
            f"{n_queries_allowed} allowed of {len(query2allowflag)}"
        )

        # now decide for each datapoint whether to include it
        # to allow max shards it would be nice to know this per shard
        n_keys = 0
        n_keys_per_shard = defaultdict(int)
        for key, value in metadata.items():
            allowed = self.filter(key, value)
            if allowed:
                if n_keys == 0:
                    logger.info(f"Example allowing {key}")
                shard = value["shard"]
                n_keys_per_shard[shard] += 1
                n_keys += 1
        n_keys_per_shard = dict(n_keys_per_shard)
        assert sum(n_keys_per_shard.values()) == n_keys, f"{n_keys=} {n_keys_per_shard=}"
        settings_result = {
            "n_keys_allowed": n_keys,
            "n_keys_total": len(metadata),
            "setting": filter_dict,
            "n_queries_allowed": n_queries_allowed,
            "n_queries_total": len(query2allowflag),
            "n_keys_per_shard": n_keys_per_shard,
        }
        dump_json(settings_result, file, create_parent=True, verbose=True, indent=2)

    def filter(self, filename, fmetadata):  # noqa
        query2allowflag = self.query2allowflag
        # api, query, resultnum = filename.split("/")  # if needed
        num_alt_texts = len(fmetadata["texts"])
        if not self.is_allowed_given_num_alt_texts(num_alt_texts):
            return False
        qhs = fmetadata["qh"]
        n_allowed = sum([query2allowflag[qh] for qh in qhs])

        allowed = True
        match self.filter_op:
            case "any":  # any one of the queries is blocked -> the entire datapoint is blocked
                if n_allowed < len(qhs):
                    allowed = False
            case "all":  # only if all the queries are blocked, block the datapoint.
                if n_allowed == 0:
                    allowed = False
            case _:
                raise ValueError(f"Unknown {self.filter_op=}")
        return allowed

    def is_allowed_given_num_alt_texts(self, num_alt_texts):
        if len(self.skip_alt_text) == 0:
            return True
        assert len(self.skip_alt_text) == 1, f"{self.skip_alt_text=}"
        sat = self.skip_alt_text[0]
        if sat == "alt":
            if num_alt_texts > 0:
                return False
        elif sat == "noalt":
            if num_alt_texts == 0:
                return False
        else:
            raise ValueError(f"Unknown {self.skip_alt_text=}")
        return True
