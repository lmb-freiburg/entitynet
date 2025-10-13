"""
text loader for entitynet, to get all positive texts for a given image.
"""

import os
from pprint import pprint
from typing import Any

import numpy as np
import pandas as pd
from attr import asdict

from entitynet.paths import get_entitynet_data_dir
from packg.log import logger
from packg.strings import hash_object
from packg.typext import PathType

from clip_benchmark.datasets.en_zeroshot_classification_templates import EN_ZSCLS_TEMPLATES
from entitynet.config.task_config import EntityNetTextAugCfg, EntityNetTextReturnMode

TEXTTYPE2FIELDS = {
    "synonyms": ["aliases", "name", "common_names", "taxon_names"],
    "description": ["description", "type_description"],
}
FIELD2TEXTTYPE = {f: t for t, fs in TEXTTYPE2FIELDS.items() for f in fs}
QUERY_TYPES = [
    "wd-attr-living",
    "wd-attr-world",
    "wd-attrnoun-living",
    "wd-attrnoun-world",
    "wd-noun-living",
    "wd-noun-world",
    "wd-noun-world-cartoon",
    "wordnet-noun-living",
]
_R = "<<<REPLACE>>>"


class EntityNetUrlTextLoader:
    """
    Text loader that will be used both by indexable and webdataset version

    """

    def __init__(
        self,
        base_dir: PathType | None = None,
        text_aug: EntityNetTextAugCfg | None = None,
        seed: int | None = None,
        query2allowflag: dict[str, bool] | None = None,
    ):
        base_dir = get_entitynet_data_dir() / f"entitynet"
        if text_aug is None:
            text_aug = EntityNetTextAugCfg()

        # load queries
        query_df = pd.read_parquet(base_dir / "entitynet-queries.parquet")
        queries = query_df.set_index("query_hash", drop=False).to_dict(orient="index")

        # load entities
        entity_df = pd.read_parquet(base_dir / "entitynet-entities.parquet")
        entities: dict[str, dict[str, Any]] = {
            row["entity"]: row for row in entity_df.to_dict(orient="records")
        }

        # load entities text. text sources are as follows:
        # {'aliases': 216109, 'name': 168981, 'description': 164212, 'type_description': 95459,
        #  'common_names': 36394, 'taxon_names': 380}
        # group these text sources as either synonyms or descriptions
        entity_synonyms, entity_descriptions = {}, {}
        for entity, entity_data in entities.items():
            entity_textvalues = entity_data["textvalues"]
            entity_textsources = entity_data["textsources"]
            texttype2texts = {k: [] for k in TEXTTYPE2FIELDS}
            for textvalue, textfield in zip(entity_textvalues, entity_textsources):
                texttype = FIELD2TEXTTYPE[textfield]
                texttype2texts[texttype].append(textvalue)
            entity_synonyms[entity] = texttype2texts["synonyms"]
            assert entity_data["entity_name"] in entity_synonyms[entity]  # note name is a synonym
            entity_descriptions[entity] = texttype2texts["description"]

        rng: np.random.Generator = np.random.default_rng()
        if seed is not None:
            rng: np.random.Generator = np.random.default_rng(seed)

        config_sorted = dict(sorted(asdict(text_aug).items(), key=lambda x: x[0]))
        config_hash = hash_object(config_sorted)
        cache_file = base_dir / f"cache/text-w-src-{config_hash}.parquet"
        if not cache_file.is_file():
            logger.info(f"Precomputing text for queries to {cache_file}")
            text_df = compute_all_chances(
                queries, entities, entity_synonyms, entity_descriptions, text_aug
            )
            os.makedirs(cache_file.parent, exist_ok=True)
            text_df.to_parquet(cache_file)
        logger.info(f"Loading cached text for queries from {cache_file}")
        text_df = pd.read_parquet(cache_file)
        query_texts = text_df.set_index("query_hash", drop=False).to_dict(orient="index")

        self.text_aug = text_aug
        self.query2allowflag = query2allowflag
        self.entities = entities
        self.queries = queries
        self.entity_synonyms = entity_synonyms
        self.entity_descriptions = entity_descriptions
        self.rng = rng
        self.query_texts = query_texts
        self.deterministic_seed = seed

    def reset_rng(self):
        if self.deterministic_seed is not None:
            self.rng = np.random.default_rng(self.deterministic_seed)

    def get_data_for_dataloader(self, key: str, metadata: dict[str, Any]):
        data = {}
        if self.text_aug.return_mode == EntityNetTextReturnMode.ALL:
            data["text_list"], data["text_chances"], data["text_sources"] = (
                self.get_all_query_texts_and_alt_texts(key, metadata)
            )
        elif self.text_aug.return_mode == EntityNetTextReturnMode.SAMPLE:
            if self.text_aug.n_texts_per_image > 0:
                data["text_list"] = self.get_multiple_random_texts_for_image(key, metadata)
            else:
                data["text"] = self.get_random_text_for_image(key, metadata)
        elif self.text_aug.return_mode == EntityNetTextReturnMode.SAMPLE_RINCE:
            data["text_list"], data["rince_level"] = self.sample_rince(key, metadata)
        else:
            raise ValueError(
                f"Unknown {self.text_aug.return_mode=} - should be ALL, SAMPLE, or SAMPLE_RINCE"
            )
        return data

    def get_keys_for_webdataset(self) -> list[str]:
        # webdataset needs to know beforehand which keys to expect
        if self.text_aug.return_mode == EntityNetTextReturnMode.ALL:
            return ["text_list", "text_chances", "text_sources"]
        elif self.text_aug.return_mode == EntityNetTextReturnMode.SAMPLE:
            if self.text_aug.n_texts_per_image > 0:
                return ["text_list"]
            else:
                return ["text"]
        elif self.text_aug.return_mode == EntityNetTextReturnMode.SAMPLE_RINCE:
            return ["text_list", "rince_level"]
        else:
            raise ValueError(
                f"Unknown {self.text_aug.return_mode=} should be in "
                f"{EntityNetTextReturnMode.values_list}"
            )

    def get_random_text_for_image(self, image_key: str, metadata: dict[str, Any]) -> str:
        # maybe choose alt-text
        alt_text_list = metadata["texts"]
        text = None
        if len(alt_text_list) > 0:
            alt_text_chance = self.text_aug.alt_text_chance
        else:
            alt_text_chance = 0.0
        if self.roll_dice(alt_text_chance):
            if len(alt_text_list) == 1:
                text = alt_text_list[0]
            else:
                text = self.rng.choice(alt_text_list)

        # go over all merged duplicates and randomly choose one of the source queries
        # _file_api, _query_hash, _image_num = image_key.split("/")
        if "qh" not in metadata:
            pprint(metadata)
        queryhashes = metadata["qh"]
        if self.query2allowflag is not None:
            queryhashes = [qh for qh in queryhashes if self.query2allowflag[qh]]
        if len(queryhashes) == 0:
            breakpoint()
            raise ValueError(
                f"No queryhashes left for {image_key} {metadata=} if no queries are allowed "
                f"then this datapoint should have been skipped."
            )
        qh = self.rng.choice(queryhashes)

        if text is None:
            # in case we did not choose alt-text...
            text = self._get_single_random_text(qh)

        return str(text)

    def _get_single_random_text(self, qh):
        query_texts = self.query_texts
        query_texts_here = query_texts[qh]
        chances = query_texts_here["chances"]
        texts = query_texts_here["texts"]
        chances_norm = np.array(chances) / np.sum(chances)
        random_text = self.rng.choice(texts, p=chances_norm)
        # print(f"---------- {self.queries[qh]['query']} ----------")
        # for chance_norm, text in zip(chances_norm, texts):
        #     print(f"{chance_norm:7.2%} {text}")
        # print(f"Result: {random_text}")
        return random_text

    def roll_dice(self, chance: float):
        if chance <= 0:
            return False
        if chance >= 1.0:
            return True
        return self.rng.random() < chance

    def get_multiple_random_texts_for_image(self, image_key: str, metadata: dict[str, Any]):
        # go over all merged duplicates
        n_texts = self.text_aug.n_texts_per_image
        assert n_texts > 0, f"Called get_multiple_random_texts_for_image with {n_texts=}"
        all_texts, all_chances, all_sources = self.get_all_query_texts_and_alt_texts(
            image_key, metadata
        )
        return sample_texts_from_list(all_texts, all_chances, n_texts, self.rng)

    def get_all_query_texts_and_alt_texts(
        self, image_key: str, metadata: dict[str, Any]
    ) -> tuple[list[str], np.ndarray]:
        """
        Return a variable number of texts, their chances, and sources, for this image.

        Returns:
            all_texts: list[str] of length n_texts
            all_chances: np.ndarray of shape (n_texts,)
            all_sources: list[str] of length n_texts
        """
        queryhashes = metadata["qh"]
        if self.query2allowflag is not None:
            queryhashes = [qh for qh in queryhashes if self.query2allowflag[qh]]
        if len(queryhashes) == 0:
            raise ValueError(
                f"No queryhashes left for {image_key} {metadata=} if no queries are allowed "
                f"then this datapoint should have been skipped."
            )
        all_texts, all_chances = [], []
        all_sources = []  # TODO
        non_alt_text_chance = 1.0 - self.text_aug.alt_text_chance
        if non_alt_text_chance > 0.0:
            # get texts from queries
            for qh in queryhashes:
                texts, chances, text_sources = self.get_all_texts_for_query(qh)
                if len(texts) == 0:
                    continue
                chances_norm = chances / np.sum(chances)
                chances = chances_norm * non_alt_text_chance / len(queryhashes)
                all_texts.extend(texts)
                all_chances.extend(chances)
                all_sources.extend(text_sources)
        if self.text_aug.alt_text_chance > 0:
            # get alt texts
            alt_text_list = metadata["texts"]
            alt_text_chances = (
                np.ones(len(alt_text_list)) * self.text_aug.alt_text_chance / len(alt_text_list)
            )
            all_texts.extend(alt_text_list)
            all_chances.extend(alt_text_chances)
            all_sources.extend(["alt_text"] * len(alt_text_list))
        all_chances = np.array(all_chances)

        return all_texts, all_chances, all_sources

    def sample_rince(self, image_key: str, metadata: dict[str, Any]):
        all_texts, all_chances, all_sources = self.get_all_query_texts_and_alt_texts(
            image_key, metadata
        )
        alt_texts, alt_text_chances, alt_text_sources = [], [], []
        query_texts, query_text_chances, query_text_sources = [], [], []
        for text, chance, source in zip(all_texts, all_chances, all_sources):
            if source == "alt_text":
                alt_texts.append(text)
                alt_text_chances.append(chance)
            else:
                query_texts.append(text)
                query_text_chances.append(chance)
        # sample 1 each
        n_alt = 1
        n_query = 1
        texts, rince_levels = [], []
        alt_texts = sample_texts_from_list(
            alt_texts, alt_text_chances, n_alt, self.rng, assert_chances_normalized=False
        )
        texts.extend(alt_texts)
        rince_levels.extend([0] * len(alt_texts))
        query_texts = sample_texts_from_list(
            query_texts, query_text_chances, n_query, self.rng, assert_chances_normalized=False
        )
        texts.extend(query_texts)
        rince_levels.extend([1] * len(query_texts))
        return texts, rince_levels

    def get_all_texts_for_query(self, qh: str):
        query_texts = self.query_texts
        query_texts_here = query_texts[qh]
        chances = query_texts_here["chances"]
        texts = query_texts_here["texts"]
        text_sources = query_texts_here["text_sources"]
        return texts, chances, text_sources


def sample_texts_from_list(
    all_texts, all_chances, n_texts, rng=None, assert_chances_normalized: bool = True
):
    if rng is None:
        rng = np.random.RandomState()

    if len(all_texts) <= n_texts:
        # not enough texts, shuffle and repeat them
        rng.shuffle(all_texts)
        return_texts = [all_texts[i % len(all_texts)] for i in range(n_texts)]
        return return_texts

    # sample texts without replacement to get n_texts
    all_chances = np.array(all_chances)
    all_chances_sum = all_chances.sum()
    if assert_chances_normalized:
        assert (all_chances_sum - 1.0) < 1e-6, f"Chances do not sum to 1.0 {all_chances=}"
    all_chances = all_chances / all_chances_sum  # due to numerical instabilities, norm again anyway
    sampled_indices = rng.choice(len(all_texts), size=n_texts, replace=False, p=all_chances)
    return_texts = [all_texts[i] for i in sampled_indices]
    return return_texts


def compute_all_chances(queries, entities, entity_synonyms, entity_descriptions, text_aug):
    qhs = list(queries.keys())
    alls = {}
    for qh in qhs:
        chances, texts, text_sources = compute_chances_for_query(
            qh, queries, entities, entity_synonyms, entity_descriptions, text_aug
        )
        alls[qh] = (chances, texts, text_sources)

    # dataframe with columsn queryhash, list of chances, list of texts
    data_dict = {"query_hash": [], "chances": [], "texts": [], "text_sources": []}
    for qh, (chances, texts, text_sources) in alls.items():
        data_dict["query_hash"].append(qh)
        data_dict["chances"].append(chances)
        data_dict["texts"].append(texts)
        data_dict["text_sources"].append(text_sources)
    text_df = pd.DataFrame(data_dict)
    return text_df


NATURALENTITY_LIVING = set(["Q729", "Q756", "wordnet:animal.n.01", "wordnet:plant.n.02"])


def compute_chances_for_query(
    qh, queries, entities, entity_synonyms, entity_descriptions, text_aug
):
    # load query data
    query_data = queries[qh]
    query = query_data["query"]
    query_type = query_data["query_type"]
    entity = query_data["entity"]
    attribute = query_data["attribute"]

    # load entity data
    entity_data = entities[entity]
    entity_synonyms_here = list(entity_synonyms[entity])
    entity_descriptions_here = list(entity_descriptions[entity])

    # decide on parent chance
    hierarchy_chance = text_aug.replace_noun_hierarchy_chance
    if entity_data["naturalentity"] in NATURALENTITY_LIVING:
        hierarchy_chance = text_aug.replace_noun_hierarchy_chance_living
    non_hierarchy_chance = 1.0 - hierarchy_chance

    # load parent data
    naturalentity = entity_data["naturalentity"]
    parent_synonyms_here = list(entity_synonyms[naturalentity])
    parent_descriptions_here = list(entity_descriptions[naturalentity])

    if text_aug.combine_synonym_and_parentsynonym > 0.0:
        raise NotImplementedError("combine_synonym_and_parentsynonym not implemented")

    if query_type.startswith("wd-attr-"):
        # ---------- wikidata attribute queries like "durable plant"
        chances, text_lists, text_sources = get_chances_for_attribute_queries(
            query,
            attribute,
            entity_synonyms_here,
            parent_synonyms_here,
            entity_descriptions_here,
            parent_descriptions_here,
            non_hierarchy_chance,
            hierarchy_chance,
            text_aug.attronly_replace_query_with_synonym,
            text_aug.attronly_build_pseudo_query,
            text_aug.attronly_strgf_replace_entity,
            text_aug.attronly_attribute_only,
            text_aug.attronly_keep_query,
            text_aug.attronly_replace_with_definition,
            text_aug.clip_prompts,
        )
    elif query_type.startswith("wd-attrnoun-"):
        # ---------- wikidata attribute-noun
        chances, text_lists, text_sources = get_chances_for_attribute_queries(
            query,
            attribute,
            entity_synonyms_here,
            parent_synonyms_here,
            entity_descriptions_here,
            parent_descriptions_here,
            non_hierarchy_chance,
            hierarchy_chance,
            text_aug.attrnoun_replace_query_with_synonym,
            text_aug.attrnoun_build_pseudo_query,
            text_aug.attrnoun_strgf_replace_entity,
            text_aug.attrnoun_attribute_only,
            text_aug.attrnoun_keep_query,
            text_aug.attrnoun_replace_with_definition,
            text_aug.clip_prompts,
        )
    elif query_type.startswith("wordnet-noun-") or query_type.startswith("wd-noun-"):
        # ---------- wordnet / wikidata noun
        chances, text_lists, text_sources = [], [], []
        synonym_chance = text_aug.replace_noun_synonym_chance
        keep_noun_chance = 1.0 - synonym_chance
        if len(entity_synonyms) > 0:
            # in case synonyms become more likely than the original query, fix the chances
            chance_per_syn = synonym_chance / len(entity_synonyms)
            if chance_per_syn > keep_noun_chance:
                synonym_chance = 0.5
                keep_noun_chance = 0.5

        clip_prompt_chance = text_aug.clip_prompts
        non_clip_prompt_chance = 1.0 - clip_prompt_chance
        if non_clip_prompt_chance > 0.0:
            chances.append(keep_noun_chance * non_clip_prompt_chance)
            text_lists.append([query])
            text_sources.append("original_query")
            chances.append(synonym_chance * non_hierarchy_chance * non_clip_prompt_chance)
            text_lists.append(entity_synonyms_here)
            text_sources.append("entity_synonym")
            chances.append(synonym_chance * hierarchy_chance * non_clip_prompt_chance)
            text_lists.append(parent_synonyms_here)
            text_sources.append("parent_synonym")

        if clip_prompt_chance > 0.0:
            prompts = EN_ZSCLS_TEMPLATES["imagenet1k"]
            for text_list, chance, text_source in (
                ([query], keep_noun_chance, "original_query"),
                (entity_synonyms_here, synonym_chance * non_hierarchy_chance, "entity_synonym"),
                (parent_synonyms_here, synonym_chance * hierarchy_chance, "parent_synonym"),
            ):
                for i, prompt in enumerate(prompts):
                    weight = clip_prompt_chance / len(prompts)
                    chances.append(chance * weight)
                    text_lists.append([prompt.format(c=text) for text in text_list])
                    text_sources.append(f"{text_source}_clip_prompt_{i}")

        description_chance = text_aug.replace_noun_definition_chance
        chances.append(description_chance * non_hierarchy_chance)
        text_lists.append(entity_descriptions_here)
        text_sources.append("entity_definition")
        chances.append(description_chance * hierarchy_chance)
        text_lists.append(parent_descriptions_here)
        text_sources.append("parent_definition")
    else:
        raise ValueError(f"Unknown query type {query_type}")

    # flatten the list
    new_chances, flat_list, flat_text_sources = [], [], []
    for chance, text_list, text_source in zip(chances, text_lists, text_sources):
        if len(text_list) == 0 or chance == 0:
            continue
        chance_per = chance / len(text_list)
        for text in text_list:
            flat_list.append(text)
            new_chances.append(chance_per)
            flat_text_sources.append(text_source)

    # go through the flat list, sort by chance desc,  delete all duplicates
    existing = {}
    final_chances, final_flat_list, final_text_sources = [], [], []
    for chance, text, text_source in sorted(
        zip(new_chances, flat_list, flat_text_sources), key=lambda x: -x[0]
    ):
        # print(f"{chance:5.3f} {text}")
        if text in existing:
            continue
        existing[text] = True
        final_flat_list.append(text)
        final_chances.append(chance)
        final_text_sources.append(text_source)
    return final_chances, final_flat_list, final_text_sources


def get_chances_for_attribute_queries(
    query: str,
    attribute: str,
    entity_synonyms_here: list[str],
    parent_synonyms_here: list[str],
    entity_descriptions_here: list[str],
    parent_descriptions_here: list[str],
    non_hierarchy_chance: float,
    hierarchy_chance: float,
    replace_query_with_synonym_chance: float,
    build_pseudo_query_chance: float,
    strgf_replace_entity_chance: float,
    attribute_only_chance: float,
    keep_query_chance: float,
    replace_with_definition: float,
    clip_prompt_chance: float,
):
    """
    Returns:
        chances: list[float] of length n_text_lists
        text_lists: list[list[str]] of length n_text_lists
        text_sources: list[list[str]] of length n_text_lists
    """
    chances, text_lists, text_sources = [], [], []
    if replace_query_with_synonym_chance > 0.0:
        # ignore the attribute and replace with only the synonym
        chances.append(replace_query_with_synonym_chance * non_hierarchy_chance)
        text_lists.append(list(entity_synonyms_here))
        text_sources.append("entity_synonym")
        chances.append(replace_query_with_synonym_chance * hierarchy_chance)
        text_lists.append(list(parent_synonyms_here))
        text_sources.append("parent_synonym")

    if build_pseudo_query_chance > 0.0:
        # create new queries by simply adding the attribute to the left or right to synonym
        texts_here = []
        for syn in entity_synonyms_here:
            texts_here += [f"{attribute} {syn}", f"{syn} {attribute}"]
        chances.append(build_pseudo_query_chance * non_hierarchy_chance)
        text_lists.append(texts_here)
        text_sources.append("attribute_plus_entity_synonym")
        texts_here = []
        for syn in parent_synonyms_here:
            texts_here += [f"{attribute} {syn}", f"{syn} {attribute}"]
        chances.append(build_pseudo_query_chance * hierarchy_chance)
        text_lists.append(texts_here)
        text_sources.append("attribute_plus_parent_synonym")
    if strgf_replace_entity_chance > 0.0:
        # in the query, find the entity and replace it with the synonym
        query_rep = query
        for entity_synonym in entity_synonyms_here:
            query_rep = query.replace(entity_synonym, _R)
            if query_rep != query:
                break
        if query_rep != query:
            # successfully replaced the entity, build new queries with synonyms
            chances.append(strgf_replace_entity_chance * non_hierarchy_chance)
            text_lists.append([query_rep.replace(_R, syn) for syn in entity_synonyms_here])
            text_sources.append("replace_entity_with_entity_synonym_in_attribute_query")
            chances.append(strgf_replace_entity_chance * hierarchy_chance)
            text_lists.append([query_rep.replace(_R, syn) for syn in parent_synonyms_here])
            text_sources.append("replace_entity_with_parent_synonym_in_attribute_query")

    if attribute_only_chance > 0.0:
        # only add the attribute
        chances.append(attribute_only_chance)
        text_lists.append([attribute])
        text_sources.append("attribute_only")

    if keep_query_chance > 0.0:
        # keep the query
        chances.append(keep_query_chance)
        text_lists.append([query])
        text_sources.append("original_query")

    # apply clip prompts
    clip_prompts = EN_ZSCLS_TEMPLATES["imagenet1k"]
    non_clip_prompt_chance = 1.0 - clip_prompt_chance
    if clip_prompt_chance > 0.0:
        new_chances, new_text_lists, new_text_sources = [], [], []
        for chance, text_list, text_source in zip(chances, text_lists, text_sources):
            for i, prompt in enumerate(clip_prompts):
                new_chances.append(chance * clip_prompt_chance / len(clip_prompts))
                new_text_lists.append([prompt.format(c=text) for text in text_list])
                new_text_sources.append(f"{text_source}_clip_prompt_{i}")
        if non_clip_prompt_chance > 0.0:
            for chance, text_list, text_source in zip(chances, text_lists, text_sources):
                new_chances.append(chance * non_clip_prompt_chance)
                new_text_lists.append(text_list)
                new_text_sources.append(text_source)
        chances = new_chances
        text_lists = new_text_lists
        text_sources = new_text_sources

    if replace_with_definition > 0.0:
        # replace the query with the definition
        chances.append(replace_with_definition * non_hierarchy_chance)
        text_lists.append(entity_descriptions_here)
        text_sources.append("entity_definition")
        chances.append(replace_with_definition * hierarchy_chance)
        text_lists.append(parent_descriptions_here)
        text_sources.append("parent_definition")

    return chances, text_lists, text_sources


def apply_random_clip_prompt(text: str, rng: np.random.Generator | None = None):
    """apply a random clip prompt to the text."""
    if rng is None:
        rng = np.random.default_rng()
    prompt = rng.choice(EN_ZSCLS_TEMPLATES["imagenet1k"])
    return prompt.format(c=text)


def main():
    etl = EntityNetUrlTextLoader(
        text_aug=EntityNetTextAugCfg(
            n_texts_per_image=8,
            replace_noun_synonym_chance=0.6,
            replace_noun_definition_chance=0.1,
            replace_noun_hierarchy_chance=0.2,
            replace_noun_hierarchy_chance_living=0.1,
            alt_text_chance=0.5,
            attronly_keep_query=1.0,
            attronly_replace_query_with_synonym=0.2,
            attronly_build_pseudo_query=0.2,
            attronly_strgf_replace_entity=0.2,
            attronly_attribute_only=0.2,
            attronly_replace_with_definition=0.05,
            attrnoun_keep_query=1.0,
            attrnoun_replace_query_with_synonym=0.2,
            attrnoun_build_pseudo_query=0.2,
            attrnoun_strgf_replace_entity=0.2,
            attrnoun_attribute_only=0.2,
            attrnoun_replace_with_definition=0.0,
        )
    )
    for _ in range(10):
        pprint(
            etl.get_multiple_random_texts_for_image(
                "ePrOftEuZLDVKSjEtm8qH1QMU3BFISmXgEaFxA,58",
                {
                    "qh": ["ePrOftEuZLDVKSjEtm8qH1QMU3BFISmXgEaFxA"],
                    "texts": [f"alt text {i}" for i in range(10)],
                },
            )
        )


if __name__ == "__main__":
    main()
