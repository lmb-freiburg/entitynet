"""
Indexable version of EntityNetUrl dataset
"""

import io
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info

from packg.log import logger
from typedparser.objects import print_datapoint
from visiontext.iotools.tar_lookup import TarLookup

from entitynet.config.task_config import EntityNetTextAugCfg, EntityNetTextReturnMode
from entitynet.datasets.entitynet import N_SHARDS
from entitynet.datasets.entitynet_load_metadata import load_entitynet_metadata_cached
from entitynet.datasets.entitynet_textloader import EntityNetUrlTextLoader
from entitynet.paths import get_entitynet_data_dir
from entitynet.preprocessor.preprocessor_factory import get_simple_transform_to_tensor


class EntityNetUrlIndexable(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        max_datapoints=None,
        text_aug: EntityNetTextAugCfg | None = None,
        eval_type: str = "default",
        deterministic_seed: int | None = None,
    ):
        base_dir = get_entitynet_data_dir() / f"entitynet"
        if text_aug is None:
            text_aug = EntityNetTextAugCfg()

        # build tar lookup to get the images
        worker = get_worker_info()
        worker = worker.id if worker else 0
        glob_str = f"data_{split}/*.tar"
        tar_files = sorted(base_dir.glob(glob_str))
        if len(tar_files) == 0:
            raise FileNotFoundError(f"Nothing found for dir {base_dir} glob {glob_str}")
        tar_files_rel = [f.relative_to(base_dir) for f in tar_files]
        index_file = base_dir / f"index_{split}.sqlite"
        tar_lookup = TarLookup(
            base_dir,
            tar_files_rel,
            index_file,
            verbose=False,
            worker_id=worker,
        )

        # # load and merge metadata on image level, this gets alt texts and llm-processed texts
        # metadata_files = []
        # for t in tar_files:
        #     tarnum = int(t.stem.split("_")[-1])
        #     metadata_file = base_dir / f"data_{split}/metadata_{tarnum:05d}.json"
        #     metadata_files.append(metadata_file)
        # metadata = {}
        # for metadata_files in metadata_files:
        #     json_data = load_json(metadata_files)
        #     for key, value in json_data.items():
        #         metadata[key] = value
        # if not len(metadata) == len(tar_lookup):
        #     logger.error(f"Image level metadata and tar content mismatch")
        #     breakpoint()

        metadata = load_entitynet_metadata_cached(split)
        text_loader = EntityNetUrlTextLoader(
            base_dir=base_dir, text_aug=text_aug, seed=deterministic_seed
        )
        if eval_type == "alttext":
            # alt-text evaluation set: only keep datapoints with alttext
            # and only keep 1 alttext to make it deterministic
            metadata_new = {}
            for image_key, metadata_item in metadata.items():
                alttexts = metadata_item["texts"]
                if len(alttexts) == 0:
                    continue
                alttexts = [alttexts[0]]  # only keep the first alt text
                metadata_item_new = deepcopy(metadata_item)
                metadata_item_new["texts"] = alttexts
                metadata_new[image_key] = metadata_item_new
            metadata = metadata_new
            # image_keys = list(meta_image.keys())
        elif eval_type == "default":
            pass
        else:
            raise ValueError(f"Unknown eval type {eval_type}")

        image_keys = list(metadata.keys())
        if max_datapoints is not None and max_datapoints > 0:
            image_keys = image_keys[:max_datapoints]
            logger.warning(f"Restricting dataset to {len(image_keys)} datapoints.")

        preloaded_texts = None
        if deterministic_seed is not None:
            assert (
                text_aug.return_mode == EntityNetTextReturnMode.SAMPLE
            ), "Only return_mode.SAMPLE is supported for eval with deterministic augmentation"
            # val or test set with random text augmentation: augment once with fixed rng
            preloaded_texts = {}
            for image_key in image_keys:
                metadata_here = metadata[image_key]
                text = text_loader.get_random_text_for_image(image_key, metadata_here)
                preloaded_texts[image_key] = text
            logger.debug(f"Preloaded {len(preloaded_texts)} texts for deterministic augmentation")

        self.text_loader = text_loader
        self.image_keys = image_keys
        self.metadata = metadata
        self.text_aug_cfg = text_aug
        self.transform = transform
        self.tar_lookup = tar_lookup
        self.preloaded_texts = preloaded_texts

    def __len__(self):
        return len(self.image_keys)

    def __getitem__(self, idx):
        if idx == 0:
            # in unshuffled val case we want all epochs to have the same rng
            self.text_loader.reset_rng()

        image_key = self.image_keys[idx]

        # load jpeg image
        filename, tarfilename, image_bytes = self.tar_lookup.get_content_from_filename(
            f"{image_key}.jpg"
        )
        assert filename.endswith(".jpg"), f"Filename {filename} does not end with .jpg."
        key = filename[:-4]
        assert image_key == key, f"Key {image_key} does not match filename {filename}."
        try:
            # image_arr = decode_jpeg(image_bytes, method=JPEGDecoderConst.PILLOW)  # TODO libturbo
            image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Error decoding image {filename} from tar {tarfilename}") from e
        if self.transform is not None:
            image_pil = self.transform(image_pil)
        ret = {
            "image": image_pil,
            "idx": idx,
            "label": -1,  # TODO not implemented yet
        }

        # load text
        metadata_here = self.metadata[image_key]
        data = self.text_loader.get_data_for_dataloader(image_key, metadata_here)
        ret.update(data)
        return ret


def collate_dicts(rows: list[dict[str, Any]], combine_tensors=True, combine_scalars=True):
    """
    Take a collection of dictionaries and create a batch.
    Adapted from webdataset/filters.py:default_collation_fn

    Args:
        rows: List of samples to be batched.
        combine_tensors: Whether to combine tensor-like objects into batches.
        combine_scalars: Whether to combine scalar values into numpy arrays.

    Returns:
        list: A batch of samples.
    """
    keys = set(rows[0].keys())
    # for row in rows[1:]:
    #     assert set(row.keys()) == keys, "keys don't match in different samples"
    result = {}
    for k in keys:
        row_tuple = [row[k] for row in rows]
        result[k] = combine_values(
            row_tuple, combine_tensors=combine_tensors, combine_scalars=combine_scalars
        )
    return result


def combine_values(b, combine_tensors=True, combine_scalars=True):
    if isinstance(b[0], (int, float)):
        if combine_scalars:
            b = np.array(list(b))
    elif isinstance(b[0], torch.Tensor):
        if combine_tensors:
            # shapes = set(x.shape for x in b)
            # assert len(shapes) == 1, f"all shapes must be equal in collation, got {shapes}"
            b = torch.stack(list(b), dim=0)
    elif isinstance(b[0], np.ndarray):
        if combine_tensors:
            # shapes = set(x.shape for x in b)
            # assert len(shapes) == 1, f"all shapes must be equal in collation, got {shapes}"
            b = np.stack(list(b), axis=0)
    else:
        b = list(b)
    return b


def main():
    simple_transform = get_simple_transform_to_tensor(224)
    text_aug = EntityNetTextAugCfg(
        replace_noun_synonym_chance=0.5,
        replace_noun_definition_chance=0.2,
        replace_noun_hierarchy_chance=0.2,
        replace_attr_query=0.4,
        alt_text_chance=0.5,
        min_sitelinks=40,
        hierarchy_balancing_strength=0.5,
        alt_text_min_vote=2.5,
    )
    for split in N_SHARDS.keys():
        if split == "train":
            print(f"skip train")
            continue
        print(f"---------- dataset {split} with random text")
        ds = EntityNetUrlIndexable(split, text_aug=text_aug, transform=simple_transform)
        for i, dp in enumerate(ds):
            if i >= 1:
                break
            print_datapoint(dp)
            print()
        print(f"---------- dataloader {split} with random text")
        dataloader = DataLoader(ds, batch_size=4, shuffle=True)
        for i, batch in enumerate(dataloader):
            if i >= 1:
                break
            print_datapoint(batch)
        print()
    print(f"---------- dataset val with all texts")
    text_aug = EntityNetTextAugCfg(return_mode=EntityNetTextReturnMode.ALL)
    ds = EntityNetUrlIndexable("val", text_aug=text_aug, transform=simple_transform)
    for i, dp in enumerate(ds):
        if i >= 3:
            break
        # print(dp["search_query"])
        print_datapoint(dp)
        print()
    print(f"---------- dataloader val with all texts")
    dataloader = DataLoader(ds, batch_size=4, shuffle=True, collate_fn=collate_dicts)
    for i, batch in enumerate(dataloader):
        if i >= 1:
            break
        print_datapoint(batch)
    print(f"done all texts")
    print()


if __name__ == "__main__":
    main()
