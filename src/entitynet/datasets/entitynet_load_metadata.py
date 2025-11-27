"""
# load all metadata at once in case we need it

"""

import torch
from loguru import logger
from natsort import natsorted

from packg.iotools import dump_json, load_json
from packg.tqdmext import tqdm_max_ncols

from entitynet.paths import get_entitynet_data_dir


class _MetadataDataset(torch.utils.data.Dataset):
    def __init__(self, json_files):
        self.json_files = json_files

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        data = load_json(json_file)
        return idx, json_file, data

    def no_collate_fn(self, batch):
        return batch[0]


def load_entitynet_metadata(
    split: str,
    num_workers: int = 0,
    disable_pbar: bool = True,
):
    base_dir = get_entitynet_data_dir() / f"entitynet"
    json_files = natsorted(list((base_dir / f"data_{split}").glob("metadata_*.json")))
    logger.info(f"Load {len(json_files)} metadata files")
    dset = _MetadataDataset(json_files)
    if num_workers > 0:
        dloader = torch.utils.data.DataLoader(
            dset,
            batch_size=1,
            collate_fn=dset.no_collate_fn,
            num_workers=num_workers,
            shuffle=False,
        )
    else:
        dloader = dset
    metadata = {}
    for i, (idx, json_file, data) in enumerate(
        tqdm_max_ncols(dloader, desc="Load metadata", total=len(dset), disable=disable_pbar)
    ):
        idx_from_json_file = int(json_file.stem.split("_")[-1].split(".")[0])
        assert idx == idx_from_json_file, f"{idx=} {idx_from_json_file=} {json_file=}"
        for k, v in data.items():
            v["shard"] = idx
            metadata[k] = v

    return metadata


def load_entitynet_metadata_cached(
    split: str,
    num_workers: int = 0,
    disable_pbar: bool = True,
):
    base_dir = get_entitynet_data_dir() / f"entitynet"
    cache_file = base_dir / "cache" / f"cached_metadata_{split}.json"
    if cache_file.is_file():
        logger.info(f"Loading metadata from cache: {cache_file}")
        return load_json(cache_file)

    logger.info("Cache not found, loading metadata from disk...")
    metadata = load_entitynet_metadata(
        split=split,
        num_workers=num_workers,
        disable_pbar=disable_pbar,
    )
    logger.info(f"Caching {len(metadata)} metadata entries to {cache_file}")
    dump_json(metadata, cache_file, indent=1)
    return metadata


def main():
    for split in ["train", "val", "test"]:
        metadata = load_entitynet_metadata_cached(split, num_workers=8, disable_pbar=False)
        logger.info(f"Loaded {len(metadata)} metadata entries for {split=}")


if __name__ == "__main__":
    main()
