"""
Create Webdataset for iNat21
"""

import os
import random
from copy import deepcopy
from pathlib import Path

from crx.datasets.inat21 import iNat21
from webdataset import TarWriter

from packg.log import logger
from packg.multiproc import FnMultiProcessor
from visiontext.images import decode_jpeg
from visiontext.mathutils import distribute_evenly

from entitynet.paths import get_entitynet_data_dir


def generate_inat21_webdataset(data_dir, split: str, num_shards: int, verbose=False, workers=8):
    data_dir = Path(data_dir)
    dataset = iNat21(split)
    ann_images = dataset.ann_images
    ann_labels = dataset.ann_labels

    data = {}
    for i, (ann_image, ann_label) in enumerate(zip(ann_images, ann_labels)):
        assert ann_image["id"] == ann_label["id"] == ann_label["image_id"]
        image_id = ann_image["id"]
        cat_id = ann_label["category_id"]
        file_name_rel = ann_image["file_name"]
        data[image_id] = {
            "category_id": cat_id,
            "image_file": file_name_rel,
            # "width": ann_image["width"],
            # "height": ann_image["height"],
            # "latitude": ann_image.get("latitude", None),
            # "longitude": ann_image.get("longitude", None),
            # "location_uncertainty": ann_image.get("location_uncertainty", None),
        }

    print(f"{len(data)=} {num_shards=}")
    image_keys = list(data.keys())
    random_keys = deepcopy(image_keys)
    random.seed(42)
    random.shuffle(random_keys)
    per_shard = distribute_evenly(len(data), num_shards)
    target_wds = data_dir / f"tar_{split}/shard_%05d.tar"
    print(per_shard)

    mp = FnMultiProcessor(
        workers=workers,
        target_fn=create_shard,
        with_output=True,
        total=num_shards,
        desc=f"Shards for {split}",
    )
    current_pos = 0
    for shard_id in list(range(num_shards)):
        target_file_shard = target_wds.as_posix() % shard_id

        # get all images to be written to this shard
        end_pos = current_pos + per_shard[shard_id]
        shard_image_keys = image_keys[current_pos:end_pos]
        filenames = [data[k]["image_file"] for k in shard_image_keys]

        # collect all the metadata for these images
        metadata_list = []
        for k in shard_image_keys:
            metadata_here = data[k]
            metadata_list.append({"category_id": metadata_here["category_id"]})

        print(
            f"Writing shard {shard_id} {len(filenames)=} {len(shard_image_keys)=} "
            f"{len(metadata_list)=} {target_file_shard=}"
        )
        mp.put(target_file_shard, filenames, shard_image_keys, data_dir, metadata_list)
        current_pos = end_pos
    mp.run()
    mp.close()
    logger.info(f"Completed split: {split}")
    logger.info(f"Done")


def create_shard(target_file_shard, filenames, filekeys, base_dir, metadata_list):
    # this writes only the jpg to the shard
    target_file_shard = Path(target_file_shard).as_posix()
    base_dir = Path(base_dir)
    os.makedirs(Path(target_file_shard).parent, exist_ok=True)
    with TarWriter(target_file_shard) as fh:
        for filename, image_str_key, metadata in zip(filenames, filekeys, metadata_list):
            full_filename = base_dir / filename
            bytes_data = full_filename.read_bytes()
            _numpy_arr = decode_jpeg(bytes_data)
            obj = {
                "__key__": str(image_str_key),
                "jpg": bytes_data,
                "json": metadata,
            }
            fh.write(obj)


def main():
    generate_inat21_webdataset(
        get_entitynet_data_dir() / "iNat/2021", split="traindev", num_shards=32, verbose=True
    )
    # generate_inat21_webdataset(
    #     get_entitynet_data_dir() / "iNat/2021", split="trainnodev", num_shards=256, verbose=True
    # )


if __name__ == "__main__":
    main()
