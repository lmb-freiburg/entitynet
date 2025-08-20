import random
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from pprint import pprint

from natsort import natsorted

from packg.iotools import dump_json, load_json

from entitynet.paths import get_entitynet_annotations_dir, get_entitynet_data_dir

infos = {
    "traindev": {
        "description": "Validation set based on iNaturalist2019 training set. "
        + "Shuffles the training data and then selects 10 images per category.",
        "version": "1.1",
        "year": 2024,
        "date_created": "2024-07-15 12:00:00.000000",
    },
    "trainnodev": {
        "description": "Training set base don iNaturalist2021 training set, "
        + "without images from traindev set.",
        "version": "1.1",
        "year": 2024,
        "date_created": "2024-07-15 12:00:00.000000",
    },
    "trainsmall": {
        "description": "Smaller validation set based on iNat2019 train set. "
        + "Selects the first 10 images per category from the train set without shuffling.",
        "version": "1.1",
        "year": 2024,
        "date_created": "2024-07-15 12:00:00.000000",
    },
}


def generate_inat19_traindev_split(inat19_dir, verbose=False):
    """
    Create traindev (smaller validation set based on train data) and trainnodev
    (remaining train data without traindev).

    traindev here shuffles the training data and then takes 10 images per category.
    """
    traindev_file = inat19_dir / "inat2019_traindev_split.json"
    trainnodev_file = inat19_dir / "inat2019_trainnodev_split.json"

    in_file = Path(inat19_dir) / "train2019.json"
    in_data = load_json(in_file)
    if verbose:
        print(in_data.keys())  # ['info', 'images', 'licenses', 'annotations', 'categories']
        pprint(in_data["info"])
        print(f"{len(in_data['licenses'])=}")
        print(f"{len(in_data['categories'])=}")
        print(f"{len(in_data['annotations'])=}")
        print(f"{len(in_data['images'])=}")

    traindev = {
        "info": deepcopy(infos["traindev"]),
        "licenses": deepcopy(in_data["licenses"]),
        "categories": deepcopy(in_data["categories"]),
        "annotations": [],
        "images": [],
    }
    trainnodev = {
        "info": deepcopy(infos["trainnodev"]),
        "licenses": deepcopy(in_data["licenses"]),
        "categories": deepcopy(in_data["categories"]),
        "annotations": [],
        "images": [],
    }

    cat2anns = defaultdict(list)
    ann2cat = {}
    for i, ann in enumerate(in_data["annotations"]):
        assert ann["image_id"] == ann["id"] == i
        cat2anns[ann["category_id"]].append(i)
        ann2cat[i] = ann["category_id"]

    dev_ids = load_json(get_entitynet_annotations_dir() / "inat/inat2019_traindev_ids.json")
    dev_ids_set = set(dev_ids)
    nodev_ids = []
    for cat, image_ids in cat2anns.items():
        for image_id in image_ids:
            if image_id not in dev_ids_set:
                nodev_ids.append(image_id)

    # # original code that produced the traindev split
    # dev_ids, nodev_ids = [], []
    # for cat, image_ids in cat2anns.items():
    #     assert len(image_ids) > 10, f"Category {cat} has less than 11 images"
    #     dev_ids += image_ids[:10]
    #     nodev_ids += image_ids[10:]
    # random.seed(42)
    # random.shuffle(dev_ids)
    # random.shuffle(nodev_ids)

    ids_cut = set(dev_ids) & set(nodev_ids)
    assert len(ids_cut) == 0, f"Intersection of dev_ids and nodev_ids: {ids_cut}"

    for image_id in dev_ids:
        traindev["annotations"].append(in_data["annotations"][image_id])
        traindev["images"].append(in_data["images"][image_id])
    for image_id in nodev_ids:
        trainnodev["annotations"].append(in_data["annotations"][image_id])
        trainnodev["images"].append(in_data["images"][image_id])

    if traindev_file.is_file():
        print(f"Already exists, skipping save: {traindev_file}")
    else:
        dump_json(traindev, traindev_file, indent=2, verbose=verbose)
    if trainnodev_file.is_file():
        print(f"Already exists, skipping save: {trainnodev_file}")
    else:
        dump_json(trainnodev, trainnodev_file, indent=2, verbose=verbose)


def main():
    dataset_dir = get_entitynet_data_dir() / "iNat/2019"
    generate_inat19_traindev_split(dataset_dir, verbose=True)
    # create list of images as txt, for each split
    for file in [
        "train2019.json",
        "val2019.json",
        "inat2019_traindev_split.json",
        "inat2019_trainnodev_split.json",
        "test2019.json",
    ]:
        filelist_file = f"filelist_{file[:-5]}.txt"
        file_full = dataset_dir / file
        filelist_file_full = dataset_dir / filelist_file
        if filelist_file_full.is_file():
            print(f"Already exists, skipping filelist creation: {filelist_file_full}")
            continue
        if not file_full.is_file():
            print(f"File does not exist, skipping filelist creation: {file_full}")
            continue
        data = load_json(file_full)
        image_data = data["images"]
        files = natsorted([image["file_name"] for image in image_data])
        files.append("")
        Path(filelist_file_full).write_text("\n".join(files))
        print(f"Saved {filelist_file_full}")


if __name__ == "__main__":
    main()
