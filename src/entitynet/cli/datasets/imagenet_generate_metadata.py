"""
Generates the metadata that splits imagenet into living and non-living classes.

Required files:
$ENTITYNET_DATA_DIR/imagenet1k/
    train/
    test/

Note: classes_data.json and val.json were created before, using lmb-freiburg/ovqa repository
"""

import random
from collections import Counter, defaultdict
from copy import deepcopy
from glob import glob
from operator import itemgetter
from pathlib import Path

from loguru import logger
from natsort import natsorted
from scipy.io import loadmat
from torchvision.datasets.utils import extract_archive

from packg.iotools import dump_json, dump_json_xz, load_json, load_json_xz
from packg.log import configure_logger
from packg.tqdmext import tqdm_max_ncols
from typedparser.objects import compare_nested_objects

from crx.datasets.wordnet import convert_wnid_to_synname, load_wordnet_nouns
from entitynet.paths import get_entitynet_annotations_dir, get_entitynet_data_dir


def main():
    configure_logger(level="INFO")
    # prepare classes data
    anno_dir = get_entitynet_annotations_dir()
    anno_dir_imagenet = anno_dir / "imagenet"
    data_dir = get_entitynet_data_dir()
    data_dir_imagenet = data_dir / "imagenet1k"
    logger.info(f"Preparing imagenet, {data_dir_imagenet=}, {anno_dir_imagenet=}")

    classes_data = create_classes_data(anno_dir_imagenet, data_dir_imagenet)
    create_living_things_labels(anno_dir_imagenet, data_dir_imagenet, classes_data)

    wnid_to_class_idx = {c["synset"]: c["class_idx"] for c in classes_data}
    # create_test_split(data_dir)  # it doesn't have labels anyway
    create_val_split(anno_dir_imagenet, data_dir_imagenet, wnid_to_class_idx)

    # use trainsmall for model development to avoid overfitting to val
    create_train_split(anno_dir_imagenet, data_dir_imagenet, wnid_to_class_idx)
    create_trainsmall_split(anno_dir_imagenet)


def create_classes_data(anno_dir_imagenet, data_dir_imagenet):
    classes_data_file = anno_dir_imagenet / "generated/classes_data.json"
    if not classes_data_file.is_file():
        classes_data = load_imagenet_classes(data_dir_imagenet, anno_dir_imagenet)
        dump_json(classes_data, classes_data_file, indent=2, create_parent=True)
    classes_data = load_json(classes_data_file)
    logger.info(f"Got {len(classes_data)} classes")
    logger.info(classes_data[0])
    return classes_data


def create_living_things_labels(anno_dir_imagenet, data_dir_imagenet, classes_data):
    logger.info(f"Creating imagenet livingthings labels from wordnet")
    # figure out which of the 82k nouns in wordnet is child of living_thing
    nouns = load_wordnet_nouns()

    def set_is_living_things(synname):
        nouns[synname]["is_living_thing"] = True
        for children_synname in nouns[synname]["children"]:
            set_is_living_things(children_synname)

    for noun_key, noun_value in nouns.items():
        noun_value["is_living_thing"] = False
    set_is_living_things("living_thing.n.01")
    print("living:", Counter([nouns[synname]["is_living_thing"] for synname in nouns.keys()]))

    class_idx_livingthing, class_idx_other = 0, 0
    class_data_livingthing, class_data_other = [], []
    for class_dict in classes_data:
        synname = convert_wnid_to_synname(class_dict["synset"])
        noun = nouns[synname]
        is_living_thing = noun["is_living_thing"]
        class_idx_1k = class_dict["class_idx"]
        if not is_living_thing:
            class_data_other.append(
                {
                    "class_idx": class_idx_other,
                    "class_idx_1k": class_idx_1k,
                    "clip_bench_label": class_dict["clip_bench_label"],
                    "synname": synname,
                }
            )
            class_idx_other += 1
        else:
            class_data_livingthing.append(
                {
                    "class_idx": class_idx_livingthing,
                    "class_idx_1k": class_idx_1k,
                    "clip_bench_label": class_dict["clip_bench_label"],
                    "synname": synname,
                }
            )
            class_idx_livingthing += 1
    print(
        f"Got {len(class_data_livingthing)} livingthing classes and "
        f"{len(class_data_other)} other classes"
    )
    outf = anno_dir_imagenet / "generated/classes_data_living_things.json"
    if outf.is_file():
        print(f"File {outf} exists")
    else:
        dump_json(class_data_livingthing, outf, indent=2)

    outf = anno_dir_imagenet / "generated/classes_data_not_living_things.json"
    if outf.is_file():
        print(f"File {outf} exists")
    else:
        dump_json(class_data_other, outf, indent=2)


def create_val_split(anno_dir_imagenet, data_dir_imagenet, wnid_to_class_idx):
    """
    Create the val split from the imagenet1k/val directory.
    """
    logger.info(f"Globbing val images in {data_dir_imagenet}")
    val_files = glob((data_dir_imagenet / "val/**/*.JPEG").as_posix(), recursive=True)
    assert len(val_files) > 0, f"No val images found in {data_dir_imagenet}"
    logger.info(f"Found {len(val_files)} val images")
    idx2dict_val = {}
    for i, val_file in enumerate(tqdm_max_ncols(val_files)):
        if i == 0:
            print("\n" + f"{val_file}")

        rel_file = Path(val_file).relative_to(data_dir_imagenet).as_posix()
        split, wnid, fn = rel_file.split("/")
        fnno = fn.split(".")[0]
        _, _, number_str = fnno.split("_")
        idx = f"{split}_{number_str}"
        idx2dict_val[idx] = {"class_idx": wnid_to_class_idx[wnid], "image": rel_file}
    print(f"Found {len(idx2dict_val)} val images")
    val_dict = {}
    for num in natsorted(list(idx2dict_val.keys())):
        content = idx2dict_val[num]
        val_dict[num] = content
    val_file = anno_dir_imagenet / "generated/val.json"
    if val_file.exists():
        print(f"Already exists, skipping save: {val_file}")
        old_val_dict = load_json(val_file)
        assert compare_nested_objects(val_dict, old_val_dict) == []
    else:
        dump_json(val_dict, val_file, indent=2, create_parent=True)


def create_train_split(anno_dir_imagenet, data_dir_imagenet, wnid_to_class_idx):
    """
    Create the train split from the imagenet1k/train directory.
    """
    logger.info(f"Globbing train images in {data_dir_imagenet}")
    train_files = glob((data_dir_imagenet / "train/**/*.JPEG").as_posix(), recursive=True)
    assert len(train_files) > 0, f"No train images found in {data_dir_imagenet}"
    logger.info(f"Found {len(train_files)} train images")
    idx2dict = {}
    for i, train_metadata_file in enumerate(tqdm_max_ncols(train_files)):
        if i == 0:
            print("\n" + f"{train_metadata_file}")

        rel_file = Path(train_metadata_file).relative_to(data_dir_imagenet).as_posix()
        split, wnid, fn = rel_file.split("/")
        fnno = fn.split(".")[0]
        wnid_again, number_str = fnno.split("_")
        if wnid != wnid_again:
            raise ValueError(f"wnid != wnid_again: {wnid} != {wnid_again} for {rel_file}")
        idx = f"{split}_{wnid}_{number_str}"
        idx2dict[idx] = {"class_idx": wnid_to_class_idx[wnid], "image": rel_file}
    print(f"Found {len(idx2dict)} images")
    train_dict = {}
    for num in natsorted(list(idx2dict.keys())):
        content = idx2dict[num]
        train_dict[num] = content
    train_metadata_file = anno_dir_imagenet / "generated/train.json"
    if train_metadata_file.exists():
        print(f"Already exists, skipping save: {train_metadata_file}")
    else:
        dump_json(train_dict, train_metadata_file, indent=2, create_parent=True)


def create_trainsmall_split(anno_dir_imagenet):
    new_split = "trainsmall"
    n_images_per_class = 50
    print(f"Creating {new_split}.json with {n_images_per_class=}")
    train_metadata_file = anno_dir_imagenet / "generated/train.json"
    train_data = load_json(train_metadata_file)
    class2keys = defaultdict(list)
    for key, val in train_data.items():
        class_idx = val["class_idx"]
        class2keys[class_idx].append(key)
    random.seed(42)
    new_train_data = {}
    for class_idx, keys in sorted(class2keys.items(), key=itemgetter(0)):
        random_keys = deepcopy(keys)
        random.shuffle(random_keys)
        keys_newval = sorted(random_keys[:n_images_per_class])
        for key in keys_newval:
            new_train_data[key] = deepcopy(train_data[key])
    new_split_file = get_entitynet_annotations_dir() / f"imagenet/generated/{new_split}.json"
    if new_split_file.is_file():
        print(f"Already exists, skipping save: {new_split_file}")
    else:
        dump_json(new_train_data, new_split_file, indent=2)

    new_train_data = load_json(new_split_file)
    print(f"Got split {new_split} with length {len(new_train_data)}")
    trainsmall_images = natsorted([v["image"] for v in new_train_data.values()])
    trainsmall_txt = anno_dir_imagenet / "generated/trainsmall_images.txt"
    with open(trainsmall_txt, "w") as f:
        for image in trainsmall_images:
            f.write(image + "\n")


def load_imagenet_classes(path, anno_path):
    # get the class number to original data mapping
    # this includes some other synsets (1860 classes in total)
    devkit12_file = path / "ILSVRC2012_devkit_t12.tar.gz"
    meta_mat_file = path / "ILSVRC2012_devkit_t12/data/meta.mat"
    if not meta_mat_file.is_file():
        logger.info(f"Extracting {devkit12_file} to {meta_mat_file}")
        extract_archive(devkit12_file, path)

    meta = loadmat((path / "ILSVRC2012_devkit_t12/data/meta.mat").as_posix())
    synsets = meta["synsets"]
    clsnum_to_synset_all, clsnum_to_label, clsnum_to_descr = {}, {}, {}
    for s in synsets:
        idx = int(s[0][0][0][0])  # class number starting with 1
        wnid = s[0][1][0]  # wordnet synset e.g. n02012849
        clsnum_to_synset_all[idx] = wnid
        clsnum_to_label[idx] = s[0][2][0]
        clsnum_to_descr[idx] = s[0][3][0]
    assert len(clsnum_to_synset_all) == 1860  # 1000

    # load the clsidx -> (synset, keras_label) mapping
    clsidx_to_synset_and_keras: dict[str, tuple[str, str]] = load_json(
        anno_path / "external/imagenet_class_index.json"
    )
    clsidx_to_synset, clsidx_to_keras = {}, {}

    for clsidx_str, (synset, keras_label) in clsidx_to_synset_and_keras.items():
        clsidx = int(clsidx_str)
        clsidx_to_synset[clsidx] = synset
        clsidx_to_keras[clsidx] = keras_label

    # build clsnum_to_clsidx mapping (only for 1000 actual classes)
    synset_to_clsidx = {v: k for k, v in clsidx_to_synset.items()}
    clsnum_to_clsidx = {
        k: synset_to_clsidx[v] for k, v in clsnum_to_synset_all.items() if v in synset_to_clsidx
    }

    clsidx_to_content = {}
    for clsnum, clsidx in clsnum_to_clsidx.items():
        clsidx_to_content[clsidx] = {
            "old_class_num": clsnum,
            "class_idx": clsidx,
            "synset": clsidx_to_synset[clsidx],
            "orig_descr": clsnum_to_descr[clsnum],
        }
    assert len(clsidx_to_content) == 1000
    all_classes = [clsidx_to_content[clsidx] for clsidx in range(1000)]

    # load clip benchmark labels list
    clipbenchlabels = load_json(anno_path / "external" / "clip_benchmark_classes_fixed.json")
    assert len(clipbenchlabels) == 1000

    for clsidx, content in clsidx_to_content.items():
        content["clip_bench_label"] = clipbenchlabels[clsidx]
    return all_classes


def create_test_split(data_dir):
    """
    Create the test split from the imagenet1k/test directory.
    """
    logger.info(f"Globbing test images in {data_dir}")
    test_files = glob((data_dir / "test/*.JPEG").as_posix(), recursive=True)
    assert len(test_files) > 0, "No test images found"
    idx2dict_test = {}
    for i, test_file in enumerate(tqdm_max_ncols(test_files)):
        if i == 0:
            print("\n" + f"{test_file}")
        rel_file_test = Path(test_file).relative_to(data_dir).as_posix()
        _, split, num_str = rel_file_test.split(".")[0].split("_")
        assert split == "test", rel_file_test

        idx = f"{split}_{num_str}"
        idx2dict_test[idx] = {"image": rel_file_test}
    print(f"Found {len(idx2dict_test)} test images")
    test_dict = {}
    for num in natsorted(list(idx2dict_test.keys())):
        content = idx2dict_test[num]
        test_dict[num] = content
    test_json_file = get_entitynet_annotations_dir() / "imagenet/generated/test.json"
    if test_json_file.exists():
        print(f"Already exists, skipping save: {test_json_file}")
    else:
        dump_json(test_dict, test_json_file, indent=2)
        # dump_json_xz(test_dict, test_json_file, indent=2)


if __name__ == "__main__":
    main()
