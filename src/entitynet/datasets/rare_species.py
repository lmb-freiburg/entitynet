"""
pip install -U datasets fsspec huggingface-hub

note that "common name" has two duplicates.

columns ['image', 'rarespecies_id', 'eol_content_id', 'eol_page_id', 'kingdom', 'phylum', 'class',
'order', 'family', 'genus', 'species', 'sciname', 'common']
"""

import os
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from packg.iotools import dump_json, load_json
from packg.log import logger
from packg.tqdmext import tqdm_max_ncols
from packg.typext import PathType

from entitynet.paths import get_entitynet_data_dir

RARE_SPECIES_LANGUAGES = ["sci", "com", "tax", "scicom", "taxcom"]
_DSET_NAME = f"imageomics/rare-species"
_BASE_DIR_REL = "rare-species"
_COLUMNS = [
    "image",
    "rarespecies_id",
    "eol_content_id",
    "eol_page_id",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "sciName",
    "common",
]


class RareSpecies(Dataset):
    def __init__(
        self,
        split,
        transform=None,
        return_dict=False,
        max_datapoints=None,
        language=RARE_SPECIES_LANGUAGES[0],
    ):
        assert split == "train", f"Only train split supported for {type(self).__name__}"
        assert (
            language in RARE_SPECIES_LANGUAGES
        ), f"{language=} not supported. Available: {RARE_SPECIES_LANGUAGES}"
        data_dir = get_entitynet_data_dir()
        base_dir_rel = _BASE_DIR_REL
        base_dir = data_dir / base_dir_rel

        # hf datasets is terribly slow so we cache the dataset instead.
        ann_file = base_dir / f"{split}_cache.json"
        class_labels_file = base_dir / "class_labels.json"
        if not ann_file.is_file():
            build_rare_species_from_huggingface(base_dir, ann_file, class_labels_file)
        class_labels_all = load_json(class_labels_file)
        class_labels = class_labels_all[language]
        ann_images = load_json(ann_file)

        if max_datapoints is not None and max_datapoints > 0:
            ann_images = ann_images[:max_datapoints]
            logger.warning(f"Reduced dataset to {len(ann_images)} since {max_datapoints=}")

        self.transform = transform
        self.ann_images = ann_images
        self.classes = class_labels
        self.return_dict = return_dict
        self.base_dir = base_dir

    def __len__(self):
        return len(self.ann_images)

    def __getitem__(self, idx):
        ann = self.ann_images[idx]
        image_file = self.base_dir / ann["file_name"]
        image = Image.open(image_file).convert("RGB")
        target = ann["class_id"]
        if self.transform is not None:
            image = self.transform(image)
        if self.return_dict:
            return {"image": image, "label": target, "idx": idx}
        return image, target


def build_rare_species_from_huggingface(
    base_dir: PathType, ann_file: PathType, class_labels_file: PathType
):
    base_dir, ann_file, class_labels_file = Path(base_dir), Path(ann_file), Path(class_labels_file)
    logger.info(f"Creating RareSpecies dataset from huggingface")
    from datasets import Dataset as HFDataset
    from datasets import Image as HFImage
    from datasets import load_dataset

    logger.info(f"Loading {_DSET_NAME} from huggingface")
    # create the datasets twice, once with images, once with image paths
    dataset: HFDataset = load_dataset(
        _DSET_NAME,
        streaming=False,
        split="train",
        revision="06e9eae",
    ).cast_column("image", HFImage(decode=False))
    ann_images = []
    for i in tqdm_max_ncols(list(range(len(dataset))), desc="Saving rare species images"):
        datapoint = dataset[i]
        ann_dict = {}
        for column in _COLUMNS:
            if column == "image":
                continue
            ann_dict[column.lower()] = dataset[i][column]

        image_source_file = datapoint["image"]["path"]
        rel_path = (Path("images") / "/".join(Path(image_source_file).parts[-2:])).as_posix()
        # images/Animalia-Arthropoda-Arachnida-Araneae-Pisauridae-Dolomedes-plantarius/
        # 10797020_1198625_eol-full-size-copy.jpg

        # pil_image: Image.Image = dataset[i]["image"]
        image_file = base_dir / rel_path
        ann_dict["file_name"] = rel_path
        ann_images.append(ann_dict)
        if not image_file.is_file():
            os.makedirs(image_file.parent, exist_ok=True)
            shutil.copy2(image_source_file, image_file)

    # unique scientific names define the classes
    scinames = [ann["sciname"] for ann in ann_images]
    sci_unique = sorted(set(scinames))
    sci2id = {species: i for i, species in enumerate(sci_unique)}
    print(f"{len(scinames)=} {len(sci_unique)=}")

    # add the scientific name id (class id) to the annotations
    # also build the taxonomic names and add them
    sciname2others = {"com": {}, "tax": {}, "scicom": {}, "taxcom": {}}
    for ann in ann_images:
        sciname = ann["sciname"]
        comname = ann["common"]
        ann["class_id"] = sci2id[sciname]
        # common name
        sciname2others["com"][sciname] = comname
        # taxonomy name: kingdom, phylum, class, order, family, genus and species
        taxname = " ".join(
            [ann[k] for k in ["kingdom", "phylum", "class", "order", "family", "genus", "species"]]
        )
        sciname2others["tax"][sciname] = taxname
        ann["taxname"] = taxname
        # scicom: scientific + common
        scicomname = f"{sciname} with common name {comname}"
        sciname2others["scicom"][sciname] = scicomname
        ann["scicomname"] = scicomname
        # taxcom: taxonomic + common
        taxcomname = f"{taxname} with common name {comname}"
        sciname2others["taxcom"][sciname] = taxcomname
        ann["taxcomname"] = taxcomname
    for language, namelist in sciname2others.items():
        namelist_unique = [namelist[sciname] for sciname in sci_unique]
        if len(set(namelist_unique)) != len(sci_unique):
            logger.warning(
                f"Duplicate {language} names found! {len(sci_unique)=} {len(set(namelist_unique))=}"
            )
            cter = {}
            for k, v in namelist.items():
                if v in cter:
                    logger.warning(f"New species {k} also named {v} same as species {cter[v]}")
                else:
                    cter[v] = k
    class_label_dict = {
        "sci": sci_unique,
    }
    for language, namelist in sciname2others.items():
        class_label_dict[language] = [namelist[sciname] for sciname in sci_unique]

    dump_json(ann_images, ann_file, verbose=True, indent=2)
    dump_json(
        class_label_dict,
        class_labels_file,
        verbose=True,
        indent=2,
        custom_format=False,
    )


def main():
    ds = RareSpecies("train")
    print(ds[0])
    print(f"Done")

    # print(f"Start loading dataset...")
    # from datasets import load_dataset, Image as HFImage, Dataset as HFDataset
    # dataset: HFDataset = load_dataset(_DSET_NAME, streaming=False, split="train")
    # dataset_with_paths: HFDataset = (
    #     load_dataset(_DSET_NAME, streaming=False, split="train")
    #     .select_columns(["image", "rarespecies_id"])
    #     .cast_column("image", HFImage(decode=False))
    # )
    # breakpoint()
    # print(f"asd")
    #
    # dataset = load_dataset(
    #     f"imageomics/rare-species",
    #     # use_auth_token=True,  # required
    #     # language=lang,
    #     streaming=False,  # this downloads to disk
    #     split="train",
    #     cache_dir=(get_entitynet_data_dir() / "rare-species" / "hf_cache").as_posix(),
    # ).cast_column("image", Image(decode=False))
    # breakpoint()
    # print(dataset)
    # info: DatasetInfo = dataset.info
    #


if __name__ == "__main__":
    main()
