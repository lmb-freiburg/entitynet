"""
TODO it's very slow. profile it / tar it if necessary. for now use 19?

Source: github.com/visipedia/inat_comp

Splits are train 2.7M train_mini 500K val 100k public_test 500k
train_mini is a subset of train
we additionally created splits "traindev" with 10 randomly selected images per category,
and "trainnodev" with all remaining images.

Annotation data format is coco style:
{
    "info": dict,
    "licenses": dict,
    "images": [
        'id' = {int} 0
        'width' = {int} 500
        'height' = {int} 500
        'file_name' = {str} 'train_mini/02912_Animalia_Chordata_Actinopterygii_Siluriformes_Ictaluridae_Ameiurus_nebulosus/d615f184-8af4-4c60-b9f8-3081c1607644.jpg'
        'license' = {int} 0
        'rights_holder' = {str} 'Ken-ichi Ueda'
        'date' = {str} '2010-07-14 20:19:00+00:00'
        'latitude' = {float} 43.83486
        'longitude' = {float} -71.22231
        'location_uncertainty' = {int} 77
    ],
    "categories: [  # 10k categories total
        'id' = {int} 0
        'name' = {str} 'Lumbricus terrestris'
        'common_name' = {str} 'Common Earthworm'
        'supercategory' = {str} 'Animalia'
        'kingdom' = {str} 'Animalia'
        'phylum' = {str} 'Annelida'
        'class' = {str} 'Clitellata'
        'order' = {str} 'Haplotaxida'
        'family' = {str} 'Lumbricidae'
        'genus' = {str} 'Lumbricus'
        'specific_epithet' = {str} 'terrestris'
        'image_dir_name' = {str} '00000_Animalia_Annelida_Clitellata_Haplotaxida_Lumbricidae_Lumbricus_terrestris',
    ],
    "annotations": [  # one for each image
        'id' = {int} 0
        'image_id' = {int} 0
        'category_id' = {int} 2912
    ]


"""

from pathlib import Path

from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg

from packg.iotools import load_json
from typedparser.objects import repr_value

from entitynet.paths import get_entitynet_annotations_dir, get_entitynet_data_dir


class iNat21TextLoader:
    def __init__(
        self,
        language: str = "en",
        categories: list[dict] | None = None,
    ):
        if categories is None:
            categories = load_json(
                get_entitynet_annotations_dir() / "inat/inat2021_categories.json"
            )
        classes_en = [category["common_name"] for category in categories]
        classes_latin = [category["name"] for category in categories]
        if language == "en":
            classes = classes_en
        elif language == "latin":
            classes = classes_latin
        else:
            raise ValueError(f"Language {language} not supported")
        class_to_idx = dict(zip(classes, range(len(classes))))

        self.classes = classes
        self.categories = categories

    def get_text_for_category_id(self, category_id):
        return self.classes[category_id]


class iNat21(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        language: str = "en",
        return_dict=False,
        max_datapoints=None,
    ):
        self.transform = transform
        self.return_dict = return_dict
        self.data_dir = get_entitynet_data_dir()
        self.base_dir_rel = "iNat/2021"
        base_dir = self.data_dir / self.base_dir_rel

        # load metadata for split
        special_splits = ("traindev", "trainnodev")
        self.split = verify_str_arg(
            split, "split", ("train", "val", "public_test", "train_mini") + special_splits
        )
        if self.split in special_splits:
            ann_file = base_dir / f"inat2021_{self.split}_split.json"
            if not ann_file.is_file():
                raise RuntimeError(f"Run inat21_generate_splits.py to create split {self.split}")
        else:
            ann_file = base_dir / f"{self.split}.json"
        ann_all = load_json(ann_file)
        self.ann_images = ann_all["images"]
        self.ann_labels = ann_all.get("annotations", None)
        if self.ann_labels is None:
            logger.debug(f"No labels found for dataset {type(self).__name__} {split}")

        if max_datapoints is not None and max_datapoints > 0:
            self.ann_images = self.ann_images[:max_datapoints]
            if self.ann_labels is not None:
                self.ann_labels = self.ann_labels[:max_datapoints]

        self.text_loader = iNat21TextLoader(language=language, categories=ann_all["categories"])
        self.classes = self.text_loader.classes

    def __len__(self):
        return len(self.ann_images)

    def get_image_file_rel(self, idx):
        return (Path(self.base_dir_rel) / self.ann_images[idx]["file_name"]).as_posix()

    def __getitem__(self, idx):
        image_file = self.data_dir / self.get_image_file_rel(idx)
        image = Image.open(image_file).convert("RGB")
        ann_label = self.ann_labels[idx]
        category_id = ann_label["category_id"]
        if self.transform:
            image = self.transform(image)
        if self.return_dict:
            text = self.text_loader.get_text_for_category_id(category_id)
            return {"image": image, "label": category_id, "idx": idx, "text": text}
        return image, category_id


def main():
    for split in ("train", "val", "public_test", "train_mini", "traindev", "trainnodev"):
        dataset = iNat21(split)
        print(f"{split=} {len(dataset)=}")
        print(repr_value(dataset[0]))


if __name__ == "__main__":
    main()
