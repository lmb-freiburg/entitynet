"""
Source: github.com/visipedia/inat_comp
"""

from pathlib import Path

from loguru import logger
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.utils import verify_str_arg

from packg.iotools import load_json

from entitynet.paths import get_entitynet_annotations_dir, get_entitynet_data_dir


class iNat19(Dataset):
    def __init__(
        self, split, transform=None, language="en", return_dict=False, max_datapoints=None
    ):
        self.return_dict = return_dict
        self.data_dir = get_entitynet_data_dir()
        self.base_dir_rel = "iNat/2019"
        base_dir = self.data_dir / self.base_dir_rel
        special_splits = ("trainsmall", "traindev", "trainnodev")
        self.split = verify_str_arg(split, "split", ("train", "val", "test") + special_splits)
        if self.split in special_splits:
            ann_file = base_dir / f"inat2019_{self.split}_split.json"
            if not ann_file.is_file():
                raise RuntimeError(f"Run inat19_generate_splits.py to create split {self.split}")
        else:
            ann_file = base_dir / f"{self.split}2019.json"
        ann_all = load_json(ann_file)
        self.ann_images = ann_all["images"]
        self.ann_categories_en = load_json(
            get_entitynet_annotations_dir() / "inat/inat2019_categories_common_no_rep.json"
        )
        self.ann_categories_latin = load_json(base_dir / "categories.json")
        self.transform = transform
        classes_en = [category["common_name"] for category in self.ann_categories_en]
        classes_latin = [category["name"] for category in self.ann_categories_latin]
        if language == "en":
            self.classes = classes_en
        elif language == "latin":
            self.classes = classes_latin
        else:
            raise ValueError(f"Language {language} not supported")
        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))

        # # superclasses can be inferred from the filename
        self.superclasses = ["Amphibians", "Birds", "Fungi", "Insects", "Plants", "Reptiles"]
        self.superclass2idx = {k: i for i, k in enumerate(self.superclasses)}
        """
        for image_ann in self.ann_images:
            file_name = image_ann["file_name"]
            superclass = Path(file_name).parts[1]
            image_ann["superclass_idx"] = self.superclass2idx[superclass]
        """
        self.ann_labels = ann_all.get("annotations", None)
        if self.ann_labels is None:
            logger.debug(f"No labels found for dataset {type(self).__name__} {split}")
        if max_datapoints is not None and max_datapoints > 0:
            self.ann_images = self.ann_images[:max_datapoints]
            if self.ann_labels is not None:
                self.ann_labels = self.ann_labels[:max_datapoints]
            logger.warning(f"Reduced dataset to {len(self.ann_images)} since {max_datapoints=}")

    def __len__(self):
        return len(self.ann_images)

    def get_image_file_rel(self, idx):
        return (Path(self.base_dir_rel) / self.ann_images[idx]["file_name"]).as_posix()

    def __getitem__(self, idx):
        image_file = self.data_dir / self.get_image_file_rel(idx)
        image = Image.open(image_file).convert("RGB")
        if self.ann_labels is None:
            raise RuntimeError(
                f"No labels exist for dataset {type(self).__name__} {self.split} (test set?)"
            )
        ann_label = self.ann_labels[idx]
        category_id = ann_label["category_id"]
        category = self.ann_categories_en[category_id]
        target = category["id"]
        if self.transform:
            image = self.transform(image)
        if self.return_dict:
            return {"image": image, "label": target, "idx": idx}
        return image, target


def main():
    for split in ("train", "val", "test", "trainsmall", "traindev", "trainnodev"):
        dataset = iNat19(split)
        print(f"{split=} {len(dataset)=}")


if __name__ == "__main__":
    main()
