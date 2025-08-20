"""
TODO check if the scripts work / are necessary:
- See imagenet_generate_metadata.py for generating the annotations.
- See create_livingthings_hierarchy.py for generating the hierarchy.
"""

from pathlib import Path

from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

from packg.iotools import load_json, load_json_xz
from visiontext.images import JPEGDecoderConst, decode_jpeg

from entitynet.paths import get_entitynet_annotations_dir, get_entitynet_data_dir


class Imagenet(Dataset):
    def __init__(
        self, split, name: str = "default", transform=None, return_dict=False, max_datapoints=None
    ):
        self.return_dict = return_dict
        self.data_dir = get_entitynet_data_dir()
        self.base_dir_rel = "imagenet1k"
        for check_subdir in ["train", "val"]:
            if not (self.data_dir / self.base_dir_rel / check_subdir).is_dir():
                raise FileNotFoundError(
                    f"Directory not found: {self.data_dir / self.base_dir_rel / check_subdir} - "
                    f"ImageNet image folders train and val must be downloaded and extracted first."
                )

        # load image annotations
        ann_dir = get_entitynet_annotations_dir() / "imagenet"
        split_file = ann_dir / "generated" / f"{split}.json"
        if not split_file.is_file():
            split_file_xz = ann_dir / "generated" / f"{split}.json.xz"
            if not split_file_xz.is_file():
                raise FileNotFoundError(f"Both files not found: {split_file} or {split_file_xz}")
            ann_data = load_json_xz(split_file_xz)
        else:
            ann_data = load_json(split_file)

        # load classes
        if name == "default":
            classes_data = load_json(ann_dir / "generated" / "classes_data.json")
            classes = [v["clip_bench_label"] for v in classes_data]
        else:
            classes_data = load_json(ann_dir / "generated" / f"classes_data_{name}.json")
            class_idx_old2new = {v["class_idx_1k"]: v["class_idx"] for v in classes_data}
            classes = [v["clip_bench_label"] for v in classes_data]
            synnames = [v["synname"] for v in classes_data]
            new_ann_data = {}
            for ann_key, ann_val in ann_data.items():
                old_class = ann_val["class_idx"]
                new_class = class_idx_old2new.get(old_class, None)
                if new_class is None:
                    continue
                new_ann_data[ann_key] = {
                    "image": ann_val["image"],
                    "class_idx": new_class,
                }
            logger.debug(
                f"Restricted imagenet for {name} new {len(new_ann_data)} datapoints / "
                f"{len(classes)} classes, old {len(ann_data)} datapoints"
            )
            ann_data = new_ann_data
        self.classes = classes
        self.split = split
        if max_datapoints is not None:
            ann_data = dict(list(ann_data.items())[:max_datapoints])
            logger.warning(f"Restricting imagenet to {max_datapoints} datapoints")
        self.ann_images = ann_data
        self.ann_keys = list(ann_data.keys())
        self.transform = transform

    def __len__(self):
        return len(self.ann_images)

    def get_image_file_rel(self, idx):
        key = self.ann_keys[idx]
        return (Path(self.base_dir_rel) / self.ann_images[key]["image"]).as_posix()

    def __getitem__(self, idx):
        ann_key = self.ann_keys[idx]
        image_file = self.data_dir / self.get_image_file_rel(idx)
        try:
            image_arr = decode_jpeg(image_file.read_bytes(), method=JPEGDecoderConst.PILLOW)
        except Exception as e:
            raise RuntimeError(f"Error decoding image {image_file}") from e
        image = Image.fromarray(image_arr)
        class_idx = self.ann_images[ann_key]["class_idx"]
        if self.transform:
            image = self.transform(image)
        return {"image": image, "label": class_idx, "idx": idx, "key": ann_key}
