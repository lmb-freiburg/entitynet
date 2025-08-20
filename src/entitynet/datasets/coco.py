from PIL import Image
from torch.utils.data import Dataset

from packg.iotools import load_json
from packg.log import logger

from entitynet.paths import get_entitynet_data_dir


class MSCocoKarpathy(Dataset):
    def __init__(
        self, split, name: str = "default", transform=None, return_dict=True, max_datapoints=None
    ):
        assert name == "default", f"Only default name is supported, got {name}"
        self.return_dict = return_dict
        self.data_dir = get_entitynet_data_dir() / "coco"
        self.image_dir = self.data_dir / f"karpathy_images/coco_karpathy_{split}"
        self.image_dir_alt = self.data_dir / f"images"

        # load image annotations
        split_file = self.data_dir / f"splits_karpathy/coco_karpathy_{split}.json"
        if not split_file.is_file():
            raise FileNotFoundError(f"File not found: {split_file}")
        ann_data: list[dict] = load_json(split_file)  # [ dict "image" str, "caption" list of str ]
        if max_datapoints is not None and max_datapoints > 0:
            ann_data = ann_data[:max_datapoints]
            logger.warning(f"Reduced dataset to {len(ann_data)} datapoints")

        # exactly 10 of the 5000 images have 6 captions instead of 5. this makes everything more
        # complicated since those datapoints cannot be batched easily, so just fix them.
        for ann in ann_data:
            if len(ann["caption"]) == 6:
                ann["caption"] = ann["caption"][:5]

        self.split = split
        self.ann_data = ann_data
        self.transform = transform

    def __len__(self):
        return len(self.ann_data)

    def __getitem__(self, idx):
        ann = self.ann_data[idx]
        image_file = self.image_dir / ann["image"]
        if not image_file.is_file():
            image_file_alt = self.image_dir_alt / ann["image"]
            if not image_file_alt.is_file():
                raise FileNotFoundError(
                    f"Image file not found in both possible locations: "
                    f"{image_file} or {image_file_alt}"
                )
            image_file = image_file_alt
        image = Image.open(image_file).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {"image": image, "idx": idx, "text_list": ann["caption"]}
