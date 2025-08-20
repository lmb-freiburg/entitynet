from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from packg.iotools import load_json

from entitynet.paths import get_entitynet_annotations_dir, get_entitynet_data_dir


class SugarCrepePPDataset(Dataset):
    """
    Dataset source: https://github.com/Sri-Harsha/scpp
    """

    def __init__(
        self, name: str = "replace_att", transform=None, return_dict=True, max_datapoints=None
    ):
        assert return_dict, f"Not implemented {return_dict=}"
        if max_datapoints is not None and max_datapoints > 0:
            raise NotImplementedError(f"{max_datapoints=}")

        json_path = get_entitynet_annotations_dir() / "sugarcrepepp" / f"{name}.json"
        assert json_path.is_file(), f"File {json_path} does not exist"
        json_data = load_json(json_path)
        image_dir = get_entitynet_data_dir() / "coco/images/val2017"
        assert image_dir.is_dir(), f"Directory {image_dir} does not exist"
        self.json_data = json_data
        self.transform = transform
        self.image_dir = image_dir

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        line = self.json_data[idx]
        img_fname = line["filename"]
        ipath = Path(self.image_dir) / img_fname
        image = Image.open(ipath).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "idx": idx,
            "caption": line["caption"],
            "negative_caption": line["negative_caption"],
            "caption2": line["caption2"],
        }
