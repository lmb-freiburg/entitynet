import os
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

from packg.typext import PathType

from entitynet.paths import get_entitynet_data_dir


class DomainNetCaptions(Dataset):
    def __init__(
        self,
        domainnet_path: PathType | None = None,
        split: str = "val",
        transform: Optional[Callable] = None,
        exclude_domains=None,
        filter_classes=None,
        mode: str = "label",
    ) -> None:
        if filter_classes is None:
            filter_classes = {}
        if exclude_domains is None:
            exclude_domains = []
        if domainnet_path is None:
            domainnet_path = (
                (get_entitynet_data_dir() / "imagenet1k/imagenet-d").absolute().as_posix()
            )
        else:
            domainnet_path = os.path.abspath(domainnet_path)

        assert split in ["train", "val"]
        split = "test" if split == "val" else split
        assert mode in ["none", "label", "caption", "label+caption"]
        self.return_label = "label" in mode
        self.return_caption = "caption" in mode

        self.samples_per_domain = {
            "clipart": 0,
            "infograph": 0,
            "painting": 0,
            "quickdraw": 0,
            "real": 0,
            "sketch": 0,
        }
        self.samples = []
        for domain in ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]:
            if domain in exclude_domains:
                continue

            with open(os.path.join(domainnet_path, f"{domain}_{split}.tsv")) as f:
                samples = f.readlines()

            samples = [sample.split("\t") for sample in samples]
            samples = [
                (os.path.join(domainnet_path, "visda-2019", path), int(label), caption.strip())
                for path, label, caption in samples
            ]

            # filter out certain classes for certain domains
            if domain in filter_classes:
                samples = [sample for sample in samples if sample[1] not in filter_classes[domain]]

            self.samples_per_domain[domain] = len(samples)
            self.samples.extend(samples)

        self.transform = transform
        self.classes = (Path(domainnet_path) / "classes.txt").read_text().splitlines()
        self.classes = [c.strip().replace("_", " ") for c in self.classes]

    def to_tsv(self, path: str) -> None:
        with open(path, "w") as f:
            f.write("filepath\ttitle\n")
            f.writelines(["\t".join([path, caption]) + "\n" for path, _, caption in self.samples])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple | str:
        path, label, caption = self.samples[index]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)

        ret = {"image": img, "idx": index}
        if self.return_label:
            ret["label"] = label
        if self.return_caption:
            ret["text"] = caption
        return ret

        # # resolve return values
        # sample = (img, label) if self.return_label else (img,)
        # sample += (caption,) if self.return_caption else ()
        # assert len(sample) > 0
        # return sample if len(sample) > 1 else sample[0]
