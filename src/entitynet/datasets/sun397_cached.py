"""
SUN397, extended cache the paths so it doesn't run rglob on 100k files on each init.
"""

from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import PIL.Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive

from packg.iotools import dump_json, load_json


class SUN397CachedPaths(VisionDataset):
    """`The SUN397 Data Set <https://vision.princeton.edu/projects/2010/SUN/>`_.

    The SUN397 or Scene UNderstanding (SUN) is a dataset for scene recognition consisting of
    397 categories with 108'754 images.

    Args:
        root (str or ``pathlib.Path``): Root directory of the dataset.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed
            version. E.g, ``transforms.RandomCrop``.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    _DATASET_URL = "http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz"
    _DATASET_MD5 = "8ca2778205c41d23104230ba66911c7a"

    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self._data_dir = (Path(self.root) / "SUN397").resolve().absolute()

        if download:
            self._download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        with open(self._data_dir / "ClassName.txt") as f:
            self.classes = [c[3:].strip() for c in f]

        self.class_to_idx = dict(zip(self.classes, range(len(self.classes))))
        cache_file = self._data_dir / f"filelist_cache.json"
        if not cache_file.is_file():
            raise FileNotFoundError(f"{cache_file} not found. TODO cache generator")
        # if cache_file.is_file():
        _image_files_rel = load_json(cache_file)
        # else:
        #     _image_files_rel = [
        #         a.relative_to(self._data_dir).as_posix() for a in self._data_dir.rglob("sun_*.jpg")
        #     ]
        #     dump_json(_image_files_rel, cache_file, indent=2, custom_format=False)
        self._image_files = [self._data_dir / a for a in _image_files_rel]
        expected_files = 108754
        assert len(self._image_files) == expected_files, (
            f"{len(self._image_files)=} incorrect, should be {expected_files}."
            f" {self._data_dir=}"
        )

        self._labels = [
            self.class_to_idx["/".join(path.relative_to(self._data_dir).parts[1:-1])]
            for path in self._image_files
        ]

    def __len__(self) -> int:
        return len(self._image_files)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        image_file, label = self._image_files[idx], self._labels[idx]
        image = PIL.Image.open(image_file.as_posix()).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def _check_exists(self) -> bool:
        return self._data_dir.is_dir()

    def _download(self) -> None:
        if self._check_exists():
            return
        download_and_extract_archive(
            self._DATASET_URL, download_root=self.root, md5=self._DATASET_MD5
        )
