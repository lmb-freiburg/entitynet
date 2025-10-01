"""
Build entitynet from URLS, step 2: download images

Optionally delete all .alpha.jpg and .error files afterwards.

The URL data looks as follows.
- "qh": N_query links to queries (all queries where this image was found under),
- "texts": N_texts unique alt texts that were found for this image,
- "images_*". N_images links, links to the html, and sizes for found duplicates of this image.

example_datapoint = {
    "key": "val000000000",
    "qh": ["rwOlC205TN2HUc49vACOMhs57tgaNibvgFEyrg", "WkFTqLvZ3oWABOlIDGUed5ApJEOkQevDz_dl-w"],
    "texts": [
        "Heteropteryx dilatata, ...",
        "Heteropteryx dilatata, (text 2)",
        "Heteropteryx dilatata, (text 3)",
    ],
    "images_link": [
        "http://.......com/-9fdb-8478afe93598.png?v=1680110385",
        "https://.......com/c4ae6-9fdb-8478afe93598.png?v=1680110385",
    ],
    "images_contextlink": [
        "https://......com/weird-giant-bug",
        "https://......com/weird-giant-bug",
    ],
    "images_width": [327, 327],
    "images_height": [540, 540],
}


"""

from collections import defaultdict
import os
from pathlib import Path
from loguru import logger
from entitynet.datasets.entitynet import EntityNetUrlBuildArgs, parse_args_for_url_build
from packg.misc import uncollate
from packg.strings.formatters import dict_to_str_comma_equals
from packg.tqdmext import tqdm_max_ncols
from packg.typext import PathType
from typedparser import TypedParser
from torch.utils.data import Dataset
import pandas as pd
from visiontext.image_downloader import download_image_with_retry_only_once, random_sleep
from torch.utils.data import DataLoader

N_RETRIES = 3
SLEEP_SEC = 1


def main():
    parser = TypedParser.create_parser(EntityNetUrlBuildArgs, description=__doc__)
    args: EntityNetUrlBuildArgs = parser.parse_args()
    logger.info(f"{args}")
    entitynet_dir, split_list = parse_args_for_url_build(args)
    metadata_dir = entitynet_dir / "images_metadata"

    # download all URLs for the given splits
    for split in split_list:
        parquets = sorted(list(metadata_dir.glob(f"{split}-*.parquet")))
        assert len(parquets) > 0, f"No parquets found for {split=} in {metadata_dir}"
        for parquet in parquets:
            logger.info(f"Process {split=} parquet {parquet.name}")
            df = pd.read_parquet(parquet)
            parquet_stem = parquet.stem
            download_dir = entitynet_dir / "images_raw" / parquet_stem
            os.makedirs(download_dir, exist_ok=True)
            key2urls = dict(zip(df["key"].tolist(), df["images_link"].tolist()))
            logger.info(f"Downloading {len(key2urls)} images to {download_dir}")
            ds = EntityNetUrlBuildStep1Multiprocessor(download_dir, key2urls)
            dl = DataLoader(ds, batch_size=1, num_workers=args.workers, collate_fn=uncollate)
            pbar = tqdm_max_ncols(desc=f"Downloading {parquet_stem}", total=len(ds))
            success_counter = defaultdict(int)
            for i, (key, out_file, success_str) in enumerate(dl):
                success_counter[success_str] += 1
                pbar.set_description(f"{dict_to_str_comma_equals(success_counter)}", refresh=False)
                pbar.update()
            pbar.close()
            logger.info(f"{parquet_stem}: {dict_to_str_comma_equals(success_counter)}")


class EntityNetUrlBuildStep1Multiprocessor(Dataset):
    def __init__(self, download_dir: Path, key2urls: dict[str, list[str]]):
        self.download_dir = download_dir
        self.key2urls = key2urls
        self.keys = list(key2urls.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        urls = self.key2urls[key]
        return _download_key(key, urls, self.download_dir)


def _download_key(key, urls, download_dir: Path):
    out_file = download_dir / f"{key}.jpg"
    success = False
    for url in urls:
        target_rgb_img, _target_alpha_img, _exif, _err = download_image_with_retry_only_once(
            url, out_file, retries=N_RETRIES, sleep=SLEEP_SEC, verbose=False
        )
        if target_rgb_img is not None:
            assert out_file.is_file(), f"Downloaded image but file {out_file} not found"
            success = True
            break
        random_sleep(SLEEP_SEC)

    success_str = "success" if success else "failed"
    return key, out_file, success_str


if __name__ == "__main__":
    main()
