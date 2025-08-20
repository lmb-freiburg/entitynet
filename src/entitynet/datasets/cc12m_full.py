"""
ConceptualCaptions12M dataset

TODO add instructions and scripts to build CC12M from scratch

Requirements: sqlite (for indexing the tar files), python>=3.10, pytorch>=2,

conda install sqlite -y
pip install torch torchvision torchaudio
pip install typedparser packg visiontext webdataset

Note that the first 3 packages are public utility code from here https://github.com/simon-ging/

Before cleaning and splitting:
    tars contain 28,119,738 files: 9,373,246 images and each has
    - jpg (the image),
    - txt (the utf-8 caption only),
    - json with fields caption, url, key, status, error_message,
      width, height, original_width, original_height, exif, sha256

Finally:
    Deduplication with downstream datasets:
        Total banned keys: 70,221
        Total source keys: 9,373,246
        Clean source keys: 9,303,025
    Result: Train: 9,263,025, Val: 20,000, Test: 20,000
    Tars only contain jpg (the image) and json with fields:
        caption, url, key width, height, exif, sha256. other fields were redudant / not interesting

Deduplication within itself:
    dups=1,068,635 matches=530,358,536
    many false positives (e.g. same actor with black shirt but different print on the shirt)
    many "dead" images (white screen only image, or "image missing" symbols)

Deduplicated downstream datasets:
{
    "train": {
        "inat17_train": 579184,
        "inat18_train": 437513,
        "inat19_train": 265213,
        "inat21_train": 2686843,
        "inat21_train_mini": 500000,
        "semi-inat20_l_train": 3959,
        "semi-inat20_u_train_in": 26640,
        "semi-inat20_u_train_out": 122208,
        "semi-inat21_l_train": 9721,
        "semi-inat21_u_train": 313248,
        "imagenet1k_train": 1281167,
        "swig_train": 75702,
        "coco2014_train": 82783,
        "coco2017_train": 118287,
        "coco-karpathy_train": 113287,
    },
    "test": {
        "waterbird_all": 11788,
        "inat17_val": 95986,
        "inat17_test": 182707,
        "inat18_val": 24426,
        "inat18_test": 149394,
        "inat19_val": 3030,
        "inat19_test": 35350,
        "inat21_val": 100000,
        "inat21_public_test": 500000,
        "semi-inat20_test": 8000,
        "semi-inat20_val": 2000,
        "semi-inat21_test": 16200,
        "semi-inat21_val": 4050,
        "oxford-flower-102": 8189,
        "oxford-pet": 7390,
        "dtd-textures": 5640,
        "imagenet1k_val": 50000,
        "imagenet1k_test": 100000,
        "imagenet-a_all": 7450,
        "imagenet-o_all": 2000,
        "imagenet-r_all": 30000,
        "imagenetv2_all": 30000,
        "imagenet-sketch_all": 50889,
        "imagenet-9_all": 32400,
        "imagenet-lt_all": 18000,
        "objectnet_all": 50273,
        "imagenet-d_all": 914998,
        "swig_dev": 25200,
        "swig_test": 25200,
        "coco2014_val": 40504,
        "coco2014_test": 81434,
        "coco2017_val": 5000,
        "coco2017_test": 40670,
        "vqa-abstract-v002_all": 81325,
        "coco-karpathy_val": 5000,
        "coco-karpathy_test": 5000,
        "visualgenome_all": 108079,
        "winoground_all": 800,
        "flickr30k_all": 31783,
        "flickr8k_all": 8091,
    },
}
"""

import math
from functools import partial
from pathlib import Path

import webdataset as wds
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import transforms
from webdataset import DataPipeline, WebLoader

from packg.iotools import load_json
from typedparser.objects import repr_value
from visiontext.distutils import get_torch_worker_id, print_with_rank
from visiontext.images import JPEGDecoderConst, decode_jpeg
from visiontext.iotools.tar_lookup import TarLookup
from visiontext.webdataset_pipeline import (
    SAMPLE_SHUFFLE_INITIAL,
    SAMPLE_SHUFFLE_SIZE,
    SHARD_SHUFFLE_INITIAL,
    SHARD_SHUFFLE_SIZE,
    SharedEpoch,
    detshuffle2,
    dict_collation_fn,
    log_and_continue,
    wds_detshuffle_datapoints,
    wds_filter_unpack_json,
    wds_print_content,
)

from entitynet.paths import get_entitynet_data_dir


def main():
    test_indexable()
    test_webdataset()
    print()
    print(f"All tests successful.")


def test_indexable():
    # test the indexable version of web dataset
    dataset = IndexableCC12M(
        split="train",
        transform=transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()]),
    )
    loader = DataLoader(dataset, batch_size=32, num_workers=4)
    for batch in loader:
        print(batch)
        break


def test_webdataset():
    dummy_transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
    # test_indexable()
    ds, dl = build_cc12m_webdataset(transform=dummy_transform, workers=4)
    print(ds, dl)
    for item in ds:
        print(repr_value(item))
        # for key, value in item.items():
        #     print(f"{key}: {repr_value(value)}")
        break
    for batch in dl:
        print(repr_value(batch))
        # print(batch.keys())
        break


def get_cc12m_clean(split, base_data_dir=None):
    if base_data_dir is not None:
        base_data_dir = Path(base_data_dir)
    else:
        base_data_dir = get_entitynet_data_dir()
    base_dir = base_data_dir / "ConceptualCaptions12M/clean"

    glob_str = f"tar_{split}/*.tar"
    tar_files = sorted(list(base_dir.glob(glob_str)))
    tar_files = [f.as_posix() for f in tar_files]
    if len(tar_files) == 0:
        raise FileNotFoundError(f"No tar files found in  {base_dir} / {glob_str}")
    return base_dir, tar_files


CC12M_SPLIT_SIZES = {"train": 9263025, "val": 20000, "test": 20000}


def build_cc12m_webdataset(
    base_data_dir=None,
    split="train",
    transform=None,
    max_shards: int | None = None,
    max_datapoints: int | None = None,
    # wds config
    epoch=0,
    is_train=True,
    seed=42,
    batch_size=64,
    workers=0,
    world_size=1,
):
    floor = False
    resampled = False

    if max_datapoints is not None and max_datapoints > 0:
        raise ValueError("max_datapoints not implemented for webdataset, use max_shards instead")
    # get tar files and number of samples
    base_dir_cc, tar_files = get_cc12m_clean(split, base_data_dir=base_data_dir)
    if len(tar_files) == 0:
        raise FileNotFoundError(f"No tar files found for CC12M in {base_dir_cc}")
    num_shards = len(tar_files)
    num_samples = CC12M_SPLIT_SIZES[split]

    # use max_shards argument to reduce dataset size
    # for this, we need to know how many datapoints to recalculate num_samples
    if max_shards is not None and max_shards > 0:
        tar_files = tar_files[:max_shards]
        num_shards = len(tar_files)
        num_images_per_shard_dict = load_json(
            base_dir_cc / f"num_images_per_shard-wds_{split}.json"
        )
        tar_files_rel = [f"{Path(tf).parent.name}/{Path(tf).name}" for tf in tar_files]
        num_images_per_shard = [num_images_per_shard_dict[tf] for tf in tar_files_rel]
        num_samples = sum(num_images_per_shard[:num_shards])
        logger.warning(f"Artificially reducing {num_shards=} due to {max_shards=}")

    logger.info(f"WebDS {epoch=} {num_samples=} {num_shards=} first {tar_files[0]}")
    shared_epoch = SharedEpoch(epoch=epoch)  # create shared epoch store to sync epoch to workers

    pipeline = [wds.SimpleShardList(tar_files)]
    if is_train:
        pipeline += [
            # print_tar_indices(extra_text=f"Before shuffling"),
            detshuffle2(
                bufsize=SHARD_SHUFFLE_SIZE,
                initial=SHARD_SHUFFLE_INITIAL,
                seed=seed,
                epoch=shared_epoch,
            ),
            # print_tar_indices(extra_text=f"After shuffling, before splitting"),
            wds.split_by_node,
            wds.split_by_worker,
            # print_tar_indices(extra_text=f"After splitting"),
            #
            # at this point each worker on each node should have a different list of tars
            wds.tarfile_to_samples(handler=log_and_continue),  # tarfile_to_samples_nothrow
            # print_sample_keys(extra_text="Before shuffling"),
            wds_detshuffle_datapoints(
                bufsize=SAMPLE_SHUFFLE_SIZE,
                initial=SAMPLE_SHUFFLE_INITIAL,
                seed=seed,
                epoch=shared_epoch,
            ),
            # wds_print_sample_keys(extra_text="After shuffling"),
            # wds_print_content(), at this point the json is still a bytestring
        ]
    else:
        pipeline += [
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ]

    # # we don't need the "no_image" filter at this point because we rebuilt a valid dataset
    # # the "no_caption" filter doesn't work yet because we first need to expand the json
    # wds.select(filter_no_caption_or_no_image)

    # TODO libturbojpeg decoding would be faster than pillow

    # example tokenizing: wds.map_dict(image=transform, text=lambda text: tokenizer(text)[0]),

    pipeline += [
        # wds.decode auto-decodes the image but also json bytes to dict and other formats
        wds.decode("pilrgb", handler=log_and_continue),
        wds_filter_unpack_json("json", ("caption",)),
        wds_print_content(),
        wds.rename(image="jpg;png;jpeg;webp", text="caption", key="__key__"),
        wds.map_dict(image=transform),
        # change: keep everything in dict space
        wds_print_content(),
        # dict_keys(['__key__', '__url__', 'json', 'image', 'text'])
        # __key__ is e.g. 000000064
    ]

    # create dictionary batches instead of tuples
    keys = ("image", "text", "key")
    coll_fn = partial(dict_collation_fn, keys=keys)
    pipeline += [
        wds.to_tuple(*keys),
        wds.batched(batch_size, partial=not is_train, collation_fn=coll_fn),
        wds_print_content(),
    ]
    dataset = DataPipeline(pipeline)

    if is_train:
        print_with_rank(f"(expected) world size: {world_size}")
        if not resampled:
            assert num_shards >= workers * world_size, "number of shards must be >= total workers"

        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)

    dataloader = WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=workers,
        persistent_workers=workers > 0,
    )

    dataset.num_samples = num_samples
    dataset.transform = transform
    dataset = dataset.with_length(num_samples, silent=True)

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples
    dataloader = dataloader.with_length(num_batches, silent=True)

    print_with_rank(
        f"Worker {get_torch_worker_id()} built webds with {num_samples=}, {num_batches=}"
    )
    print_with_rank(f"Dataloader length {len(dataloader)} dataset length {len(dataset)}")
    dataset.shared_epoch = shared_epoch
    return dataset, dataloader


class IndexableCC12M(Dataset):
    def __init__(self, base_data_dir=None, split="train", transform=None, max_datapoints=None):
        worker_id = get_torch_worker_id()
        base_dir, tar_files = get_cc12m_clean(split, base_data_dir=base_data_dir)
        lookup = TarLookup(
            base_dir,
            tar_files,
            base_dir / f"db-tar_{split}.sqlite",
            verbose=False,
            worker_id=worker_id,
        )
        keys_to_captions_file = base_dir / f"captions_{split}.json"
        print_with_rank(
            f"Worker {worker_id} loading {keys_to_captions_file} of size "
            f"{keys_to_captions_file.stat().st_size / 1024 ** 3:.2f} GB"
        )
        keys_to_caption = load_json(keys_to_captions_file)
        keys = list(keys_to_caption.keys())

        if max_datapoints is not None:
            keys = keys[:max_datapoints]
            logger.warning(f"Restricting dataset to {max_datapoints} datapoints")

        print(
            f"Got {len(keys_to_caption):,d} keys to captions total and {len(keys):,d} "
            f"datapoints for split {split}"
        )
        self.transform = transform
        self.tar_files = tar_files
        self.lookup = lookup
        self.keys_to_caption = keys_to_caption
        self.keys = keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, item):
        key = self.keys[item]
        caption = self.keys_to_caption[key]
        filename, tarfilename, image_bytes = self.lookup.get_content_from_filename(f"{key}.jpg")
        try:
            image_pil = decode_jpeg(image_bytes, method=JPEGDecoderConst.PILLOW_IMAGE)
        except Exception as e:
            raise RuntimeError(f"Error decoding image {filename} from tar {tarfilename}") from e
        if self.transform is not None:
            image_pil = self.transform(image_pil)
        return {
            "image": image_pil,
            "text": caption,
            "idx": item,
            "key": key,
        }


if __name__ == "__main__":
    main()
