""" """

import math
from functools import partial

import webdataset as wds
from loguru import logger
from torchvision.transforms import transforms
from webdataset import DataPipeline, WebLoader

from typedparser.objects import repr_value
from visiontext.distutils import get_torch_worker_id, print_with_rank
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
)

from entitynet.datasets.inat21 import iNat21TextLoader
from entitynet.paths import get_entitynet_data_dir

split_info = {
    "trainnodev": {
        "size": 2586843,
        "num_shards": 256,
    },
    "traindev": {
        "size": 100000,
        "num_shards": 32,
    },
}


def build_inat21_webdataset(
    split: str,
    transform=None,
    language="en",
    epoch=0,
    is_train=True,
    seed=42,
    batch_size=64,
    workers=0,
    world_size=1,
):
    floor = False
    resampled = False

    # get tar files and number of samples
    base_dir = get_entitynet_data_dir() / "iNat/2021"
    if split not in split_info:
        raise ValueError(
            f"webdataset not implemented for split {split} in iNat21, "
            f"available splits: {split_info.keys()}"
        )
    split_info_here = split_info[split]
    num_shards = split_info_here["num_shards"]
    tar_dir = base_dir / f"tar_{split}"
    tar_files = [a.as_posix() for a in tar_dir.glob("*.tar")]
    assert len(tar_files) == num_shards, (
        f"Expected {num_shards} shards but got {len(tar_files)} for split "
        f"{split} in iNat21 directory {tar_dir}"
    )
    num_samples = split_info_here["size"]

    logger.info(f"WebDS {epoch=} {num_samples=} {num_shards=} first {tar_files[0]}")
    shared_epoch = SharedEpoch(epoch=epoch)  # create shared epoch store to sync epoch to workers

    pipeline = [wds.SimpleShardList(tar_files)]
    if is_train:
        pipeline += [
            detshuffle2(
                bufsize=SHARD_SHUFFLE_SIZE,
                initial=SHARD_SHUFFLE_INITIAL,
                seed=seed,
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
            # at this point each worker on each node should have a different list of tars
            wds.tarfile_to_samples(handler=log_and_continue),  # tarfile_to_samples_nothrow
            wds_detshuffle_datapoints(
                bufsize=SAMPLE_SHUFFLE_SIZE,
                initial=SAMPLE_SHUFFLE_INITIAL,
                seed=seed,
                epoch=shared_epoch,
            ),
        ]
    else:
        pipeline += [
            wds.split_by_node,
            wds.split_by_worker,
            wds.tarfile_to_samples(handler=log_and_continue),
        ]

    # here we use the same text loader as the jpeg based dataset
    text_loader = iNat21TextLoader(language=language)

    pipeline += [
        # wds.decode auto-decodes the image but also json bytes to dict and other formats
        wds.decode("pilrgb", handler=log_and_continue),
        wds_filter_load_metadata_inat21(text_loader),
        wds.rename(image="jpg;png;jpeg;webp", text="text", label="label", idx="__key__"),
        wds.map_dict(image=transform, idx=int),
    ]

    # create dictionary batches (instead of the tuples usually used in webdataset)
    # label is the class number, idx is the same datapoint number
    keys = ("image", "text", "label", "idx")
    coll_fn = partial(dict_collation_fn, keys=keys)
    pipeline += [
        wds.to_tuple(*keys),
        wds.batched(batch_size, partial=not is_train, collation_fn=coll_fn),
        # wds_print_content(),
    ]
    dataset = DataPipeline(pipeline)

    print_with_rank(f"World size: {world_size}")
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

    # # this special handling for validation does not work, it times out the dataloader
    # else:
    #     last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / batch_size)

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

    # finally in order to do classification with this dataset we need the classes
    dataset.classes = text_loader.classes

    return dataset, dataloader


class wds_filter_load_metadata_inat21(wds.PipelineStage):
    def __init__(self, text_loader: iNat21TextLoader):
        self.text_loader = text_loader
        self.json_key = "json"

    def run(self, src):
        for data in src:
            # at this point, wds.decode has decoded the json bytestring and it is already a dict
            json_data = data.pop(self.json_key)
            cat_id = json_data["category_id"]
            data["label"] = cat_id
            data["text"] = self.text_loader.get_text_for_category_id(cat_id)
            yield data


def main():
    for split in ("traindev", "trainnodev"):
        dataset, dataloader = build_inat21_webdataset(
            split=split,
            transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
            language="en",
            epoch=0,
            is_train=True,
            seed=42,
            batch_size=64,
            workers=0,
        )
        print(f"{split=} {len(dataset)=}")
        print()

        # note that in this webdataset, the dataset is already batched and there is no easy
        # way to get "just" a single datapoint. this is for performance reasons
        print(f"********************************************************************* dataset")
        for d in dataset:
            print(repr_value(d))
            break
        print()

        print(f"********************************************************************* dataloader")
        for d in dataloader:
            print(repr_value(d))
            break
        break


if __name__ == "__main__":
    main()
