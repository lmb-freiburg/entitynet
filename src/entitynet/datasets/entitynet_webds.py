"""
webdataset for training of entitynet
"""

import math
from functools import partial
from pathlib import Path
from typing import Any

import webdataset as wds
from webdataset import DataPipeline, WebLoader, reraise_exception

from entitynet.datasets.entitynet import N_SHARDS
from entitynet.datasets.entitynet_textloader import EntityNetUrlTextLoader
from entitynet.datasets.entitynet_webds_queryfilter import EntityNetQueryFilter
from entitynet.paths import get_entitynet_data_dir
from packg.iotools import load_json
from packg.log import logger
from packg.typext import PathType
from typedparser.objects import print_datapoint
from visiontext.distutils import get_torch_worker_id, print_with_rank
from visiontext.webdataset_pipeline import (
    SAMPLE_SHUFFLE_INITIAL,
    SAMPLE_SHUFFLE_SIZE,
    SHARD_SHUFFLE_INITIAL,
    SHARD_SHUFFLE_SIZE,
    SharedEpoch,
    detshuffle2,
    dict_collation_fn,
    wds_detshuffle_datapoints,
)
from visiontext.webdataset_pipeline_separate_metadata import (
    wds_decode_with_metadata,
    wds_pipeline_expand_tarfile_and_json_to_samples,
    wds_tar_file_iterator_with_metadata,
)

from entitynet.config.task_config import EntityNetTextAugCfg, EntityNetTextReturnMode
from entitynet.preprocessor.preprocessor_factory import get_simple_transform_to_tensor


def build_entityneturl_webdataset(
    # dataset config
    split: str,
    transform=None,
    max_shards: int | None = None,
    max_datapoints: int | None = None,
    text_aug: EntityNetTextAugCfg | None = None,
    eval_type: str = "default",
    filter_op: str | None = None,
    filter_dict: dict[str, Any] | None = None,
    # wds config
    epoch=0,
    is_train=True,
    seed=42,
    batch_size=64,
    workers=0,
    world_size=1,
):
    assert seed is not None, f"seed must be set for webdataset but got {seed=}"
    resampled = False
    if max_datapoints is not None and max_datapoints > 0:
        raise ValueError("max_datapoints not implemented for webdataset, use max_shards instead")
    assert eval_type == "default", f"Unknown {eval_type=}"

    # get tar files and number of samples
    base_dir = get_entitynet_data_dir() / "entitynet"
    if split not in N_SHARDS:
        raise ValueError(
            f"webdataset not implemented for split {split} in iNat21, "
            f"available splits: {N_SHARDS.keys()}"
        )
    num_shards = N_SHARDS[split]
    tar_dir = base_dir / f"data_{split}"
    tar_files = sorted([a.as_posix() for a in tar_dir.glob("*.tar")])
    if len(tar_files) == 0:
        raise FileNotFoundError(f"No tar files found in {tar_dir}")

    num_images_per_shard = load_json(base_dir / f"num_images_per_shard-wds_{split}.json")
    assert len(tar_files) == num_shards, f"{len(tar_files)=} != {num_shards=}"
    assert len(num_images_per_shard) == num_shards, f"{len(num_images_per_shard)=} != {num_shards=}"
    num_samples = sum(num_images_per_shard)

    # use max_shards argument to reduce dataset size
    if max_shards is not None and max_shards > 0:
        tar_files = tar_files[:max_shards]
        num_shards = len(tar_files)
        num_samples = sum(num_images_per_shard[:num_shards])
        logger.warning(f"Reducing dataset to {num_shards=} {num_samples=} due to {max_shards=}")
    else:
        max_shards = num_shards

    logger.info(f"WebDS {epoch=} {num_samples=} {num_shards=} first {tar_files[0]}")
    shared_epoch = SharedEpoch(epoch=epoch)  # create shared epoch store to sync epoch to workers

    # here we use the same text loader as the jpeg based dataset
    text_loader = EntityNetUrlTextLoader(base_dir=base_dir, text_aug=text_aug)
    # pprint(asdict(text_aug))

    select_files = None
    if filter_dict is not None:
        query_filter = EntityNetQueryFilter(filter_op, filter_dict, text_loader.queries)
        filtering_result = query_filter.load_precomputed_filtering_results(split)
        select_files = query_filter.filter
        # combine filtering and limiting the shards, to get the total num samples.
        n_keys_per_shard = filtering_result["n_keys_per_shard"]
        shard_indices = list(range(num_shards))
        num_samples = sum(n_keys_per_shard[str(si)] for si in shard_indices)  # str due to json
        assert num_samples > 0, f"{num_samples=} {filtering_result=} {filter_dict=} {max_shards=}"
        # add the query allow info to the text loader, so we only load knowledge graph info
        # for the allowed queries
        text_loader.query2allowflag = query_filter.query2allowflag
        logger.info(f"Filtered: {filter_dict=} {filter_op=} Result {num_samples=} {num_shards=}")

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
            # wds.tarfile_to_samples(handler=log_and_continue),  # tarfile_to_samples_nothrow
            wds_pipeline_expand_tarfile_and_json_to_samples(
                handler=reraise_exception,
                select_files=select_files,
                tar_file_iter=wds_tar_file_iterator_with_metadata,
                add_metadata_to_iter=True,
                metadata_filename="metadata_{:05d}.json",  # shard_00000.tar -> metadata_00000.json
            ),
            wds_detshuffle_datapoints(
                bufsize=SAMPLE_SHUFFLE_SIZE,
                initial=SAMPLE_SHUFFLE_INITIAL,
                seed=seed,
                epoch=shared_epoch,
            ),
        ]
    else:
        raise NotImplementedError(
            f"Validation currently will not work due to problems with lightning. It is also not "
            f"necessary, use indexable dataset for validation."
        )
        # pipeline += [
        #     wds.split_by_node,
        #     wds.split_by_worker,
        #     wds.tarfile_to_samples(handler=log_and_continue),
        # ]

    pipeline += [
        # wds.decode auto-decodes the image but also json bytes to dict and other formats
        wds_decode_with_metadata("pilrgb", handler=reraise_exception),
        wds_pipeline_add_metadata_to_datapoint_v10(text_loader),
        wds.rename(image="jpg;png;jpeg;webp", key="__key__"),  # text="text", label="label",
    ]
    text_loader_keys = text_loader.get_keys_for_webdataset()
    if transform is not None:
        pipeline += [
            wds.map_dict(image=transform),
        ]

    # create dictionary batches (instead of the tuples usually used in webdataset)
    keys = ["image", "key"] + list(text_loader_keys)
    coll_fn = partial(dict_collation_fn, keys=keys)
    pipeline += [
        wds.to_tuple(*keys),
        wds.batched(batch_size, partial=not is_train, collation_fn=coll_fn),
        # wds_print_content(),
    ]
    dataset = DataPipeline(pipeline)
    dataset.text_loader = text_loader  # to keep the reference  # type: ignore

    print_with_rank(f"World size: {world_size}")
    if not resampled:
        assert num_shards >= workers * world_size, f"{num_shards=} < {workers=} * {world_size=}"

    # roll over and repeat a few samples to get same number of full batches on each node
    floor = False
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
    print_with_rank(f"{batch_size=} {world_size=} {global_batch_size=} {num_worker_batches=}")
    print_with_rank(f"Dataloader length {len(dataloader)} dataset length {len(dataset)}")
    dataset.shared_epoch = shared_epoch

    return dataset, dataloader


class wds_pipeline_add_metadata_to_datapoint_v10(wds.PipelineStage):
    def __init__(self, text_loader: EntityNetUrlTextLoader):
        self.text_loader = text_loader
        self.json_key = "json"

    def run(self, src):
        for data in src:
            # data contains keys __key__, __url__, jpg, metadata
            # verify metadata integrity
            key = data["__key__"]
            # qh, num = key.split("/")
            metadata: dict = data.pop("metadata")
            _, _, image_filename = metadata["img"].split("/")
            assert key == image_filename.split(".")[0], f"{key=} {image_filename=} {metadata=}"
            data_update = self.text_loader.get_data_for_dataloader(key, metadata)
            data.update(data_update)
            yield data


def main():
    simple_transform = get_simple_transform_to_tensor(224)

    # text_aug = TextAugCfg(
    #     replace_noun_synonym_chance=0.5,
    #     replace_noun_definition_chance=0.2,
    #     replace_noun_hierarchy_chance=0.2,
    #     replace_attr_query=0.4,
    #     alt_text_chance=0.5,
    #     min_sitelinks=40,
    #     hierarchy_balancing_strength=0.5,
    #     alt_text_min_vote=2.5,
    # )
    # for split in N_SHARDS.keys():
    #     if split == "train":
    #         print(f"skip train")
    #         continue
    #     print(f"---------- dataset {split} with random text")
    #     ds, loader = build_entitynetv10_webdataset(
    #         split,
    #         transform=simple_transform,
    #         text_aug=text_aug,
    #         batch_size=4,
    #     )
    #     for i, dp in enumerate(ds):
    #         if i >= 1:
    #             break
    #         print_datapoint(dp)
    #         print()
    #     print(f"---------- dataloader {split} with random text")
    #     for i, batch in enumerate(loader):
    #         if i >= 1:
    #             break
    #         print_datapoint(batch)
    #     print()

    print(f"---------- dataset val with all texts")
    text_aug = EntityNetTextAugCfg(
        n_texts_per_image=8,
        replace_noun_synonym_chance=0.6,
        replace_noun_definition_chance=0.1,
        replace_noun_hierarchy_chance=0.2,
        replace_noun_hierarchy_chance_living=0.1,
        alt_text_chance=0.5,
        attronly_keep_query=1.0,
        attronly_replace_query_with_synonym=0.2,
        attronly_build_pseudo_query=0.2,
        attronly_strgf_replace_entity=0.2,
        attronly_attribute_only=0.2,
        attronly_replace_with_definition=0.05,
        attrnoun_keep_query=1.0,
        attrnoun_replace_query_with_synonym=0.2,
        attrnoun_build_pseudo_query=0.2,
        attrnoun_strgf_replace_entity=0.2,
        attrnoun_attribute_only=0.2,
        attrnoun_replace_with_definition=0.0,
    )
    split = "test"
    ds, loader = build_entityneturl_webdataset(
        split,
        transform=simple_transform,
        text_aug=text_aug,
        batch_size=4,
    )
    for i, dp in enumerate(ds):
        if i >= 3:
            break
        # print(dp["search_query"])
        print_datapoint(dp)
        print()
    print(f"---------- dataloader val with all texts")
    for i, batch in enumerate(loader):
        if i >= 1:
            break
        print_datapoint(batch)
    print(f"done all texts")
    print()
    print(f"done")


if __name__ == "__main__":
    main()
