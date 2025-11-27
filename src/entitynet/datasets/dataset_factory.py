from __future__ import annotations

import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from webdataset.utils import pytorch_worker_info

from packg import Const

from entitynet.config.main_config import Config
from entitynet.config.task_config import BaseTaskCfg, DatasetCfg
from entitynet.datasets.cc12m_full import IndexableCC12M, build_cc12m_webdataset
from entitynet.datasets.coco import MSCocoKarpathy
from entitynet.datasets.converted_vtab_dataset import build_converted_vtab_dataset
from entitynet.datasets.cub import Cub
from entitynet.datasets.domainnet import DomainNetCaptions
from entitynet.datasets.entitynet_indexable import EntityNetUrlIndexable
from entitynet.datasets.entitynet_webds import build_entityneturl_webdataset
from entitynet.datasets.imagenet import Imagenet
from entitynet.datasets.inat21 import iNat21
from entitynet.datasets.inat21_webdataset import build_inat21_webdataset
from entitynet.datasets.rare_species import RARE_SPECIES_LANGUAGES, RareSpecies
from entitynet.datasets.sugarcrepepp_dataset import SugarCrepePPDataset
from entitynet.paths import get_entitynet_data_dir, get_entitynet_output_dir
from entitynet.preprocessor.preprocessor_factory import build_vis_preprocessor_from_config


class DatasetFactoryC(Const):
    ENTITYNETURL = "entityneturl"
    CLIP_BENCHMARK = "clip_benchmark"
    CLIP_BENCHMARK_CONTRASTIVE = "clip_benchmark_contrastive"
    CLIP_BENCHMARK_CONTRASTIVE_MULTITEXT = "clip_benchmark_contrastive_multitext"
    IMAGENET1K = "imagenet1k"
    CONCEPTUALCAPTIONS12M = "conceptualcaptions12m"
    CONCEPTUALCAPTIONS12M_WDS = "conceptualcaptions12m_wds"
    INAT21 = "inat21"
    INAT21_WEBDATSET = "inat21_webdataset"
    RARE_SPECIES = "rare_species"
    DOMAINNET = "domainnet"
    COCO = "coco"
    DATASET_MERGER_INDEXABLE = "dataset_merger_indexable"  # meta class to merge multiple datasets
    CUB = "cub"
    SUGARCREPEPP = "sugarcrepepp"


def build_eval_datasets(config: Config):
    """Build eval datasets given config.eval_tasks"""
    eval_datasets_dict = {}
    eval_loader_dict = {}
    for task_key, task_cfg in config.eval_tasks.items():
        workers = config.trainer.workers
        dataset, loader = build_eval_dataset_for_task(task_key, task_cfg, workers)
        eval_datasets_dict[task_key] = dataset
        eval_loader_dict[task_key] = loader
    return eval_datasets_dict, eval_loader_dict


def build_eval_dataset_for_task(
    task_key: str,
    task_cfg: BaseTaskCfg,
    workers: int = 0,
) -> tuple[Dataset, DataLoader]:
    # create preprocessor
    eval_vis_preproc = build_vis_preprocessor_from_config(task_cfg.vis_preproc)
    logger.debug(f"Created eval vis preprocessor: {eval_vis_preproc}")

    # load dataset
    dataset_cfg = task_cfg.dataset
    logger.debug(f"Loading dataset for task {task_key}: {dataset_cfg}")
    batch_size = task_cfg.dataset.batch_size_eval
    dataset, loader = build_dataset_from_config(
        dataset_cfg,
        transform=eval_vis_preproc,
        batch_size=batch_size,
        workers=workers,
        is_train=False,
    )
    return dataset, loader


def build_dataset_from_config(
    dataset_cfg: DatasetCfg,
    transform: Callable | None = None,
    batch_size: int | None = None,
    workers: int = 0,
    is_train: bool = False,
    seed: int = 42,
    world_size: int = 1,
) -> tuple[Dataset, DataLoader]:
    dataset_factory = dataset_cfg.dataset_factory
    dataset_name = dataset_cfg.dataset_name
    dataset_split = dataset_cfg.dataset_split
    max_datapoints = dataset_cfg.max_datapoints
    max_shards = dataset_cfg.max_shards
    text_aug = dataset_cfg.text_aug
    eval_type = dataset_cfg.eval_type
    deterministic_seed = dataset_cfg.deterministic_seed
    filter_op = dataset_cfg.filter_op
    filter_dict = dataset_cfg.filter_dict
    ds, loader = None, None

    if dataset_factory == DatasetFactoryC.DATASET_MERGER_INDEXABLE:
        assert (
            dataset_cfg.merge_datasets is not None
        ), f"merge_datasets is required for {dataset_factory=}"
        ds = build_merged_dataset(
            dataset_cfg.merge_datasets,
            dataset_cfg.merge_transforms,
            transform,
            is_train,
            seed,
        )

    if dataset_factory == DatasetFactoryC.ENTITYNETURL:
        if dataset_name == "indexable":
            if filter_dict is not None:
                raise NotImplementedError(
                    "Filtering not implemented for indexable dataset, use webdataset or implement."
                )
            ds = EntityNetUrlIndexable(
                dataset_split,
                transform=transform,
                max_datapoints=max_datapoints,
                text_aug=text_aug,
                eval_type=eval_type,
                deterministic_seed=deterministic_seed,
            )
        elif dataset_name == "webdataset":
            ds, loader = build_entityneturl_webdataset(
                dataset_split,
                transform=transform,
                max_shards=max_shards,
                max_datapoints=max_datapoints,
                text_aug=text_aug,
                eval_type=eval_type,
                filter_op=filter_op,
                filter_dict=filter_dict,
                epoch=0,
                is_train=is_train,
                seed=seed,
                batch_size=batch_size,
                workers=workers,
                world_size=world_size,
            )
        else:
            raise ValueError(f"Unknown dataset factory: {dataset_factory}")

    if dataset_factory.startswith(DatasetFactoryC.CLIP_BENCHMARK):
        # note that the clip_benchmark builder will call the function
        # crx/datasets/clip_benchmark_extension.py build_dataset_for_clip_benchmark()
        from clip_benchmark.datasets.builder import build_dataset

        root = get_entitynet_data_dir() / "clip_benchmark" / dataset_name.replace("/", "_")
        if dataset_factory == DatasetFactoryC.CLIP_BENCHMARK_CONTRASTIVE_MULTITEXT:
            label_name = "text_list"
            task = "zeroshot_retrieval"
        elif dataset_factory == DatasetFactoryC.CLIP_BENCHMARK_CONTRASTIVE:
            label_name = "text"
            task = "zeroshot_retrieval"
        elif dataset_factory == DatasetFactoryC.CLIP_BENCHMARK:
            label_name = "label"
            task = "zeroshot_classification"
        else:
            raise ValueError(f"Unknown dataset factory name: {dataset_factory}")

        if dataset_name.startswith("converted_vtab/"):
            ds = build_converted_vtab_dataset(
                root,
                dataset_name,
                dataset_split,
                label_name,
                task,
                transform,
            )
        else:
            clip_benchmark_ds = build_dataset(
                dataset_name,
                root=Path(root).as_posix(),
                split=dataset_split,
                transform=transform,
                task=task,
            )
            ds = ClipBenchmarkWrapper(
                clip_benchmark_ds, label_name=label_name, max_datapoints=max_datapoints, task=task
            )

    if dataset_factory == DatasetFactoryC.CONCEPTUALCAPTIONS12M:
        ds = IndexableCC12M(split=dataset_split, transform=transform, max_datapoints=max_datapoints)
    if dataset_factory == DatasetFactoryC.CONCEPTUALCAPTIONS12M_WDS:
        if dataset_split != "train":
            raise NotImplementedError(
                "Current implementation of webdataset does not work for val/test, only for train. "
                "Use an indexable version of the dataset instead (with tarlookup)."
            )
        ds, loader = build_cc12m_webdataset(
            split=dataset_split,
            transform=transform,
            max_shards=max_shards,
            # max_datapoints=max_datapoints,  # not implemented
            is_train=is_train,
            seed=seed,
            batch_size=batch_size,
            workers=workers,
            world_size=world_size,
        )

    if dataset_factory == DatasetFactoryC.IMAGENET1K:
        ds = Imagenet(
            split=dataset_split,
            name=dataset_name,
            transform=transform,
            return_dict=True,
            max_datapoints=max_datapoints,
        )

    if dataset_factory == DatasetFactoryC.COCO:
        ds = MSCocoKarpathy(
            split=dataset_split,
            name=dataset_name,
            transform=transform,
            return_dict=True,
            max_datapoints=max_datapoints,
        )

    if dataset_factory == DatasetFactoryC.DOMAINNET:
        domainnet_dir = None
        ds = DomainNetCaptions(
            domainnet_path=domainnet_dir, split=dataset_split, transform=transform
        )

    if dataset_factory == DatasetFactoryC.RARE_SPECIES:
        dsetname2lang = {}
        for lang in RARE_SPECIES_LANGUAGES:
            dsetname2lang[f"rare_species_{lang}"] = lang
        if dataset_name not in dsetname2lang:
            raise ValueError(f"Unknown {dataset_name=}, not in {list(dsetname2lang.keys())}")
        language = dsetname2lang[dataset_name]
        ds = RareSpecies(
            split=dataset_split,
            transform=transform,
            return_dict=True,
            max_datapoints=max_datapoints,
            language=language,
        )

    if dataset_factory in [DatasetFactoryC.INAT21, DatasetFactoryC.INAT21_WEBDATSET]:
        if dataset_name == "inat21-latin":
            language = "latin"
        elif dataset_name == "inat21-en":
            language = "en"
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
        kws = dict(
            transform=transform,
            language=language,
        )
        if dataset_factory == DatasetFactoryC.INAT21:
            # jpeg based dataloader
            ds = iNat21(dataset_split, return_dict=True, max_datapoints=max_datapoints, **kws)
        elif dataset_factory == DatasetFactoryC.INAT21_WEBDATSET:
            if not is_train:
                raise ValueError(
                    "Webdataset only works for training, use indexable dataset for eval."
                )
            # webdataset
            ds, loader = build_inat21_webdataset(
                dataset_split,
                **kws,
                is_train=is_train,
                seed=seed,
                batch_size=batch_size,
                workers=workers,
                world_size=world_size,
            )
        else:
            raise ValueError(f"Unknown dataset factory name: {dataset_factory}")
    if dataset_factory == DatasetFactoryC.CUB:
        ds = Cub(
            dataset_split, name=dataset_name, transform=transform, max_datapoints=max_datapoints
        )
    if dataset_factory == DatasetFactoryC.SUGARCREPEPP:
        assert dataset_split == "test", f"Only test split supported for {dataset_factory=}"
        ds = SugarCrepePPDataset(
            dataset_name, transform=transform, return_dict=True, max_datapoints=max_datapoints
        )

    if ds is None:
        raise ValueError(f"Unknown dataset factory name: {dataset_factory}")

    if loader is None:
        if batch_size is None:
            logger.warning(f"batch_size is None, not creating a DataLoader.")
        else:
            loader = DataLoader(
                ds,
                batch_size=batch_size,
                num_workers=workers,
                # worker_init_fn=worker_init_fn_verbose,
                shuffle=is_train,
            )
    return ds, loader


class ClipBenchmarkWrapper(Dataset):
    def __init__(
        self,
        dataset,
        label_name="label",
        max_datapoints: int | None = None,
        task="zeroshot_classification",
    ):
        # print(
        #     f"Init ClipBenchmarkWrapper with dataset {type(dataset).__name__} "
        #     f"worker {pytorch_worker_info()}"
        # )
        self.dataset = dataset
        self.transform = dataset.transform
        self.label_name = label_name

        self.dataset_len = len(dataset)
        if max_datapoints is not None:
            print(f"Limiting dataset to {max_datapoints} datapoints from {len(self.dataset)}")
            self.dataset_len = max_datapoints

        dataset_name = type(dataset).__name__
        if task == "zeroshot_classification":
            self.classes = dataset.classes
            self.templates = dataset.templates
            logger.info(f"Created ClipBenchmark {dataset_name} with classes {self.classes[:5]}...")
        else:
            logger.info(f"Created ClipBenchmark {dataset_name} with task {task}")

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        try:
            image, label = self.dataset[idx]
        except NotImplementedError as e:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support __getitem__ with idx {idx}, it wraps "
                f"{self.dataset=} {self.dataset.__class__.__name__}"
            )
        if self.label_name == "text" and isinstance(label, list):
            # clip benchmark returns list of length 1 for retrievals with only 1 text.
            # but in our scripts we want to get a string when requesting the label "text"
            # if you want a list, change the dataset_factory to clip_benchmark_contrastive_multitext
            # which will lead to label_name "text_list"
            assert len(label) == 1, f"{label=} {type(self)=} {type(self.dataset)=}"
            label = label[0]
        return {
            "image": image,
            self.label_name: label,  # "label" for classification, "text_list" for multi-captions
            "idx": idx,
        }


def worker_init_fn_verbose(worker_id):
    print(f"worker_init_fn with worker_id {worker_id} worker {pytorch_worker_info()}")
    return worker_id


def build_merged_dataset(
    merge_datasets: dict,
    merge_transforms: dict | None,
    transform: Callable | None,
    is_train: bool,
    seed: int,
) -> "MergedDataset":
    """Build a merged dataset from multiple dataset configurations."""
    datasets = []
    for dataset_key, dataset_cfg in merge_datasets.items():
        # Use specific transform if available, otherwise use default
        dataset_transform = transform
        if merge_transforms is not None and dataset_key in merge_transforms:
            dataset_transform_cfg = merge_transforms[dataset_key]
            dataset_transform = build_vis_preprocessor_from_config(dataset_transform_cfg)

        # Create the dataset
        dataset, _ = build_dataset_from_config(
            dataset_cfg,
            transform=dataset_transform,
            is_train=is_train,
            seed=seed,
        )
        logger.debug(
            f"Created dataset {dataset_key} of type {type(dataset).__name__} with {len(dataset)} "
            f"items and transform {dataset_transform}"
        )
        datasets.append(dataset)

    return MergedDataset(datasets)


class MergedDataset(Dataset):
    """A dataset that merges multiple datasets into a single indexable dataset."""

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.lengths = [len(ds) for ds in datasets]
        self.cumulative_lengths = []
        cumsum = 0
        for length in self.lengths:
            cumsum += length
            self.cumulative_lengths.append(cumsum)
        self.total_length = cumsum

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range for dataset of length {self.total_length}")

        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumulative_length in enumerate(self.cumulative_lengths):
            if idx < cumulative_length:
                dataset_idx = i
                break

        # Calculate the index within the specific dataset
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self.cumulative_lengths[dataset_idx - 1]

        return self.datasets[dataset_idx][local_idx]
