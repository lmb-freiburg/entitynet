"""
Logic to consolidate results from running evaluation on multiple GPUs.
"""

import os
from pathlib import Path
from typing import Any, Callable, Union

import lightning as lit
import torch
from loguru import logger
from torch import distributed as dist

from packg.iotools.jsonext import dump_json, load_json
from packg.typext import PathType
from typedparser.objects import repr_value
from visiontext.distutils import WorldInfo


def _no_print(*args, **kwargs):  # type: ignore  # noqa: F821  # pylint: disable=unused-argument
    pass


def consolidate_outputs(list_of_dict_of_outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Concatenates a list of dictionaries with identical keys into a single dictionary.
    Handles torch.Tensor, list, and scalar (int/float/str) values appropriately.
    Returns empty dict if input list is empty.
    """
    if len(list_of_dict_of_outputs) == 0:
        return {}
    fields = list(list_of_dict_of_outputs[0].keys())
    dict_of_merged_outputs = {k: [] for k in fields}
    for batch_dict in list_of_dict_of_outputs:
        for field, value in batch_dict.items():
            dict_of_merged_outputs[field].append(value)
    for field in fields:
        first_item = dict_of_merged_outputs[field][
            0
        ]  # for example, when filed == image_features, first item should be tensor of shape (batch_size, emb_dim) in case of contrastive3d

        if isinstance(first_item, torch.Tensor):
            # logger.info(f"{len(dict_of_merged_outputs[field])=}")
            try:
                dict_of_merged_outputs[field] = torch.cat(dict_of_merged_outputs[field], dim=0)
            except RuntimeError as e:
                logger.error(f"cat dim 0 {field=} error {e=}")
                for i, item in enumerate(dict_of_merged_outputs[field]):
                    logger.error(f"{i=} {item.shape=}")
                breakpoint()
                # raise e  # TODO raise
        elif isinstance(first_item, list):
            # this sum statement starts with [] and then adds all the contents (i.e. concatenates)
            dict_of_merged_outputs[field] = sum(dict_of_merged_outputs[field], [])
        elif isinstance(first_item, (int, float, str)):
            # scalars are concatenated as lists, that is fine
            pass
        else:
            raise ValueError(f"Unsupported type {type(first_item)}")
    return dict_of_merged_outputs


def deduplicate_outputs(
    dict_of_outputs: dict[str, Union[torch.Tensor, Any]], deduplicate_field: str
) -> dict[str, Union[torch.Tensor, Any]]:
    """
    Removes duplicate entries from a dictionary based on values in deduplicate_field.
    Preserves the first occurrence of each unique value and updates all fields accordingly.
    Works with both tensor and non-tensor fields.
    """
    keys = dict_of_outputs[deduplicate_field]
    if isinstance(keys, torch.Tensor):
        keys = keys.tolist()
    keys_seen = set()
    dedup_idx = []
    for i, key in enumerate(keys):
        if key not in keys_seen:
            dedup_idx.append(i)
        keys_seen.add(key)
    dedup_tensor = torch.tensor(dedup_idx, dtype=torch.long)
    # dedup tensor indices all datapoints that should remain

    # print(f"Deduplicated {n_total_inputs} into {len(dedup_tensor)}")

    for field in list(dict_of_outputs.keys()):
        if isinstance(dict_of_outputs[field], torch.Tensor):
            # field is a tensor shape [n_datapoints_with_dups, ...] and we can directly index
            dict_of_outputs[field] = dict_of_outputs[field][dedup_tensor]
        else:
            # field is a list of length [n_datapoints_with_dups] and we need to index each element
            dict_of_outputs[field] = [dict_of_outputs[field][i] for i in dedup_idx]

    return dict_of_outputs


def gather_object(obj, global_rank, world_size, group=None) -> list[Any]:
    """
    Gathers objects from all processes to rank 0 using PyTorch distributed.
    Returns gathered list on rank 0, None on other ranks.

    This is necessary since lightning cannot do this:
    https://github.com/Lightning-AI/pytorch-lightning/issues/14362
    """
    raise NotImplementedError(
        "This code can lead to CUDA sync issues. To gather tensors, use regular gather. To gather "
        f"objects, use the gather_object_on_filesystem function below instead."
    )
    group = group if group is not None else torch.distributed.group.WORLD
    if global_rank == 0:
        list_gather_obj = [None] * world_size  # the container of gathered objects.
        dist.gather_object(obj=obj, object_gather_list=list_gather_obj, dst=0, group=group)
        return list_gather_obj
    # ranks 1+
    dist.gather_object(obj=obj, object_gather_list=None, dst=0, group=group)
    return None


def gather_object_on_filesystem(
    obj, global_rank, world_size, base_dir, trainer=None, timeout_seconds=1800
) -> list[Any] | None:
    """
    Gathers objects from all processes to rank 0 using the filesystem.

    This is useful for when you need to gather e.g. dicts of strings.
    It will convert tensors to lists, so for tensors, use an appropriate all_gather instead.

    torch.gather_object was supposed to do this but it leads to CUDA sync issues.

    Args:
        obj: Object to gather from this rank
        global_rank: Global rank of this process
        world_size: Total number of processes
        base_dir: Base directory to use for temporary files
        timeout_seconds: Maximum time to wait for all ranks

    Returns:
        List of gathered objects on rank 0, None on other ranks
    """
    wi = WorldInfo(trainer)
    base_path = Path(base_dir)
    os.makedirs(base_path, exist_ok=True)
    # Each rank dumps its object
    filename_template = "temp_obj_{}_of_{}.json"
    obj_file = base_path / filename_template.format(global_rank, world_size)
    # wi.print_with_rank(f"Dumping {obj_file}")
    dump_json(obj, obj_file, indent=2, custom_format_nan_to_none=True, verbose=False)
    wi.barrier_safe()  # first barrier ensures all files are written

    if global_rank == 0:
        # load all objects
        list_gather_obj = []
        for r in range(world_size):
            obj_file_r = base_path / filename_template.format(r, world_size)
            obj_r = load_json(obj_file_r)
            list_gather_obj.append(obj_r)
        # clean up files
        for r in range(world_size):
            obj_file_r = base_path / filename_template.format(r, world_size)
            obj_file_r.unlink()
        wi.barrier_safe()  # second barrier to return all ranks at the same time
        return list_gather_obj
    else:
        wi.barrier_safe()  # second barrier to return all ranks at the same time
        return None


def save_outputs(
    trainer: lit.Trainer | None,
    test_outputs: list[dict[str, torch.Tensor]],
    all_gather_fn: Callable,  # all_gather reference: lightning.pytorch.core.module # line 666
    target_file: PathType,
    target_file_extras: PathType | None,
    extra_output_keys: list[str] | None = None,
    skip_save: bool = False,
    verbose: bool = False,
):
    """
    Gather outputs from all GPUs, deduplicate them, save them.
    """
    target_file = Path(target_file)
    if target_file_extras is None:
        # backwards compatibility for now
        logger.warning(f"target_file_extras should be set when calling base_model.py save_outputs")
        target_file_str = Path(target_file).as_posix()
        assert target_file_str.endswith("outputs.pt")
        target_file_extras = Path(f"{target_file_str[:-len('outputs.pt')]}extras.pt")
    else:
        target_file_extras = Path(target_file_extras)
    if extra_output_keys is None:
        extra_output_keys = []
    wi = WorldInfo(trainer)
    # ---------- consolidate the outputs on all processes individually
    merged_outputs: dict[str, Any] = consolidate_outputs(test_outputs)
    # print_with_rank(f"Test outputs: {merged_outputs['idx'].shape} {merged_outputs['idx'].device}")

    # ---------- gather the outputs from all processes.
    print_fn = wi.print_with_rank if verbose else _no_print
    print_fn(f"Gathering {merged_outputs.keys()}")
    if wi.world_size == 1:
        # nothing to gather here
        synced_outputs = merged_outputs
    else:
        # split into tensors and non-tensors
        keys = list(merged_outputs.keys())
        tensor_keys, object_keys = [], []
        for key in keys:
            if isinstance(merged_outputs[key], torch.Tensor):
                tensor_keys.append(key)
            else:
                object_keys.append(key)

        synced_tensor_outputs = {}
        if len(tensor_keys) > 0:
            # merge tensors
            tensor_outputs = {k: merged_outputs[k] for k in tensor_keys}
            tensor_outputs["__rank_check__"] = torch.tensor(wi.global_rank).to(
                tensor_outputs[tensor_keys[0]].device
            )

            for k, v in tensor_outputs.items():
                print_fn(f"Tensor '{k}' before sync: {v.shape}")
            synced_tensor_outputs = all_gather_fn(tensor_outputs)
            for k, v in synced_tensor_outputs.items():
                print_fn(f"Tensor '{k}' after sync: {v.shape}")

            # if processes > 1, will return shape (n_processes, *old_shape)
            # if processes == 1, will return the old shape
            # also, all_gather puts the data back on the GPU. it seems not easily possible all gather on CPU

            # sto_repr = ", ".join(f"{k}: {v.shape}" for k, v in synced_tensor_outputs.items())
            # print_with_rank(f"\nSynced tensor: {sto_repr}\n")
            rank_check = synced_tensor_outputs.pop("__rank_check__").cpu().tolist()
            expected_ranks = list(range(wi.world_size))
            assert rank_check == expected_ranks, f"{rank_check=} != {expected_ranks=}"

        synced_object_outputs, synced_object_outputs_list = {}, None
        if len(object_keys) > 0:
            print_fn(f"Object keys: {object_keys}")
            # merge objects
            object_outputs = {k: merged_outputs[k] for k in object_keys}
            object_outputs["__rank_check__"] = wi.global_rank
            for k, v in object_outputs.items():
                print_fn(f"Object '{k}' before sync: {repr_value(v)}")
            synced_object_outputs_list = gather_object_on_filesystem(
                object_outputs, wi.global_rank, wi.world_size, target_file.parent, trainer=trainer
            )
            if synced_object_outputs_list is None:
                print_fn(f"{synced_object_outputs_list=}")
            else:
                print_fn(f"Synced objects: {repr_value(synced_object_outputs_list)}")

        # return all non-main processes since the gathering is done
        if not wi.is_global_zero:
            del (
                merged_outputs,
                synced_tensor_outputs,
                synced_object_outputs,
                synced_object_outputs_list,
            )
            wi.barrier_safe()
            return

        # consolidate tensors (currently shaped (world_size, *other_shape)) and move to cpu
        for k in tensor_keys:
            value = synced_tensor_outputs[k]
            synced_tensor_outputs[k] = value.view(-1, *value.shape[2:]).cpu()

        if len(object_keys) > 0:
            # consolidate the object outputs (currently a list of dicts)
            synced_object_outputs = consolidate_outputs(synced_object_outputs_list)
            # print_with_rank(f"\nSynced consolidated objects: {synced_object_outputs=}\n")
            rank_check = synced_object_outputs.pop("__rank_check__")
            assert rank_check == list(range(wi.world_size)), f"{rank_check=}"

        synced_outputs = {**synced_tensor_outputs, **synced_object_outputs}

    # ---------- deduplicate and save
    keys_set = set(synced_outputs.keys())
    possible_dedup_keys = ("idx", "key")
    deduplicate_field = None
    for k in possible_dedup_keys:
        if k in keys_set:
            deduplicate_field = k
            break
    if deduplicate_field is None:
        raise ValueError(
            f"Could not find a field for deduplication in output keys: {keys_set}, "
            f"searched for {possible_dedup_keys}. Either add datapoint index integer as 'idx' or "
            f"a string unique identifier as 'key'."
        )
    clean_outputs = deduplicate_outputs(synced_outputs, deduplicate_field)

    # epoch_ident = epoch_identifier
    # logger.info(f"Completed test for task {eval_task_key} checkpoint {epoch_ident}")
    # for k, v in clean_outputs.items():
    #     logger.info(f"Got test output {k}: {v.shape}")

    if not skip_save:
        target_file = Path(target_file)
        logger.debug(f"Saving eval outputs to {target_file} on main process")
        os.makedirs(target_file.parent, exist_ok=True)
        extra_outputs = {}
        for key_extra in extra_output_keys:
            if key_extra in clean_outputs:
                extra_outputs[key_extra] = clean_outputs.pop(key_extra)
        torch.save(clean_outputs, target_file)
        if len(extra_outputs) > 0:
            torch.save(extra_outputs, target_file_extras)

    # all non-main processes are waiting, use barrier to meet them
    print_fn(f"Final outputs: {repr_value(clean_outputs)}")
    wi.barrier_safe()
    return clean_outputs
