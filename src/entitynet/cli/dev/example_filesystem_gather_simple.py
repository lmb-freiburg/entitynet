#!/usr/bin/env python
"""
Gather arbitrary objects from all GPU‑backed processes via a shared filesystem
and print the merged result on rank 0.  Designed to be launched with *torchrun*:

λ torchrun --standalone --nproc_per_node=5  -m entitynet.cli.example_filesystem_gather_simple
"""

from __future__ import annotations

import argparse
from typing import Any, List

import torch
import torch.distributed as dist

from entitynet.litext.distributed_gathering import gather_object_on_filesystem


def run(rank: int, world_size: int, base_dir: str) -> None:
    """Body executed by every rank."""
    # Set the GPU corresponding to this rank
    torch.cuda.set_device(rank)

    dummy_tensor = torch.full((2, 3), rank, dtype=torch.float32)
    dummy_text = f"sample_{rank}"
    payload: dict[str, Any] = {
        "tensor": dummy_tensor.tolist(),
        "text": dummy_text,
    }

    gathered: List[dict[str, Any]] = gather_object_on_filesystem(
        payload, rank, world_size, base_dir
    )

    print(f"\nRank {rank}: Merged result: {gathered}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Filesystem gather example driven by torchrun")
    parser.add_argument(
        "--base-dir",
        default="./gather_fs_tmp",
        help="Directory that all ranks can write to/read from",
    )
    args = parser.parse_args()

    # torchrun has already exported MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE, etc.
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    try:
        run(rank, world_size, args.base_dir)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
