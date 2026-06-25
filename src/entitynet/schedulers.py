import functools
import math

import torch

from packg import Const


class SchedulerC(Const):
    COSINE = "cosine"
    COSINE_CONSTANT = "cosine_constant"
    CONST = "const"
    NONE = "none"


def create_scheduler(optimizer, scheduler_name, total_steps, warmup_steps, constant_steps):
    # note seems the open_clip schedulers are not compatible with lightning
    if scheduler_name == SchedulerC.COSINE:
        scheduler = WarmupCosineScheduler(optimizer, total_steps, warmup_steps)
    elif scheduler_name == SchedulerC.COSINE_CONSTANT:
        scheduler = WarmupConstantCosineScheduler(
            optimizer, total_steps, warmup_steps, constant_steps
        )
    elif scheduler_name == SchedulerC.CONST:
        scheduler = WarmupConstantScheduler(optimizer, warmup_steps)
    elif scheduler_name == SchedulerC.NONE:
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")
    return scheduler


def WarmupConstantCosineScheduler(optimizer, total_steps, warmup_steps, constant_steps):
    _decay_fn = functools.partial(
        warmup_const_cosine,
        warmup_steps=warmup_steps,
        constant_steps=constant_steps,
        total_steps=total_steps,
    )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_fn)


def warmup_const_cosine(current_step, warmup_steps, total_steps, constant_steps):
    pre_decay_steps = warmup_steps + constant_steps
    if current_step <= warmup_steps and warmup_steps > 0:
        return current_step / warmup_steps
    if current_step <= pre_decay_steps and constant_steps > 0:
        return 1.0
    if total_steps - pre_decay_steps < 0:
        raise ValueError(f"{total_steps=} {warmup_steps=} {constant_steps=} {current_step=}")
    rel_step = (current_step - pre_decay_steps) / (total_steps - pre_decay_steps)
    return 0.5 * (1 + math.cos(math.pi * rel_step))


def WarmupCosineScheduler(optimizer, total_steps, warmup_steps):
    _decay_fn = functools.partial(warmup_cosine, warmup_steps=warmup_steps, total_steps=total_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_fn)


def warmup_cosine(current_step, warmup_steps, total_steps):
    if current_step <= warmup_steps and warmup_steps > 0:
        return current_step / warmup_steps
    if total_steps - warmup_steps == 0:
        raise ValueError(f"{total_steps=} {warmup_steps=} {current_step=}")
    rel_step = (current_step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * rel_step))


def WarmupConstantScheduler(optimizer, warmup_steps):
    _decay_fn = functools.partial(warmup_const, warmup_steps=warmup_steps)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _decay_fn)


def warmup_const(current_step, warmup_steps):
    if current_step <= warmup_steps:
        return current_step / warmup_steps
    return 1.0
