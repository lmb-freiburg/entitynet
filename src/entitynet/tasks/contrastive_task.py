from pprint import pformat
from timeit import default_timer
from typing import Any

import torch
from attr import asdict
from lightning.fabric.utilities.exceptions import MisconfigurationException

from packg.log import logger
from visiontext.distutils import WorldInfo, print_with_rank

from entitynet.config.task_config import ClipContrastiveTaskCfg
from entitynet.models.base_model import LitBaseModel
from entitynet.tasks.base_task import EVAL_OUTPUT_TYPE, BaseTask


class ContrastiveRetrievalTask(BaseTask):
    def run_eval_step(self, model: LitBaseModel, batch: dict):
        task_cfg: ClipContrastiveTaskCfg = self.task_cfg

        # run contrastive task, compute loss
        batch["tokens"] = model.tokenize_text(batch["text"]).to(batch["image"].device)
        batch_size = batch["image"].shape[0]
        out_dict = model.model(batch["image"], batch["tokens"])
        result = {}
        print_with_rank(f"contrastive eval task with loss {model.loss}")
        if model.loss is not None:
            model.loss.is_val = True
            loss = model.loss(**out_dict, output_dict=False)
            loss_key = f"{model.eval_phase}_loss"
            new_name = model.log_with_auto_rename(loss_key, loss, batch_size, task_cfg.task_key)
            # print_with_rank(f"logged loss {loss} as {new_name=} {task_cfg.task_key=}")
            # the result dict expects one entry per datapoint, so we repeat the averaged loss.
            result = {"loss": [loss.detach().cpu().item()] * batch_size}
        if not task_cfg.run_retrieval:
            return result
        result["image_features"] = out_dict["image_features"].detach().cpu()
        result["text_features"] = out_dict["text_features"].detach().cpu()
        if "idx" in batch:
            # incrementing datapoint number
            result["idx"] = batch["idx"].cpu()
        if "key" in batch:
            result["key"] = batch["key"]  # string key
        return result

    def on_eval_end(self, model: LitBaseModel, dataset, eval_output: EVAL_OUTPUT_TYPE):
        task_cfg: ClipContrastiveTaskCfg = self.task_cfg
        wi = WorldInfo(model.trainer)
        if wi.is_global_zero:
            all_metrics = {}
            loss = self.aggregate_loss_if_exists(model, dataset, eval_output)
            if loss is not None:
                all_metrics["loss"] = loss
            if not task_cfg.run_retrieval:
                return all_metrics

            image_features = eval_output["image_features"]  # (N, D)
            text_features = eval_output["text_features"]  # (N, D)
            wi.print_with_rank(
                f"Evaluate retrieval for image_features: {image_features.shape} "
                f"text_features: {text_features.shape}"
            )
            assert len(image_features) == len(text_features), (
                f"Number of images and texts must match but are shapes "
                f"{image_features.shape=} {text_features.shape=}"
            )
            if not model.trainer.sanity_checking:
                assert len(image_features) == len(dataset), (
                    f"Number of images and texts must match but are "
                    f"{len(image_features)=} {len(dataset)=}"
                )
            dot = chunked_pairwise_cosine_similarity(image_features, text_features)  # (N, N)
            metrics_img2txt, _ = compute_retrieval_cosine(dot)
            for k, v in metrics_img2txt.items():
                all_metrics[f"i2t_{k}"] = v
            metrics_txt2img, _ = compute_retrieval_cosine(dot.T)
            for k, v in metrics_txt2img.items():
                all_metrics[f"t2i_{k}"] = v
            merge_retrieval_metrics_inplace(all_metrics)
            return all_metrics
        return None


def merge_retrieval_metrics_inplace(all_metrics):
    for k in list(all_metrics.keys()):
        if k.startswith("i2t_"):
            k2 = f"t2i_{k[4:]}"
            if k2 in all_metrics:
                all_metrics[f"both_{k[4:]}"] = (all_metrics[k] + all_metrics[k2]) / 2
            else:
                logger.error(f"Metric {k} exists but {k2} does not, in {all_metrics=}")


def chunked_pairwise_cosine_similarity(
    x: torch.Tensor, y: torch.Tensor, chunk_size: int = 1000, eps=1e-8
) -> torch.Tensor:
    """
    Pairwise cosine similarity with following features:
    - second matrix is chunked to avoid OOM / increase speed
    - using float32 for computation if input is float16 (suggested by torchmetrics cos_sim)
    - l2-normalize input vectors but avoid division by zero
    """
    shape_or_dim_error = f"cosine_similarity got input with wrong shapes: {x.shape=} {y.shape=}"
    assert x.ndim == 2, shape_or_dim_error
    assert y.ndim == 2, shape_or_dim_error
    assert x.shape[1] == y.shape[1], shape_or_dim_error
    # t1 = default_timer()
    # print(f"{default_timer()-t1:.3f}s start cos sim with {x.shape=} {x.dtype=} {y.shape=} {y.dtype=}")

    # torchmetrics wants to do the matmul in float32, do it here as well
    cast_back = False
    if x.dtype == torch.float16 or y.dtype == torch.float16:
        x = x.float()
        y = y.float()
        cast_back = True

    eps_tensor = x.new_ones(x.shape[0]) * eps
    norm = torch.maximum(torch.norm(x, p=2, dim=1), eps_tensor)
    x = x / norm.unsqueeze(1)
    eps_tensor = y.new_ones(y.shape[0]) * eps
    norm = torch.maximum(torch.norm(y, p=2, dim=1), eps_tensor)
    y = y / norm.unsqueeze(1)
    # print(f"{default_timer()-t1:.3f}s done normalizing. running the matmul")
    outputs = []
    for chunk_start in range(0, len(y), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(y))
        # print(f"{default_timer()-t1:.3f}s matmul chunk {chunk_start} to {chunk_end}")
        y_chunk = y[chunk_start:chunk_end]
        output_chunk = torch.matmul(x, y_chunk.T)
        outputs.append(output_chunk)
    output = torch.cat(outputs, dim=1)

    if cast_back:
        output = output.half()
    # print(f"{default_timer()-t1:.3f}s returning {output.shape=}")
    return output


def compute_retrieval_cosine(dot: torch.Tensor) -> tuple[dict[str, float], dict[str, Any]]:
    """
    Args:
        dot: cosine similarity computed as image @ text.T with shape (N, N)

    Returns:
        dictionary of metrics,
        dictionary of other values  (top1 pred index for each row, rank for each row)
    """
    n = len(dot)
    ranks = torch.empty(n)
    top1 = torch.empty(n)

    t1 = default_timer()
    # print(f"{default_timer()-t1:.3f}s start retrieval metrics {dot.shape=}")

    # loop rows
    for index in range(n):
        # sort columns by highest similarity descending
        inds = torch.argsort(dot[index], descending=True)
        # the label (correct pair) is also "index". get rank of this correct embedding
        where = torch.where(inds == index)
        rank = where[0][0]
        ranks[index] = rank

        # to save the top1 result:
        top1[index] = inds[0]

    # print(f"{default_timer()-t1:.3f}s done computing ranks")

    # compute retrieval metrics
    r1 = len(torch.where(ranks < 1)[0]) / len(ranks)
    r5 = len(torch.where(ranks < 5)[0]) / len(ranks)
    # r10 = len(torch.where(ranks < 10)[0]) / len(ranks)
    medr = (torch.floor(torch.median(ranks)) + 1).item()
    meanr = (ranks.mean() + 1).item()
    report_dict = {
        "r1": r1,
        "r5": r5,
        # "r10": r10,
        "medr": medr,
        "meanr": meanr,
        "n": n,
    }
    other = {
        "top1": top1,
        "ranks": ranks,
    }
    # print(f"{default_timer()-t1:.3f}s done: {report_dict=}")
    return report_dict, other
