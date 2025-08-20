from timeit import default_timer
from typing import Any

import torch

from packg.log import logger
from visiontext.distutils import WorldInfo

from entitynet.config.task_config import ClipContrastiveMultiTextCfg
from entitynet.models.base_model import LitBaseModel
from entitynet.tasks.base_task import EVAL_OUTPUT_TYPE, BaseTask
from entitynet.tasks.contrastive_task import (
    chunked_pairwise_cosine_similarity,
    merge_retrieval_metrics_inplace,
)


class ContrastiveRetrievalTaskMultiText(BaseTask):
    """
    Logic for multitext retrieval inspired by
    https://github.com/salesforce/LAVIS/blob/main/lavis/tasks/retrieval.py

    This class assumes exactly the same amount of texts per each image, given by
    task_cfg.n_texts_per_image.
    """

    task_cfg: ClipContrastiveMultiTextCfg

    def run_eval_step(self, model, batch: dict):
        image = batch["image"]
        device = image.device
        image_feat = model.encode_image(batch["image"], normalize=True)  # (B, D)
        # encode all text descriptors
        text_feature_list = []
        text_list = batch["text_list"]  # list len n_descriptors, of len batchsize, of string
        assert len(text_list) == self.task_cfg.n_texts_per_image
        for i_descriptor, text in enumerate(text_list):
            # for each image we have multiple descriptors e.g. 5 for coco
            tokens = model.tokenize_text(text).to(batch["image"].device)
            text_feature = model.model.encode_text(tokens.to(device), normalize=True)
            text_feature_list.append(text_feature)
        text_feature_list = torch.stack(text_feature_list, dim=1)  # (B, N_descriptors, D)
        # collect all image and text features for retrieval later
        result = {}
        result["image_features"] = image_feat.detach().cpu()
        result["text_list_features"] = text_feature_list.detach().cpu()
        if "idx" in batch:
            result["idx"] = batch["idx"].cpu()
        if "key" in batch:
            result["key"] = batch["key"]  # string key
        return result

    def on_eval_end(self, model: LitBaseModel, dataset, eval_output: EVAL_OUTPUT_TYPE):
        wi = WorldInfo(model.trainer)
        if not wi.is_global_zero:
            return None
        all_metrics = {}

        # get features from eval_output
        image_feature: torch.Tensor = eval_output["image_features"]  # (N_datapoints, D)
        text_list_feature: torch.Tensor = eval_output["text_list_features"]  # (N_d, N_descr, D)
        n_datapoints, n_texts_per_image, dim = text_list_feature.shape

        # sanity checks / prints
        wi.print_with_rank(
            f"Evaluate retrieval for image_features: {image_feature.shape} "
            f"text_features: {text_list_feature.shape}"
        )
        assert len(image_feature) == len(text_list_feature), (
            f"Number of images and texts must match but are shapes "
            f"{image_feature.shape=} {text_list_feature.shape=}"
        )
        if not model.trainer.sanity_checking:
            assert len(image_feature) == len(dataset), (
                f"Number of images and texts must match but are "
                f"{len(image_feature)=} {len(dataset)=}"
            )

        text_features_flat = text_list_feature.view(-1, dim)
        img2txt_indices = torch.arange(len(text_features_flat)).reshape(
            n_datapoints, n_texts_per_image
        )
        use_model_device = False  # True
        if use_model_device:
            image_feature = image_feature.to(model.device)
            text_features_flat = text_features_flat.to(model.device)

            logger.info(f"Compute dot product {image_feature.shape} x {text_features_flat.shape}T")
        dot = chunked_pairwise_cosine_similarity(image_feature, text_features_flat)

        # dot is now (N_datapoints, N_datapoints * N_descriptors)
        metrics_img2txt, _ = compute_retrieval_cosine_multitext_i2t(dot, img2txt_indices)
        for k, v in metrics_img2txt.items():
            all_metrics[f"i2t_{k}"] = v
        metrics_txt2img, _ = compute_retrieval_cosine_multitext_t2i(dot, img2txt_indices)
        for k, v in metrics_txt2img.items():
            all_metrics[f"t2i_{k}"] = v
        merge_retrieval_metrics_inplace(all_metrics)
        return all_metrics


def compute_retrieval_cosine_multitext_i2t(
    dot: torch.Tensor, img2txt_indices: torch.Tensor
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    assumes a fixed amount of texts per image. for each given image, finds the best rank over
    all texts that match the image, and reports that.

    Args:
        dot: cosine similarity computed as image @ text.T with shape
            (N_images, N_images * N_texts_per_image)
        img2txt_indices: tensor of shape (N_images, N_texts_per_image)
            with indices of the text descriptors for each image e.g. [[0, 1, 2], [3, 4, 5], ...]

    Returns:
        dictionary of metrics for image to text retrieval,
        dictionary of other values  (top1 pred index for each row, rank for each row)
    """
    dot = dot.detach().cpu()
    n = len(dot)
    ranks = torch.empty(n)
    # top1 = torch.empty(n)

    logger.info(f"Start computing retrieval metrics: {dot.shape=}")

    # loop rows (images)
    for index in range(n):
        # sort columns by highest similarity descending
        inds = torch.argsort(dot[index], descending=True)

        # now we find the best rank over all the texts for this image
        rank = 2_147_483_647  # max signed int32
        text_indices = img2txt_indices[index]
        for txt_i in text_indices:
            where = torch.where(inds == txt_i)[0][0].item()
            rank = min(rank, where)
        ranks[index] = rank
        # here we do not save the top1 result

    # print(f"{default_timer()-t1:.3f}s done computing ranks")

    # compute retrieval metrics
    r1 = len(torch.where(ranks < 1)[0]) / len(ranks)
    r5 = len(torch.where(ranks < 5)[0]) / len(ranks)
    r10 = len(torch.where(ranks < 10)[0]) / len(ranks)
    r20 = len(torch.where(ranks < 20)[0]) / len(ranks)
    r50 = len(torch.where(ranks < 50)[0]) / len(ranks)
    medr = (torch.floor(torch.median(ranks)) + 1).item()
    meanr = (ranks.mean() + 1).item()
    report_dict = {
        "r1": r1,
        "r5": r5,
        "r10": r10,
        "r20": r20,
        "r50": r50,
        "medr": medr,
        "meanr": meanr,
        "n": n,
    }
    other = {
        "ranks": ranks,
    }
    # print(f"{default_timer()-t1:.3f}s done: {report_dict=}")
    return report_dict, other


def compute_retrieval_cosine_multitext_t2i(
    dot: torch.Tensor, img2txt_indices: torch.Tensor
) -> tuple[dict[str, float], dict[str, Any]]:
    """
    assumes a fixed amount of texts per image. for each text, finds the image

    Args:
        dot: cosine similarity computed as image @ text.T with shape
            (N_images, N_images * N_texts_per_image)
        img2txt_indices: tensor of shape (N_images, N_texts_per_image)
            with indices of the text descriptors for each image e.g. [[0, 1, 2], [3, 4, 5], ...]

    Returns:
        dictionary of metrics for text to image retrieval,
        dictionary of other values  (top1 pred index for each row, rank for each row)
    """
    n_images, n_texts_per_image = img2txt_indices.shape
    n_texts_total = n_images * n_texts_per_image

    dot = dot.detach().cpu()
    ranks = torch.empty(n_texts_total)
    # top1 = torch.empty(n)

    txt2img_indices = torch.arange(n_images).repeat(n_texts_per_image, 1).T.flatten()
    # [[ 0, 0, 0], [ 1, 1, 1], ...]

    logger.info(f"Start computing retrieval metrics: {dot.shape=}")

    # loop columns (texts)
    for index in range(n_texts_total):
        # sort columns by highest similarity descending
        sims = dot[:, index]
        inds = torch.argsort(sims, descending=True)
        correct_image_index = txt2img_indices[index]
        where = torch.where(inds == correct_image_index)
        rank = where[0][0]
        ranks[index] = rank
        # top1[index] = inds[0]  # to save the top1 result:

    # ranks = ranks.reshape(n_images, n_texts_per_image)
    # ranks = ranks.min(dim=1).values

    # compute retrieval metrics
    r1 = len(torch.where(ranks < 1)[0]) / len(ranks)
    r5 = len(torch.where(ranks < 5)[0]) / len(ranks)
    r10 = len(torch.where(ranks < 10)[0]) / len(ranks)
    r20 = len(torch.where(ranks < 20)[0]) / len(ranks)
    r50 = len(torch.where(ranks < 50)[0]) / len(ranks)
    medr = (torch.floor(torch.median(ranks)) + 1).item()
    meanr = (ranks.mean() + 1).item()
    report_dict = {
        "r1": r1,
        "r5": r5,
        "r10": r10,
        "r20": r20,
        "r50": r50,
        "medr": medr,
        "meanr": meanr,
        "n": n_images,
    }
    other = {
        "ranks": ranks,
    }
    # print(f"{default_timer()-t1:.3f}s done: {report_dict=}")
    return report_dict, other
