import torch
from torch.nn.functional import cosine_similarity

from entitynet.models.base_model import LitBaseModel
from entitynet.tasks.base_task import EVAL_OUTPUT_TYPE, BaseTask


class SugarCrepePPTask(BaseTask):
    """
    Algorithm source: https://github.com/Sri-Harsha/scpp
    """

    def run_eval_step(self, model, batch: dict):
        image, idx = batch["image"], batch["idx"]
        p1, p2, neg = batch["caption"], batch["caption2"], batch["negative_caption"]
        img_feats = model.model.encode_image(image, normalize=True)  # (B, D)
        device = image.device
        p1_txt = model.tokenize_text(p1).to(device)
        p1_feats = model.model.encode_text(p1_txt, normalize=True)  # (B, D)
        p2_txt = model.tokenize_text(p2).to(device)
        p2_feats = model.model.encode_text(p2_txt, normalize=True)  # (B, D)
        neg_txt = model.tokenize_text(neg).to(device)
        neg_feats = model.model.encode_text(neg_txt, normalize=True)  # (B, D)

        cos_p1 = cosine_similarity(img_feats, p1_feats)
        # cosine similarities between image and P1 (positive caption 1)
        cos_p2 = cosine_similarity(img_feats, p2_feats)
        # cosine similarities between image and P2 (positive caption 2)
        cos_neg = cosine_similarity(img_feats, neg_feats)
        # cosine similarities between image and Negative (negative caption)
        cos_p1p2 = cosine_similarity(p1_feats, p2_feats)
        # cosine similarities between P1 and P2 for text-only task
        cos_p1_neg = cosine_similarity(p1_feats, neg_feats)
        # cosine similarities between P1 and Negative for text-only task
        cos_p2_neg = cosine_similarity(p2_feats, neg_feats)
        # cosine similarities between P2 and Negative for text-only task

        result = {
            "idx": idx.detach().cpu(),  # (B,)
            "cos_p1": cos_p1,
            "cos_p2": cos_p2,
            "cos_neg": cos_neg,
            "cos_p1p2": cos_p1p2,
            "cos_p1_neg": cos_p1_neg,
            "cos_p2_neg": cos_p2_neg,
        }
        return result

    def on_eval_end(self, model: LitBaseModel, dataset, eval_output: EVAL_OUTPUT_TYPE):
        if not model.trainer.is_global_zero:
            return
        _idx = eval_output["idx"]  # datapoint numbers
        cos_p1 = eval_output["cos_p1"]
        cos_p2 = eval_output["cos_p2"]
        cos_neg = eval_output["cos_neg"]
        cos_p1p2 = eval_output["cos_p1p2"]
        cos_p1_neg = eval_output["cos_p1_neg"]
        cos_p2_neg = eval_output["cos_p2_neg"]

        total = len(cos_p1)
        # if cos_p1 > cos_neg and cos_p2 > cos_neg: correct_full += 1
        correct_full = torch.logical_and(cos_p1 > cos_neg, cos_p2 > cos_neg).int().sum().item()
        # if cos_p1 > cos_neg: correct_img_p1 += 1
        correct_img_p1 = (cos_p1 > cos_neg).int().sum().item()  # noqa
        # if cos_p2 > cos_neg: correct_img_p2 += 1
        correct_img_p2 = (cos_p2 > cos_neg).int().sum().item()  # noqa
        # if cos_p1p2 > cos_p1_neg and cos_p1p2 > cos_p2_neg: correct_text += 1
        cor_txt = torch.logical_and(cos_p1p2 > cos_p1_neg, cos_p1p2 > cos_p2_neg).int().sum().item()

        metrics = {
            "scpp_i2t": correct_full / total,
            "scpp_ip1n": correct_img_p1 / total,
            "scpp_ip2n": correct_img_p2 / total,
            "scpp_tont": cor_txt / total,
        }
        return metrics
