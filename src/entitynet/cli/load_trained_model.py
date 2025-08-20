"""
loads a model checkpoint and runs a retrieval task
"""

import gc
from pathlib import Path
from traceback import format_exception

import torch
from attrs import define
from loguru import logger

from packg.log import SHORTEST_FORMAT, configure_logger, get_logger_level_from_args
from packg.tqdmext import tqdm_max_ncols
from typedparser import TypedParser, VerboseQuietArgs, add_argument

from entitynet.config.main_config import Config
from entitynet.datasets.dataset_factory import build_eval_dataset_for_task
from entitynet.models.base_model import LitBaseModel
from entitynet.models.load_trained_model import load_trained_model
from entitynet.tasks.contrastive_task import (
    ContrastiveRetrievalTask,
    chunked_pairwise_cosine_similarity,
    compute_retrieval_cosine,
)
from entitynet.tasks.task_factory import create_task_from_config


@define
class Args(VerboseQuietArgs):
    config_file: Path = add_argument("config_file", type=str, help="Experiment config file")
    ckpt_mode: str = add_argument(
        type=str,
        help="Load the best checkpoint (instead of the last one which is the default)",
        default="last",
        choices=["best", "last", "none"],
    )
    run_id: str | None = add_argument(help="Run id for neptune logger and subfolder")
    device: str = add_argument(type=str, help="Device to load the model on", default="cpu")
    val_task: str | None = add_argument(help="Validation task to run", default=None)
    workers: int = add_argument(type=int, help="Number of workers to use", default=0)


def main():
    parser = TypedParser.create_parser(Args, description=__doc__)
    args: Args = parser.parse_args()
    configure_logger(level=get_logger_level_from_args(args), format=SHORTEST_FORMAT)
    logger.info(f"{args}")

    model, vis_prep, tokenizer, config = load_trained_model(
        args.config_file, args.ckpt_mode, args.run_id, args.device
    )
    logger.info(f"Model loaded successfully")
    if args.val_task is not None:
        run_val_retrieval_task(args.val_task, args.workers, model, config)
        return
    # export image-text pairs. both en-de, including duplicate texts.


def run_val_retrieval_task(task_key: str, workers: int, model: LitBaseModel, config: Config):
    task_cfg = config.eval_tasks[task_key]
    task_cfg.dataset.batch_size_eval = 64
    dataset, loader = build_eval_dataset_for_task(task_key, task_cfg, workers)
    val_task = create_task_from_config(task_key, task_cfg)
    assert isinstance(val_task, ContrastiveRetrievalTask)

    # technical setup
    if config.trainer.set_float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(config.trainer.set_float32_matmul_precision)

    with torch.no_grad():
        if config.trainer.precision == "bf16-mixed":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                returns = run_inference(model, loader)
        elif config.trainer.precision == "16-mixed":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                returns = run_inference(model, loader)
        else:
            returns = run_inference(model, loader)

    image_feat, text_feat, texts, idx = returns

    # get metrics
    dot_i2t = chunked_pairwise_cosine_similarity(image_feat, text_feat)
    print(dot_i2t.shape)
    metrics_i2t, other = compute_retrieval_cosine(dot_i2t)
    print(metrics_i2t)
    dot_t2i = chunked_pairwise_cosine_similarity(text_feat, image_feat)
    print(dot_t2i.shape)
    metrics_t2i, other = compute_retrieval_cosine(dot_t2i)
    print(metrics_t2i)

    average_meanr = (metrics_i2t["meanr"] + metrics_t2i["meanr"]) / 2
    print(f"Average meanr: {average_meanr}")


def run_inference(model, loader):
    # get embeddings
    device = model.device
    collector = {"image_feat": [], "text_feat": [], "idx": [], "key": [], "text": []}
    for i, batch in enumerate(tqdm_max_ncols(loader)):
        image = batch["image"].to(device)
        image_feat_here = model.encode_image(image)
        tokens_here = model.tokenize_text(batch["text"])
        text_feat_here = model.encode_text(tokens_here.to(device), normalize=True)
        collector["image_feat"].append(image_feat_here.detach().cpu())
        collector["text_feat"].append(text_feat_here.detach().cpu())
        collector["idx"].extend(batch["idx"].detach().long().cpu().tolist())
        collector["text"].extend(batch["text"])

    image_feat = torch.cat(collector["image_feat"], dim=0).float().cpu()
    text_feat = torch.cat(collector["text_feat"], dim=0).float().cpu()
    idx = collector["idx"]
    texts = collector["text"]
    del collector
    try:
        del image, image_feat_here, tokens_here, text_feat_here
    except Exception as e:
        print(f"{format_exception(e)}")
    gc.collect()
    torch.cuda.empty_cache()
    return image_feat, text_feat, texts, idx


if __name__ == "__main__":
    main()
