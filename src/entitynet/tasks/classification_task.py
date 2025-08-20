"""
Classification tasks.

Notes:
    There are several options when confronted with multiple templates or synonyms.
    The default OpenAI CLIP way is to average the embeddings of the synonyms to get one
    embedding per class. This is the one most commonly used and the one implemented here.

    Alternatively one could compute the logits per synonym and then take the max or mean over
    groups of synonyms to get the logits per class, see function reduce_synonym_logits_over_classes
"""

import math

import torch
import torch.distributed as dist
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy

from visiontext.distutils import WorldInfo

from entitynet.config.task_config import ClipZsClsTaskCfg
from entitynet.datasets.clip_templates import CLASSIFICATION_TEMPLATES
from entitynet.models.base_model import LitBaseModel
from entitynet.tasks.base_task import EVAL_OUTPUT_TYPE, BaseTask


class ClipZeroshotClassificationTask(BaseTask):
    task_cfg: ClipZsClsTaskCfg

    def setup(self):
        self.class_embeddings = None

    def on_eval_start(self, model: LitBaseModel, dataset) -> None:
        assert hasattr(
            dataset, "classes"
        ), f"Eval dataset {self } missing 'classes' attr. (list[str] of class names)"
        classes = dataset.classes
        templates = get_templates(dataset, self.task_cfg.clip_zs_template)
        if self.task_cfg.use_synonyms:
            for attribute in ["synonyms", "synonym_to_class"]:
                assert hasattr(dataset, attribute), (
                    f"{self.task_cfg.use_synonyms=} but dataset {type(dataset)} is missing the "
                    f"{attribute} attribute. Given e.g. the classes ['dog', 'cat'] then synonyms "
                    f"should be e.g. ['dog', 'hound', 'doggo', 'cat', 'kitty'] and synonym_to_class "
                    f"should be a list of integers, where the index is the synonym id and the value "
                    f"is the class id, for this example the mapping should be [0, 0, 0, 1, 1] "
                )
            synonyms = dataset.synonyms
            synonym_to_class = dataset.synonym_to_class
            class_embeddings = create_class_embeddings_with_synonyms(
                model,
                classes,
                templates,
                synonyms,
                synonym_to_class,
                batch_size=self.task_cfg.dataset.batch_size_eval,
                trainer=model.trainer,
            )
        else:
            class_embeddings = create_class_embeddings(
                model,
                classes,
                templates,
                batch_size=self.task_cfg.dataset.batch_size_eval,
                trainer=model.trainer,
            )
        self.class_embeddings = class_embeddings

    def run_eval_step(self, model, batch):
        class_embeddings = self.class_embeddings
        force_output_embs = model.config.trainer.eval_force_output_embeddings
        image, label, idx = batch["image"], batch["label"], batch["idx"]
        image_feat = model.encode_image(image, normalize=True)  # (B, D)
        sim = torch.einsum("bd,cd->bc", image_feat, class_embeddings)  # (B, C)
        pred = sim.argmax(dim=1)
        result = {
            "logits": sim.detach().cpu(),
            "pred": pred.detach().cpu(),
            "label": label.detach().cpu(),
            "idx": idx.detach().cpu(),
        }
        if force_output_embs:
            result["image_features"] = image_feat.detach().cpu()
        return result

    def on_eval_end(self, model: LitBaseModel, dataset, eval_output: EVAL_OUTPUT_TYPE):
        """
        For synonyms:
            Assuming we have C concepts and N classes, where multiple classes are synonyms
            for the same concept (N < C). We get the logit per concept by taking the argmax over all
            synonyms, and get the concept of the label my mapping from class to concept.

        """
        if not model.trainer.is_global_zero:
            return
        # warnings.filterwarnings("ignore", message=".*It is recommended to use.*")
        classes: list[str] = dataset.classes

        logits, label = eval_output["logits"], eval_output["label"]
        if not model.trainer.sanity_checking:
            assert len(logits) == len(dataset), f"{len(logits)=} != {len(dataset)=}"
        if len(classes) == 2:
            return calculate_binary_classification_metrics(len(classes), logits, label)
        else:
            return calculate_classification_metrics(len(classes), logits, label)


def calculate_classification_metrics(n_classes: int, logits, label):
    metrics_dict = get_classification_metrics(n_classes)
    metrics_dict.update(logits, label)
    results_dict = {}
    for metric_name, metric in metrics_dict.items():
        metric_value = metric.compute()
        results_dict[metric_name] = metric_value
    return results_dict


def calculate_binary_classification_metrics(n_classes: int, logits, label):
    metrics_dict = get_classification_metrics(n_classes)
    # for binary classification task there will be an dimension mismatch error,
    # e.g logits [n_sample, 2], label [n_sample]
    # I think here needs to be further checked as argmax decreased the dim of the logits
    # which might influence multiclassAccuracy
    # now somehow the result is correct.
    logits = torch.argmax(logits, dim=1)
    TP = torch.sum((logits == 1) & (label == 1)).item()
    FP = torch.sum((logits == 1) & (label == 0)).item()
    TN = torch.sum((logits == 0) & (label == 0)).item()
    FN = torch.sum((logits == 0) & (label == 1)).item()
    acc1_positive = TP / (TP + FN) if (TP + FN) > 0 else 0
    acc1_negative = TN / (TN + FP) if (TN + FP) > 0 else 0
    metrics_dict.update(logits, label)
    results_dict = {}
    for metric_name, metric in metrics_dict.items():
        metric_value = metric.compute()
        results_dict[metric_name] = metric_value
    results_dict["acc1_positive"] = acc1_positive
    results_dict["acc1_negative"] = acc1_negative
    return results_dict


def get_classification_metrics(n_classes):
    kw = dict(sync_on_compute=False)  # must be set for metrics that run only on main_process

    if n_classes >= 5:
        return MetricCollection(
            {
                "acc1": MulticlassAccuracy(num_classes=n_classes, average="micro", top_k=1, **kw),
                "acc5": MulticlassAccuracy(num_classes=n_classes, average="micro", top_k=5, **kw),
                "acc1_macro": MulticlassAccuracy(
                    num_classes=n_classes, average="macro", top_k=1, **kw
                ),
                "acc5_macro": MulticlassAccuracy(
                    num_classes=n_classes, average="macro", top_k=5, **kw
                ),
            },
            compute_groups=True,
        )
    else:
        return MetricCollection(
            {
                "acc1": MulticlassAccuracy(num_classes=n_classes, average="micro", top_k=1, **kw),
                "acc1_macro": MulticlassAccuracy(
                    num_classes=n_classes, average="macro", top_k=1, **kw
                ),
            },
            compute_groups=True,
        )


def get_templates(dataset, clip_zs_template):
    if clip_zs_template == "dataset":
        try:
            templates = dataset.templates
        except AttributeError:
            raise ValueError(
                f"{clip_zs_template=} but {dataset=} {type(dataset)=} missing templates attribute."
            )
    else:
        templates = CLASSIFICATION_TEMPLATES[clip_zs_template]
    return templates


@torch.inference_mode()
def create_class_embeddings(
    model, class_names: list[str], templates: list[str], batch_size: int = 256, trainer=None
):
    """Create class embeddings for zero-shot classification."""
    world_info = WorldInfo(trainer)
    device = model.device
    n_classes = len(class_names)
    n_templates = len(templates)
    if world_info.is_global_zero:
        all_texts = []
        for i, c in enumerate(class_names):
            texts = [template.format(c=c) for template in templates]
            all_texts.extend(texts)

        all_vectors = []
        n_batches = math.ceil(len(all_texts) / batch_size)
        for n_batch in range(n_batches):
            start_i = n_batch * batch_size
            end_i = min(start_i + batch_size, len(all_texts))
            texts_here = all_texts[start_i:end_i]
            tokens = model.tokenize_text(texts_here)
            vectors = model.encode_text(tokens.to(device), normalize=True)
            # Embeddings are collected and averaged on CPU to avoid running OOM
            all_vectors.append(vectors.detach().cpu())
        all_vectors = torch.cat(all_vectors)
        all_vectors = all_vectors.reshape(n_classes, n_templates, -1)
        class_embeddings = all_vectors.mean(dim=1)
        class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=-1)
        class_embeddings = class_embeddings.to(device)
    else:
        class_embeddings = torch.zeros(
            n_classes, model.embed_dim, dtype=torch.float32, device=device
        )
    world_info.barrier_safe()
    if world_info.world_size > 1:
        dist.broadcast(class_embeddings, src=0)
    return class_embeddings


@torch.inference_mode()
def create_class_embeddings_with_synonyms(
    model,
    class_names: list[str],
    templates: list[str],
    synonyms: list[str],
    synonym_to_class: list[int],
    batch_size: int = 256,
    trainer=None,
):
    """
    Create class embeddings for zero-shot classification, with a variable number of synonyms
    given per class and a list of templates to use for each synonym.
    """
    world_info = WorldInfo(trainer)
    device = model.device
    n_classes = len(class_names)
    n_synonyms = len(synonyms)
    n_templates = len(templates)
    if world_info.is_global_zero:
        # create average embeddings for each synonym, averaging over templates
        all_texts = []
        for i, c in enumerate(synonyms):
            texts = [template.format(c=c) for template in templates]
            all_texts.extend(texts)
        all_vectors = []
        n_batches = math.ceil(len(all_texts) / batch_size)
        for n_batch in range(n_batches):
            start_i = n_batch * batch_size
            end_i = min(start_i + batch_size, len(all_texts))
            texts_here = all_texts[start_i:end_i]
            tokens = model.tokenize_text(texts_here)
            vectors = model.encode_text(tokens.to(device), normalize=True)
            # Embeddings are collected and averaged on CPU to avoid running OOM
            all_vectors.append(vectors.detach().cpu())
        all_vectors = torch.cat(all_vectors)
        all_vectors = all_vectors.reshape(n_synonyms, n_templates, -1)
        all_vectors = torch.nn.functional.normalize(all_vectors, p=2, dim=-1)
        synonym_embeddings = all_vectors.mean(dim=1)
        synonym_embeddings = torch.nn.functional.normalize(synonym_embeddings, p=2, dim=-1)
        synonym_embeddings = synonym_embeddings.to(device)

        # now average over synonyms to get class embeddings. each class id has at least one synonym,
        # but the synonyms are not necessarily ordered by classes.
        class_to_synonym = [[] for _ in range(n_classes)]
        for synonym_id, class_id in enumerate(synonym_to_class):
            class_to_synonym[class_id].append(synonym_id)
        class_embeddings = torch.zeros(
            n_classes, model.embed_dim, dtype=torch.float32, device=device
        )
        for class_id, class_ids in enumerate(class_to_synonym):
            class_embeddings[class_id] = torch.mean(synonym_embeddings[class_ids], dim=0)
        class_embeddings = torch.nn.functional.normalize(class_embeddings, p=2, dim=-1)
    else:
        class_embeddings = torch.zeros(
            n_classes, model.embed_dim, dtype=torch.float32, device=device
        )
    world_info.barrier_safe()
    if world_info.world_size > 1:
        dist.broadcast(class_embeddings, src=0)
    return class_embeddings


def check_synonym_to_class_mapping(synonym_to_class: list[int]):
    class_ids = set(synonym_to_class)
    num_classes = len(class_ids)
    assert sorted(class_ids) == list(range(num_classes)), (
        f"synonym_to_class must map synonyms to a contiguous range of classes. There are "
        f"{num_classes} classes but the sorted set of classes looks like this: "
        f"{sorted(class_ids)}"
    )


def reduce_synonym_logits_over_classes(
    logits: torch.Tensor,
    synonym_to_class_tensor: torch.Tensor,
    op: str = "max",
) -> torch.Tensor:
    """
    Unused. Reduces logits over synonyms to logits over classes.

    Usage:
        class_logits = reduce_synonym_logits_over_classes(synonym_logits, synonym_to_class_tensor)

    Original source: github.com/lmb-freiburg/ovqa

    Args:
        logits: shape(n_datapoints, n_synonyms)
        synonym_to_class_tensor: shape(n_synonyms) in [0, n_classes) e.g. [0, 2, 1, 1, 7, 2, ...]
        op: "max" or "mean"

    Returns:
        logits: shape(n_datapoints, n_classes)
    """

    # so we now the concept ids are contiguous but the class ids are not ordered.
    num_concepts = synonym_to_class_tensor.max().item() + 1
    concept_to_class = [[] for _ in range(num_concepts)]
    for class_id, concept_id in enumerate(synonym_to_class_tensor.tolist()):
        concept_to_class[concept_id].append(class_id)

    # map predictions over the set of synonyms back to the smaller set of classes
    # by taking the max or average syn
    concept_logits = logits.new_zeros((logits.shape[0], num_concepts))
    for concept_id, class_ids in enumerate(concept_to_class):
        # logits for class i is the maximum logit of synonym logits
        if op == "mean":
            concept_logits[:, concept_id] = torch.mean(logits[:, class_ids], dim=1)
        elif op == "max":
            concept_logits[:, concept_id] = torch.max(logits[:, class_ids], dim=1).values
        else:
            raise ValueError(f"Unknown {op=}, must be 'max' or 'mean'.")
    return concept_logits
