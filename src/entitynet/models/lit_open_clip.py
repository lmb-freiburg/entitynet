import torch
from lightning.pytorch.core.optimizer import LightningOptimizer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from loguru import logger
from torch import nn
from torch.optim.lr_scheduler import LRScheduler

from visiontext.distutils import WorldInfo

import open_clip
from entitynet.config.main_config import Config
from entitynet.config.model_config import ClipModelCfg, ModelFactoryC
from entitynet.config.task_config import ClipContrastiveTaskCfg
from entitynet.loralib.utils import apply_lora, mark_only_lora_as_trainable
from entitynet.loss_ext import get_init_logits_for_loss_name
from entitynet.models.base_model import LitBaseModel, log_batch_images
from entitynet.models.clip_misc_utils import HF_HUB_PREFIX, process_clip_model_name
from entitynet.models.clip_param_groups import get_clip_param_groups
from entitynet.models.tokenizer_factory import build_tokenizer_from_config
from entitynet.tasks.contrastive_task import ContrastiveRetrievalTask
from open_clip import CLIP, ClipLoss
from open_clip.loss import SigLipLoss


class LitOpenClip(LitBaseModel):

    def get_param_groups_dict(self):
        param_groups_dict = get_clip_param_groups(self.model, self.config)
        return param_groups_dict

    def __init__(self, config: Config):
        """
        Create the clip model given the config.
        """
        super().__init__()
        self.automatic_optimization = False

        # load config
        cm: ClipModelCfg = config.model
        if config.train_task is not None:
            train_task_cfg: ClipContrastiveTaskCfg = config.train_task
            assert train_task_cfg.loss_name == cm.model_loss_name, (
                f"Misconfiguration: {train_task_cfg.loss_name=} != {cm.model_loss_name=}"
                f" but they must be the same"
            )
        self.accum_steps = config.trainer.accum_steps
        self.reset_accum()

        model_name, pretrained = process_clip_model_name(cm.model_ident)
        init_logit_scale, init_logit_bias = get_init_logits_for_loss_name(cm.model_loss_name)

        logger.info(f"Load model: {model_name=} {pretrained=}")
        if cm.model_factory == ModelFactoryC.OPEN_CLIP:
            update_text_cfg_dict = dict(context_length=cm.context_length)
            hf_load_text_separately = False
            if cm.hf_text_encoder_name is not None:
                # this can be used to frankenmerge two different towers together
                cm.force_custom_text = True
                cm.hf_text_encoder_name = cm.hf_text_encoder_name.removeprefix(HF_HUB_PREFIX)
                update_text_cfg_dict["hf_model_name"] = cm.hf_text_encoder_name
                # update_text_cfg_dict["tokenizer_name"] = cm.tokenizer_name
                hf_load_text_separately = True
            model: CLIP = open_clip.create_model(
                model_name,
                # e.g. "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
                pretrained,  # if hugging face, None
                force_patch_dropout=cm.force_patch_dropout,
                force_custom_text=cm.force_custom_text,
                update_text_cfg_dict=update_text_cfg_dict,
                hf_load_text_separately=hf_load_text_separately,
                resize_text_pos_emb=cm.resize_text_pos_emb,
                init_logit_scale=init_logit_scale,
                init_logit_bias=init_logit_bias,
                weights_only=cm.weights_only,
                model_loss_name=cm.model_loss_name,
                strict=cm.ckpt_loading_strict,
            )
        else:
            raise ValueError(f"Unknown {cm.model_factory=}, options: {ModelFactoryC.values_list()}")
        model.output_dict = True

        # verify model has set embed_dim attribute
        try:
            embed_dim = model.embed_dim
        except AttributeError as e:
            raise AttributeError(
                f"Model {model_name=} {type(model)=} does not have embed_dim attribute. "
                f"Go to the model __init__ code and add self.embed_dim=embed_dim  as first line."
            ) from e
        # apply lora if configured
        use_lora = False
        if cm.lora_cfg is not None and cm.lora_cfg.use_lora:
            use_lora = True
            _ = apply_lora(args=cm.lora_cfg, clip_model=model)
            mark_only_lora_as_trainable(model)
            # self.print_params_grads()

        # lock towers if configured
        if cm.lock_image_encoder:
            # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
            model.lock_image_tower(
                unlocked_groups=cm.lock_image_unlocked_groups,
                freeze_bn_stats=cm.lock_image_freeze_bn_stats,
            )
        if cm.lock_text_encoder:
            model.lock_text_tower(
                unlocked_layers=cm.lock_text_unlocked_layers,
                freeze_layer_norm=cm.lock_text_freeze_layer_norm,
            )
        tokenizer = build_tokenizer_from_config(cm)

        self.model = model
        self.tokenizer = tokenizer
        self.embed_dim = embed_dim
        self.use_lora = use_lora
        self.model_name = model_name
        self.pretrained = pretrained
        self.context_length = cm.context_length

        self.epoch_identifier = "pretrained"
        self.loss: nn.Module | None = None
        self.config = config

    def setup(self, stage: str) -> None:
        """
        Create the clip loss.

        This initialization is called after all the multi-gpu setup is done.
        So only here the values for rank and world_size are correct, and only here we can
        create the loss object properly.
        """
        wi = WorldInfo(self.get_trainer_if_attached())
        wi.print_with_rank(f"LitOpenClip setup called")
        if self.train_task is None:
            logger.warning("No train task config found, skipping loss initialization")
            return
        assert isinstance(
            self.train_task, ContrastiveRetrievalTask
        ), f"Unknown task type: {self.train_task}"
        tcfg: ClipContrastiveTaskCfg = self.train_task.task_cfg
        if tcfg.loss_name == "clip":
            self.loss = ClipLoss(
                rank=wi.global_rank,
                world_size=wi.world_size,
                local_loss=tcfg.loss_local,
                gather_with_grad=tcfg.loss_gather_with_grad,
            )
        elif tcfg.loss_name == "siglip":
            self.loss = SigLipLoss(
                rank=wi.global_rank,
                world_size=wi.world_size,
            )
        else:
            raise ValueError(f"Unknown loss: {tcfg.loss_name}")
        wi.print_with_rank(f"Created loss: {tcfg.loss_name} {type(self.loss)}")

    def encode_image(self, image, normalize: bool = False) -> torch.Tensor:
        return self.model.encode_image(image, normalize=normalize)

    def encode_image_batched(
        self, images, normalize: bool = False, batch_size: int = 64
    ) -> torch.Tensor:
        encoded_batches = []
        # Loop over the input batch in chunks along dimension 0.
        for i in range(0, images.shape[0], batch_size):
            batch = images[i : i + batch_size]
            encoded_batch = self.encode_image(batch, normalize=normalize)
            encoded_batches.append(encoded_batch)
        # Concatenate all encoded chunks along the first dimension.
        return torch.cat(encoded_batches, dim=0)

    def tokenize_text(self, text: list[str]):
        return self.tokenizer(text, context_length=self.context_length)

    def encode_text(self, text, normalize: bool = False):
        return self.model.encode_text(text, normalize=normalize)

    def reset_accum(self):
        self.accum_images, self.accum_tokens, self.accum_features = [], [], {}

    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        self.reset_accum()
        if self.use_lora:
            self.model.train()
            mark_only_lora_as_trainable(self.model)

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        wi = WorldInfo(self.trainer)
        batch_size = batch["image"].shape[0]
        self.loss.is_val = False

        # prepare tokens from text in the batch
        if "text_list" in batch:
            # multiple text inputs per image
            token_list = []
            for text in batch["text_list"]:
                token_list.append(self.tokenize_text(text))
            t = torch.stack(token_list).to(batch["image"].device)  # [B, n_text, context_length]
            batch["tokens_list"] = t
            # fold multiple texts into batch dimension, tokens: (B, N_t, D) -> (B*N_t, D)
            batch["tokens"] = batch["tokens_list"].view(-1, batch["tokens_list"].size(-1))
        else:
            # regular training with 1 image-text pair
            batch["tokens"] = self.tokenize_text(batch["text"]).to(batch["image"].device)

        # logging - log_data_locally ensures no training data is logged to the cloud
        # this includes using the stdout logger which is captured by e.g. wandb
        self.training_step_log_lr()
        if batch_idx == 0 and wi.is_global_zero and not self.config.trainer.log_data_locally:
            log_batch_input(batch)
        if batch_idx == 0 and wi.is_global_zero and self.current_epoch == 0:
            train_set_transform = getattr(self.train_dataset, "transform", None)
            log_batch_images(
                batch,
                self.logger,
                train_set_transform,
                self.config.trainer.n_images_to_save,
                self.config.trainer.log_data_locally,
                self.config.trainer.output_dir,
            )

        if self.accum_steps == 1:  # ---------- regular optimization without batch accumulation
            out_dict = self.model(image=batch["image"], text=batch["tokens"])
            if batch_idx == 0 and wi.is_global_zero:
                log_batch_output(out_dict)

            loss = self.loss(**out_dict, output_dict=False)
            if batch_idx == 0 and self.current_epoch == 0:
                wi.print_with_rank(f"loss={repr(loss)}")
            # self.print_params_grads()  # debugging

            self.manual_backward(loss)
            if self.config.trainer.check_for_nans:
                self.check_for_nans()

            if self.config.optimizer.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.clip_grad_norm
                )
                self.log("train_grad_norm", grad_norm, on_step=True, on_epoch=False)

            opt: LightningOptimizer = self.optimizers()
            opt.step()
            opt.zero_grad()
            lr_sched: LRScheduler = self.lr_schedulers()
            lr_sched.step()
            self.log("train_loss", loss, batch_size=batch_size, on_step=True, on_epoch=False)
            if wi.is_global_zero:
                self.log(
                    "train_logit_scale",
                    out_dict["logit_scale"],
                    batch_size=batch_size,
                    on_step=True,
                    on_epoch=False,
                    rank_zero_only=True,
                )
                if "logit_bias" in out_dict:
                    self.log(
                        "train_logit_bias",
                        out_dict["logit_bias"],
                        batch_size=batch_size,
                        on_step=True,
                        on_epoch=False,
                        rank_zero_only=True,
                    )
            return

        # ---------- accumulate gradients over multiple steps
        if batch_idx == 0 and self.current_epoch == 0 and wi.is_global_zero:
            logger.info(f"Running with accum_steps code path")
        # see open_clip/training/train.py
        # first, cache features without gradient tracking

        # Using torch.no_grad() would be faster but it reduces the accuracy a lot.
        # Debugged this but couldn't find the problem:
        # - the batch input stays the same (so no inplace modification of the data)
        # - no gradients are created (the no_grad works as indendent)
        # - running 5 forward or 1 forward inside no_grad breaks in the same way.
        # - model state_dict is the same before and after that forward_pass
        # - model.training is always True (train mode). model.eval doesn't help
        # - running it with gradient: the bug doesn't appear!
        # - weirdly doing torch.inference_mode() raises an error!
        # solution: keep gradients enabled all the time even when they are not used.
        # vvvvvvvvvvvvvvvvvvvvvvvvvvvvv
        # torch.set_grad_enabled(False)  # this reduces accuracy alot, probably a lightning bug.
        out_dict: dict = self.model(batch["image"], batch["tokens"])
        for f in ("logit_scale", "logit_bias", "logit_scale2"):
            if f in out_dict.keys():
                _to_remove = out_dict.pop(f, None)
        for key, val in out_dict.items():
            val = val.detach()
            if key in self.accum_features:
                self.accum_features[key].append(val)
            else:
                self.accum_features[key] = [val]
        self.accum_images.append(batch["image"].detach())
        self.accum_tokens.append(batch["tokens"].detach())
        del _to_remove, out_dict
        torch.set_grad_enabled(True)

        if (batch_idx + 1) % self.accum_steps == 0:
            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other
            # batches as negatives. Call backwards each time, but only step optimizer at the end.
            total_loss = 0
            for j in range(self.accum_steps):
                images = self.accum_images[j]
                tokens = self.accum_tokens[j]
                out_dict = self.model(images, tokens)
                if j == 0 and wi.is_global_zero:
                    # logit scale depends on model weights, so only logging it once per accum
                    # and only on GPU 0 is sufficient
                    self.log(
                        "train_logit_scale",
                        out_dict["logit_scale"],
                        batch_size=batch_size,
                        on_step=True,
                        on_epoch=False,
                        rank_zero_only=True,
                    )
                    if "logit_bias" in out_dict:
                        self.log(
                            "train_logit_bias",
                            out_dict["logit_bias"],
                            batch_size=batch_size,
                            on_step=True,
                            on_epoch=False,
                            rank_zero_only=True,
                        )
                inputs_no_accum = {"logit_scale": out_dict.pop("logit_scale")}
                if "logit_bias" in out_dict:
                    inputs_no_accum["logit_bias"] = out_dict.pop("logit_bias")

                inputs = {}
                for key, val in self.accum_features.items():
                    accumulated = self.accum_features[key]
                    inputs[key] = torch.cat(
                        accumulated[:j] + [out_dict[key]] + accumulated[j + 1 :]
                    )
                loss = self.loss(**inputs, **inputs_no_accum, output_dict=False)
                if self.current_epoch == 0 and batch_idx + 1 == self.accum_steps:
                    wi.print_with_rank(f"accumulated loss {j}/{self.accum_steps}: {repr(loss)}")
                del inputs
                del inputs_no_accum

                self.manual_backward(loss)
                # self.check_for_nans()
                total_loss += loss

            if self.config.optimizer.clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.clip_grad_norm
                )
                self.log("train_grad_norm", grad_norm, on_step=True, on_epoch=False)
            opt: LightningOptimizer = self.optimizers()
            opt.step()
            opt.zero_grad()
            lr_sched: LRScheduler = self.lr_schedulers()
            if lr_sched is not None:
                lr_sched.step()
            self.log("train_loss", total_loss / self.accum_steps, batch_size=batch_size)
            self.reset_accum()


def log_batch_input(batch):
    image, tokens = batch["image"], batch["tokens"]
    if "text_list" in batch:
        text_list = batch["text_list"]
        text_str = f"{len(text_list)=} {len(text_list[0])=} first 2: {text_list[:2]}"
    else:
        text = batch["text"]
        if len(text) > 20:
            text_str = f"{text[:20]} ... ({len(text)} total texts)"
        else:
            text_str = f"{text}"
    logger.info(f"First batch, {image.shape=} {text_str=} {tokens.shape=}")


def log_batch_output(out_dict):
    out_dict_str = ", ".join([f"{k}={v.shape}" for k, v in out_dict.items()])
    logger.info(f"First batch, out_dict: {out_dict_str}")
