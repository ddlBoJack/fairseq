# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from omegaconf import II

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from fairseq.modules import EMAModule, EMAModuleConfig, PositionalEmbedding, FairseqDropout
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model
from .wav2vec2 import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
    TransformerEncoder4Text,
)
from fairseq.modules import (
    GradMultiply,
    LayerNorm,
)
from fairseq.utils import index_put
from fairseq import utils


logger = logging.getLogger(__name__)


@dataclass
class Data2VecJtConfig(Wav2Vec2Config):

    loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False
    batch_norm_target_layer: bool = False
    group_norm_target_layer: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer"},
    )
    ema_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )

    max_update: int = II("optimization.max_update")

    min_target_var: float = field(
        default=0.1, metadata={"help": "stop training if target var falls below this"}
    )
    min_pred_var: float = field(
        default=0.01,
        metadata={"help": "stop training if prediction var falls below this"},
    )

    # below are for text(equal to encoder_embed_dim in this version)
    text_embed_dim: int = field(
        default=768, metadata={"help": "embedding dim for text"}
    )
    max_source_positions: int = field(
        default=1024, metadata={"help": "max source sequence length"}
    )
    text_start_transformer_layer: int = field(
        default=0, metadata={"help": "which transformer layer to start loss for text"}
    )
    text_transofmer_layers: int = field(
        default=6, metadata={"help": "how many transformer layers to use for text"}
    )
    text_dropout_input: float = field(
        default=0, metadata={"help": "text input dropout"}
    )
    ctc_start_step: int = field(
        default=0, metadata={"help": "which step to start ctc loss"}
    )
    ctc_loss_alpha: float = field(
        default=1, metadata={"help": "weight for ctc loss"}
    )



def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_jt", dataclass=Data2VecJtConfig)
class Data2VecJtModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecJtConfig, task=None):
        super().__init__()
        self.cfg = cfg

        feature_enc_layers = eval(cfg.conv_feature_layers) # v-ziyangma: [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)], x/320
        self.extractor_embed = feature_enc_layers[-1][0] # v-ziyangma: 512

        self.ema = None
        self.embed = cfg.encoder_embed_dim # v-ziyangma: 768

        self.average_top_k_layers = cfg.average_top_k_layers # v-ziyangma: 8, how many layers to compute the loss on
        self.loss_beta = cfg.loss_beta # v-ziyangma: 0.0
        self.loss_scale = cfg.loss_scale # v-ziyangma: None

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=cfg.extractor_mode,
            conv_bias=cfg.conv_bias,
        ) # v-ziyangma: from wav2vec2

        self.post_extract_proj = nn.Linear(self.extractor_embed, cfg.encoder_embed_dim) # v-ziyangma: after x/320, proj with 512 -> 768 for transformer

        self.mask_prob = cfg.mask_prob # v-ziyangma: 0.65
        self.mask_selection = cfg.mask_selection # v-ziyangma: static; todo: dynamic
        self.mask_other = cfg.mask_other # v-ziyangma: 0.0
        self.mask_length = cfg.mask_length # v-ziyangma: 10
        self.no_mask_overlap = cfg.no_mask_overlap # v-ziyangma: False
        self.mask_min_space = cfg.mask_min_space # v-ziyangma: 1

        # v-ziyangma: no channel masking by default
        self.mask_channel_prob = cfg.mask_channel_prob # v-ziyangma: 0.0
        self.mask_channel_before = cfg.mask_channel_before # v-ziyangma: False
        self.mask_channel_selection = cfg.mask_channel_selection # v-ziyangma: static
        self.mask_channel_other = cfg.mask_channel_other # v-ziyangma: 0.0
        self.mask_channel_length = cfg.mask_channel_length # v-ziyangma: 10
        self.no_mask_channel_overlap = cfg.no_mask_channel_overlap # v-ziyangma: False
        self.mask_channel_min_space = cfg.mask_channel_min_space # v-ziyangma: 1

        self.dropout_input = nn.Dropout(cfg.dropout_input) # v-ziyangma: 0.0
        self.dropout_features = nn.Dropout(cfg.dropout_features) # v-ziyangma: 0.0

        self.feature_grad_mult = cfg.feature_grad_mult # v-ziyangma: 1.0

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(cfg.encoder_embed_dim).uniform_()
        ) # v-ziyangma: torch.Size([768]), from a uniform distribution

        self.encoder = TransformerEncoder(cfg) # v-ziyangma: from wav2vec2
        self.layer_norm = LayerNorm(self.extractor_embed) # v-ziyangma: LayerNorm(512)

        self.final_proj = nn.Linear(self.embed, self.embed) # v-ziyangma: p768 -> 768

        self.num_updates = 0

        # below are for text module
        # self.task = task
        if task.__class__.__name__ == "Data2vevJtPretrainingTask":
            self.blank_idx = (
                task.target_dictionary.index(task.blank_symbol)
                if hasattr(task, "blank_symbol")
                else 0)
            self.source_dictionary = task.source_dictionary
            self.target_dictionary = task.target_dictionary

            self.text_embed_tokens = nn.Embedding(len(self.source_dictionary), cfg.text_embed_dim, self.source_dictionary.pad())
            self.text_embed_positions = PositionalEmbedding(
                cfg.max_source_positions,
                cfg.text_embed_dim,
                self.source_dictionary.pad(),
                learned=True)
            # self.text_layernorm_embedding = LayerNorm(cfg.text_embed_dim)
            # self.text_dropout_module = nn.Dropout(cfg.text_dropout_input)

            self.text_post_extract_proj = nn.Linear(cfg.text_embed_dim, cfg.text_embed_dim)
            self.text_proj_d = nn.Linear(cfg.text_embed_dim, len(self.target_dictionary))
            self.text_encoder = TransformerEncoder4Text(cfg)

    def forward_embedding(self, src_tokens):
        x = self.text_embed_tokens(src_tokens) + self.text_embed_positions(src_tokens)
        # x = self.text_layernorm_embedding(x)
        # x = self.text_dropout_module(x)
        return x

    def make_ema_teacher(self):
        # v-ziyangma: "We found it more efficient and slightly more accurate to share the parameters of the feature encoder and the positional encoder between the teacher and student networks."
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_layers_only:
            self.cfg.ema_transformer_only = True
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")
                # skip_keys：              
                # 'pos_conv.1.0.weight'
                # 'pos_conv.3.0.bias'
                # 'pos_conv.4.0.weight'
                # 'pos_conv.0.0.bias'
                # 'pos_conv.2.0.bias'
                # 'pos_conv.4.0.bias'
                # 'pos_conv.2.0.weight'
                # 'pos_conv.0.0.weight'
                # 'pos_conv.1.0.bias'
                # 'pos_conv.3.0.weight'

        self.ema = EMAModule(
            self.encoder if self.cfg.ema_transformer_only else self,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None: # v-ziyangma: finetuning mode got self.final_proj is None.
            logger.info(f"making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema.set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.encoder if self.cfg.ema_transformer_only else self)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecJtConfig, task=None):
        """Build a new model instance."""

        return cls(cfg, task)

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=1,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.cfg.require_same_masks,
                    mask_dropout=self.cfg.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb) # v-ziyangma: WHY using mask_emb(nn.Parameter)?
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            return torch.floor((input_length - kernel_size) / stride + 1)

        conv_cfg_list = eval(self.cfg.conv_feature_layers)

        for i in range(len(conv_cfg_list)):
            input_lengths = _conv_out_length(
                input_lengths, conv_cfg_list[i][1], conv_cfg_list[i][2]
            )

        return input_lengths.to(torch.long)

    def forward(
        self,
        source,
        source_label=None,
        target_label=None,
        padding_mask=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        features = source # v-ziyangma: batch_size x seq_len

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(features) # v-ziyangma: batch_size x feature_dim x seq_len(downsampled)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult) # v-ziyangma: scale the backward gradient
        else:
            with torch.no_grad():
                features = self.feature_extractor(features)

        features = features.transpose(1, 2) # # v-ziyangma: batch_size x seq_len x feature_dim

        features = self.layer_norm(features)

        orig_padding_mask = padding_mask

        if padding_mask is not None and padding_mask.any():
            input_lengths = (1 - padding_mask.long()).sum(-1)
            # apply conv formula to get real output_lengths
            output_lengths = self._get_feat_extract_output_lengths(input_lengths)

            padding_mask = torch.zeros(
                features.shape[:2], dtype=features.dtype, device=features.device
            )

            # these two operations makes sure that all values
            # before the output lengths indices are attended to
            padding_mask[
                (
                    torch.arange(padding_mask.shape[0], device=padding_mask.device),
                    output_lengths - 1,
                )
            ] = 1
            padding_mask = (1 - padding_mask.flip([-1]).cumsum(-1).flip([-1])).bool()
        else:
            padding_mask = None

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features) # v-ziyangma: 512 -> 768

        pre_encoder_features = None # v-ziyangma: unmasked features input of the teacher.
        if self.cfg.ema_transformer_only:
            pre_encoder_features = features.clone()

        features = self.dropout_input(features)

        if mask:
            x, mask_indices = self.apply_mask(
                features,
                padding_mask,
                mask_indices=mask_indices,
                mask_channel_indices=mask_channel_indices,
            ) # v-ziyangma: x: batch_size x seq_len x feature_dim, mask_indices: batch_size x seq_len with True/False
        else:
            x = features
            mask_indices = None

        x, layer_results = self.encoder(
            x,
            padding_mask=padding_mask,
            layer=layer,
        )

        if features_only:
            return {
                "x": x,
                "padding_mask": padding_mask,
                "layer_results": layer_results,
            }

        result = {
            "losses": {},
        }

        with torch.no_grad():
            self.ema.model.eval()

            if self.cfg.ema_transformer_only:
                y, layer_results = self.ema.model.extract_features(
                    pre_encoder_features,
                    padding_mask=padding_mask,
                    min_layer=self.cfg.encoder_layers - self.average_top_k_layers,
                ) # v-ziyangma: layer_results return top_k_layers' results
                y = {
                    "x": y,
                    "padding_mask": padding_mask,
                    "layer_results": layer_results,
                }
            else:
                y = self.ema.model.extract_features(
                    source=source,
                    padding_mask=orig_padding_mask,
                    mask=False,
                )

            target_layer_results = [l[2] for l in y["layer_results"]] # v-ziyangma: layer_results is layers before dropout3, residual, final_layer_norm.

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    tl.permute(1, 2, 0) for tl in target_layer_results  # TBC -> BCT
                ]
                permuted = True

            if self.cfg.batch_norm_target_layer:
                target_layer_results = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in target_layer_results
                ]

            if self.cfg.instance_norm_target_layer:
                target_layer_results = [
                    F.instance_norm(tl.float()) for tl in target_layer_results
                ]

            if permuted:
                target_layer_results = [
                    tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
                ]

            if self.cfg.group_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-2:])
                    for tl in target_layer_results
                ]

            if self.cfg.layer_norm_target_layer:
                target_layer_results = [
                    F.layer_norm(tl.float(), tl.shape[-1:])
                    for tl in target_layer_results
                ]

            y = sum(target_layer_results) / len(target_layer_results)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.float().transpose(1, 2)).transpose(1, 2)

            if not permuted:
                y = y.transpose(0, 1)

            y = y[mask_indices] # v-ziyangma: total_length x feature_dim

        x = x[mask_indices]
        x = self.final_proj(x)

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)
        
        speech_loss = loss.sum() * scale
        result["loss_speech"] = speech_loss
        result["losses"]["regression"] = speech_loss

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

        # below are for text ctc loss
        if self.num_updates >= self.cfg.ctc_start_step and source_label is not None and target_label is not None:
            text_padding_mask = source_label.eq(self.source_dictionary.pad())
            has_pads = text_padding_mask.any()
            # self.check_label_range(source_label, self.source_dictionary)
            text_x = self.forward_embedding(source_label)
            if has_pads:
                text_x = text_x * (1 - text_padding_mask.unsqueeze(-1).type_as(text_x))
            text_x = self.text_post_extract_proj(text_x)
            
            # text encoder
            text_x, _ = self.text_encoder(
            text_x,
            padding_mask=text_padding_mask,
            layer=None,
            )
            # share the same transformer with speech
            text_x, _ = self.encoder.forward_text(
            text_x,
            padding_mask=text_padding_mask,
            tgt_layer=None, 
            min_layer=12,
            start_layer=self.cfg.text_start_transformer_layer
            )
            text_x = text_x.transpose(0, 1)
            text_x = self.text_proj_d(text_x)
            
            # comute ctc loss
            lprobs = self.get_normalized_probs(
            text_x, 
            text_padding_mask,
            log_probs=True
            ).contiguous()  # (T, B, C) from the encoder
            
            if text_padding_mask is not None:
                non_padding_mask = ~text_padding_mask
                source_lengths = non_padding_mask.long().sum(-1)
            else:
                source_lengths = lprobs.new_full(
                    (lprobs.size(1),), lprobs.size(0), dtype=torch.long
                )
            
            target_pad_mask = (target_label != self.target_dictionary.pad()) & (target_label != self.target_dictionary.eos())
            targets_flat = target_label.masked_select(target_pad_mask)
            target_lengths = target_pad_mask.sum(-1)
            
            # with torch.backends.cudnn.flags(enabled=False):
            with torch.backends.cudnn.flags(deterministic=True):
                ctc_loss = F.ctc_loss(
                    lprobs,
                    targets_flat,
                    source_lengths,
                    target_lengths,
                    blank=self.blank_idx,
                    reduction="mean",
                    zero_infinity=True,
                )
            # ctc_loss = ctc_loss / target_lengths.sum().item()
            ctc_loss = self.cfg.ctc_loss_alpha * ctc_loss * result["sample_size"]
            result["ctc_loss"] =  ctc_loss
            result["losses"]["regression"] = result["losses"]["regression"] + ctc_loss

        with torch.no_grad():
            result["target_var"] = self.compute_var(y)
            result["pred_var"] = self.compute_var(x.float())

        if self.num_updates > 5000 and result["target_var"] < self.cfg.min_target_var:
            logger.error(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
            raise Exception(
                f"target var is {result['target_var'].item()} < {self.cfg.min_target_var}, exiting"
            )
        if self.num_updates > 5000 and result["pred_var"] < self.cfg.min_pred_var:
            logger.error(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )
            raise Exception(
                f"pred var is {result['pred_var'].item()} < {self.cfg.min_pred_var}, exiting"
            )

        if self.ema is not None:
            result["ema_decay"] = self.ema.get_decay() * 1000

        return result

    @staticmethod
    def compute_var(y):
        y = y.view(-1, y.size(-1))
        if dist.is_initialized():
            zc = torch.tensor(y.size(0)).cuda()
            zs = y.sum(dim=0)
            zss = (y ** 2).sum(dim=0)

            dist.all_reduce(zc)
            dist.all_reduce(zs)
            dist.all_reduce(zss)

            var = zss / (zc - 1) - (zs ** 2) / (zc * (zc - 1))
            return torch.sqrt(var + 1e-6).mean()
        else:
            return torch.sqrt(y.var(dim=0) + 1e-6).mean()

    def extract_features(
        self, source, padding_mask, mask=False, layer=None
    ):
        res = self.forward(
            source,
            padding_mask,
            mask=mask,
            features_only=True,
            layer=layer,
        )
        return res

    def remove_pretraining_modules(self, last_layer=None):
        self.final_proj = None
        self.ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )
        
        # remove text modules
        self.text_embed_tokens = None
        self.text_embed_positions = None
        self.text_layernorm_embedding = None
        self.text_dropout_module = None
        self.text_post_extract_proj = None
        self.text_proj_d = None # TODO: what about init proj_d when finetuning with text_proj_d?

    def get_logits(self, logits, padding_mask, normalize=False):
        if padding_mask is not None and padding_mask.any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[padding_mask.T] = masking_tensor.type_as(logits)

        if normalize:
            logits = utils.log_softmax(logits.float(), dim=-1)

        return logits

    def get_normalized_probs(self, logits, padding_mask, log_probs):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(logits, padding_mask)

        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1) 

    def check_label_range(self, label, dictionary):
        maxx = label.max().item()
        print(f"max label is {maxx}")
        minn = label.min().item()
        print(f"min label is {minn}")
        for line in label:
            for token in line:
                if token not in range(len(dictionary)):
                    raise Exception(
                        f"Token {token} not in dictionary {dictionary}"
                    )