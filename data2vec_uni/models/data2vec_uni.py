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

from fairseq.modules import EMAModule, EMAModuleConfig
from fairseq.data.data_utils import compute_mask_indices
from fairseq.models import BaseFairseqModel, register_model, FairseqEncoder
from fairseq.models.wav2vec import (
    ConvFeatureExtractionModel,
    Wav2Vec2Config,
    TransformerEncoder,
)
from fairseq.modules import (
    GradMultiply,
    LayerNorm,
)
from fairseq.utils import index_put

from fairseq.models.transformer import TransformerConfig
from fairseq.models.transformer import TransformerEncoder as TextTransformerEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from .cross_model_ema_module import CrossModelEMAModule


logger = logging.getLogger(__name__)


@dataclass
class Data2VecUniConfig(Wav2Vec2Config):

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

    speech_pretrained_model: bool = field(
    default=False, 
    metadata={"help": "whether to use pretrained speech model"}
    )
    speech_model_path: Optional[str] = field(
        default=None, metadata={"help": "path to pretrained speech model"}
    )

    ########## below are for text teacher ##########
    text_model_path: Optional[str] = field(
        default=None, metadata={"help": "path to pretrained text model"}
    )

    text_max_positions: int = field(
        default=512, metadata={"help": "max sequence length"}
    )

    text_head_layers: int = 1

    text_transformer: TransformerConfig = TransformerConfig()

    text_loss_beta: float = field(
        default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"}
    )
    text_loss_scale: Optional[float] = field(
        default=None,
        metadata={
            "help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"
        },
    )
    text_average_top_k_layers: int = field(
        default=8, metadata={"help": "how many layers to average"}
    )

    text_layer_norm_target_layer: bool = False
    text_instance_norm_target_layer: bool = False
    text_batch_norm_target_layer: bool = False
    text_instance_norm_targets: bool = False
    text_layer_norm_targets: bool = False

    text_ema_decay: float = field(
        default=0.999, 
        metadata={"help": "initial ema decay rate"}
        )
    text_ema_end_decay: float = field(
        default=0.9999, metadata={"help": "final ema decay rate"}
    )

    # when to finish annealing ema decay rate
    text_ema_anneal_end_step: int = II("optimization.max_update")

    text_ema_transformer_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )
    text_teacher: bool = field(
        default=False, metadata={"help": "whether to use text teacher"}
        )

    text_init_transformer: bool = field(
        default=False, 
        metadata={"help": "whether to init the transformer of the text model"}
        )
    
    text_do_ema: bool = field(
        default=True, 
        metadata={"help": "whether to use ema"}
        )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_uni", dataclass=Data2VecUniConfig)
class Data2VecUniModel(BaseFairseqModel):
    def __init__(self, cfg: Data2VecUniConfig, text_encoder):
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

        # for text model
        self.text_encoder = text_encoder
        self.text_ema = None

        # load pretrained speech model if specified
        if cfg.speech_pretrained_model:
            logger.info(f"loading pretrained speech model from {cfg.speech_model_path} ...")
            state_dict=torch.load(cfg.speech_model_path)
            self.load_state_dict(state_dict["model"], strict=False)


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
    
    def make_text_ema_teacher(self):
        ema_config = EMAModuleConfig(
            ema_decay=self.cfg.text_ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.text_ema_transformer_layers_only:
            for k, _ in self.text_encoder.sentence_encoder.embed_positions.named_parameters():
                skip_keys.add(f"embed_tokens.{k}")
            for k, _ in self.text_encoder.sentence_encoder.embed_positions.named_parameters():
                skip_keys.add(f"embed_positions.{k}")
            if self.text_encoder.sentence_encoder.layernorm_embedding is not None:
                for (
                    k,
                    _,
                ) in self.text_encoder.sentence_encoder.layernorm_embedding.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")
            if self.text_encoder.sentence_encoder.layer_norm is not None:
                for k, _ in self.text_encoder.sentence_encoder.layer_norm.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")
                    self.text_encoder.text_ema
            
            for k, _ in self.encoder.pos_conv.named_parameters():
                skip_keys.add(f"pos_conv.{k}")
            if self.encoder.layer_norm is not None:
                for k, _ in self.encoder.layer_norm.named_parameters():
                    skip_keys.add(f"layer_norm.{k}")

        self.text_ema = CrossModelEMAModule(
            self.text_encoder.sentence_encoder,
            ema_config,
            skip_keys=skip_keys,
        )

        # del self.text_encoder.sentence_encoder  #TODO: del the sentence_encoder to save memory

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.final_proj is not None:
            logger.info(f"making speech ema teacher")
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
        

        if self.cfg.text_teacher and self.cfg.text_do_ema:
            if self.text_ema is None and self.text_encoder.regression_head is not None:
                logger.info(f"making text ema teacher")
                self.make_text_ema_teacher()
            elif self.training and self.ema is not None:
                if self.cfg.text_ema_decay != self.cfg.text_ema_end_decay:
                    if num_updates >= self.cfg.text_ema_anneal_end_step:
                        text_decay = self.cfg.text_ema_end_decay
                    else:
                        text_decay = get_annealed_rate(
                        self.cfg.text_ema_decay,
                        self.cfg.text_ema_end_decay,
                        num_updates,
                        self.cfg.text_ema_anneal_end_step,
                        )
                    self.text_ema.set_decay(text_decay)
                if self.text_ema.get_decay() < 1:
                    self.text_ema.step(self.encoder)

        self.num_updates = num_updates

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        # using when save_checkpoint in `state_dict = utils.move_to_cpu(self.state_dict())` and `"model": self.model.state_dict()` in trainer.py
        state = super().state_dict(destination, prefix, keep_vars)

        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params
        if self.text_ema is not None:
            state[prefix + "_text_ema"] = self.text_ema.fp32_params

        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        if self.text_ema is not None:
            k = prefix + "_text_ema"
            assert k in state_dict
            self.text_ema.restore(state_dict[k], True)
            del state_dict[k]

        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    @classmethod
    def build_model(cls, cfg: Data2VecUniConfig, task=None):
        """Build a new model instance."""
        if cfg.text_teacher: 

            if cfg.text_model_path is not None:
                text_encoder = Data2VecTextEncoder(cfg, task.target_dictionary)
            
            pretrained_text_model = torch.load(cfg.text_model_path)
            state_dict = text_encoder.state_dict()
            text_model_key = [name for name in text_encoder.state_dict()]
            logger.info(f"loading text model from {cfg.text_model_path} ...")
            if cfg.text_init_transformer:
                pass # TODO: init the embed_tokens.weight and embed_positions.weight from the pretrained model.
            else:
                for key in pretrained_text_model["model"].keys():
                    local_key = ".".join(key.split(".")[1:])
                    if local_key in state_dict:
                        # logger.info(f"loading {local_key} from {key} in {cfg.text_model_path}")
                        state_dict[local_key].copy_(pretrained_text_model["model"][key])
                        text_model_key.remove(local_key)
                    else:
                        logger.info(f"skipping key: {key} in {cfg.text_model_path}")
                for key in text_model_key:
                    logger.info(f"initializing key: {key} in {text_encoder.__class__.__name__}")
        
        else:
            text_encoder = None

        
        return cls(cfg, text_encoder)

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
        padding_mask=None,
        target=None,
        meta=None,
        target_lengths=None,
        ntokens=None,
        mask=True,
        features_only=False,
        layer=None,
        mask_indices=None,
        mask_channel_indices=None,
        padding_count=None,
    ):
        text_teacher = False if target == None else self.cfg.text_teacher # deside whether to use text teacher by the input tpye

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


            # below are for text teacher model
            if text_teacher: 
                self.text_ema.model.eval()
                text_encoder_out = self.text_ema.model(
                    target,
                    return_all_hiddens=True,
                )
                text_y = text_encoder_out["fc_results"]
                text_y = text_y[-self.text_encoder.average_top_k_layers :]

                permuted = False
                if self.cfg.text_instance_norm_target_layer or self.cfg.text_batch_norm_target_layer:
                    text_y = [tl.permute(1, 2, 0) for tl in text_y]  # TBC -> BCT
                    permuted = True

                if self.cfg.text_batch_norm_target_layer:
                    text_y = [
                        F.batch_norm(
                            tl.float(), running_mean=None, running_var=None, training=True
                        )
                        for tl in text_y
                    ]

                if self.cfg.text_instance_norm_target_layer:
                    text_y = [F.instance_norm(tl.float()) for tl in text_y]

                if permuted:
                    text_y = [tl.transpose(1, 2) for tl in text_y]  # BCT -> BTC

                if self.cfg.text_layer_norm_target_layer:
                    text_y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in text_y]

                text_y = sum(text_y) / len(text_y)

                if not permuted:
                    text_y = text_y.transpose(0, 1)

                if self.cfg.text_layer_norm_targets:
                    text_y = F.layer_norm(text_y.float(), text_y.shape[-1:])

                if self.cfg.text_instance_norm_targets:
                    text_y = F.instance_norm(text_y.transpose(1, 2)).transpose(1, 2)

                # do upsampling with text_y and meta
                text_upsampling_list = []
                for sentence, sentence_meta in zip(text_y, meta):
                    sentence_upsampling = torch.cat(
                        [token_.repeat(meta_,1) for token_ ,meta_ in zip(sentence, sentence_meta)], dim=0
                    )
                    if len(sentence_upsampling) < len(mask_indices[-1]):
                        sentence_upsampling = torch.cat(
                            [sentence_upsampling, sentence_upsampling.new_zeros(len(mask_indices[-1]) - len(sentence_upsampling), text_y.size(-1))], dim=0
                        ).unsqueeze(0)
                    else:
                        sentence_upsampling = sentence_upsampling[:len(mask_indices[-1])].unsqueeze(0)
                    text_upsampling_list.append(sentence_upsampling)
                text_upsampling = torch.cat(text_upsampling_list, dim=0)
                assert (text_upsampling.size() == x.size()), f"text_upsampling.size() = {text_upsampling.size()} does not match x.size() = {x.size()}"
                text_y = text_upsampling[mask_indices]

        # compute loss
        x = x[mask_indices]
        x = self.final_proj(x)
        text_x = self.text_encoder.regression_head(x) if text_teacher else None

        sz = x.size(-1)

        if self.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.loss_beta
            ).sum(dim=-1)
        
        if text_teacher:
            if self.cfg.text_loss_beta == 0:
                text_loss = F.mse_loss(text_x.float(), text_y.float(), reduction="none").sum(dim=-1)
            else:
                text_loss = F.smooth_l1_loss(
                    text_x.float(), text_y.float(), reduction="none", beta=self.cfg.text_loss_beta
                ).sum(dim=-1)

        if self.loss_scale is not None:
            scale = self.loss_scale
        else:
            scale = 1 / math.sqrt(sz)
        
        if text_teacher:
            if self.cfg.text_loss_scale is not None:
                text_scale = self.cfg.text_loss_scale
            else:
                text_scale = 1 / math.sqrt(sz)

        result["losses"]["regression"] = loss.sum() * scale + text_loss.sum() * text_scale if text_teacher else loss.sum() * scale

        if "sample_size" not in result:
            result["sample_size"] = loss.numel()

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
        
        if text_teacher:
            if self.text_ema is not None:
                result["text_ema_decay"] = self.text_ema.get_decay() * 1000

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
        self.text_ema = None
        if last_layer is not None:
            self.encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.layers) if i <= last_layer
            )


class Data2VecTextEncoder(FairseqEncoder):
    def __init__(self, cfg: Data2VecUniConfig, dictionary):
        super().__init__(dictionary)

        self.cfg = cfg
        
        embed_tokens = self.build_embedding(
            len(dictionary), cfg.text_transformer.encoder.embed_dim, dictionary.pad()
        )

        self.sentence_encoder = self.build_encoder(cfg, dictionary, embed_tokens)
        self.mask_idx = dictionary.index("<mask>")
        assert self.mask_idx != dictionary.unk(), dictionary.symbols

        self.ema = None
        self.average_top_k_layers = cfg.text_average_top_k_layers
        self.loss_scale = cfg.text_loss_scale

        assert self.cfg.text_head_layers >= 1

        embed_dim = cfg.text_transformer.encoder.embed_dim
        curr_dim = embed_dim
        projs = []
        for i in range(self.cfg.text_head_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim

        projs.append(nn.Linear(curr_dim, embed_dim))
        self.regression_head = nn.Sequential(*projs)

        self.num_updates = 0

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, cfg, dictionary, embed_tokens):
        encoder = TextTransformerEncoder(cfg.text_transformer, dictionary, embed_tokens, return_fc=True)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    # def state_dict(self, destination=None, prefix="", keep_vars=False):
    #     state = super().state_dict(destination, prefix, keep_vars)
    #     if self.ema is not None:
    #         state[prefix + "_ema"] = self.ema.fp32_params
    #     return state

    # def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
    #     if self.ema is not None:
    #         k = prefix + "_ema"
    #         assert k in state_dict
    #         self.ema.restore(state_dict[k], True)
    #         del state_dict[k]
    #     return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        src_tokens,
        target_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """

        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )

        if features_only:
            return x, extra

        assert target_tokens is not None

        with torch.no_grad():
            # use EMA parameter as the teacher
            self.ema.model.eval()

            encoder_out = self.ema.model(
                target_tokens,
                return_all_hiddens=True,
            )
            y = encoder_out["fc_results"]

            y = y[-self.average_top_k_layers :]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                y = [tl.permute(1, 2, 0) for tl in y]  # TBC -> BCT
                permuted = True

            if self.cfg.batch_norm_target_layer:
                y = [
                    F.batch_norm(
                        tl.float(), running_mean=None, running_var=None, training=True
                    )
                    for tl in y
                ]

            if self.cfg.instance_norm_target_layer:
                y = [F.instance_norm(tl.float()) for tl in y]

            if permuted:
                y = [tl.transpose(1, 2) for tl in y]  # BCT -> BTC

            if self.cfg.layer_norm_target_layer:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

            y = sum(y) / len(y)

            if not permuted:
                y = y.transpose(0, 1)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        masked_indices = src_tokens.eq(self.mask_idx)

        x = x[masked_indices]
        y = y[masked_indices]

        x = self.regression_head(x)

        sz = x.size(-1)
        if self.cfg.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(
                x.float(), y.float(), reduction="none", beta=self.cfg.loss_beta
            ).sum(dim=-1)

        result = {
            "losses": {
                "main": loss.sum() / math.sqrt(sz)
                if self.loss_scale <= 0
                else loss.sum() * self.loss_scale,
            },
            "sample_size": loss.numel(),
        }

        # logging other values
        other_logs = {
            "ema_decay": self.ema.get_decay() * 1000
        }
        result["logs"] = other_logs
        return result

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {
            "inner_states": inner_states,
            "encoder_embedding": encoder_out["encoder_embedding"][0],
        }

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.cfg.max_positions
