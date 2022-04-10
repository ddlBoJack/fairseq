# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import logging
import os
import sys
import numpy as np

from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional
from omegaconf import MISSING, II, OmegaConf

from fairseq.data import Dictionary, BinarizedAudioDataset, FileAudioDataset
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

from fairseq.tasks import FairseqTask, register_task
from ..data import UniDataset, MultitaskDataset


logger = logging.getLogger(__name__)


class LabelEncoder(object):
    def __init__(self, dictionary):
        self.dictionary = dictionary

    def __call__(self, label):
        return self.dictionary.encode_line(
            label, append_eos=False, add_if_not_exist=False
        )


def label_len_fn(label):
    return len(label.split(" "))


@dataclass
class InferredW2vConfig:
    # The following are needed to precompute mask and mask channel indices
    #   before model's forward.((required for TPU))
    mask_length: Optional[int] = II("model.mask_length")
    mask_prob: Optional[float] = II("model.mask_prob")
    mask_selection: Optional[str] = II("model.mask_selection")
    mask_other: Optional[float] = II("model.mask_other")
    no_mask_overlap: Optional[bool] = II("model.no_mask_overlap")
    mask_min_space: Optional[int] = II("model.mask_min_space")
    mask_channel_length: Optional[int] = II("model.mask_channel_length")
    mask_channel_prob: Optional[float] = II("model.mask_channel_prob")
    mask_channel_selection: Optional[str] = II("model.mask_channel_selection")
    mask_channel_other: Optional[float] = II("model.mask_channel_other")
    no_mask_channel_overlap: Optional[bool] = II("model.no_mask_channel_overlap")
    mask_channel_min_space: Optional[int] = II("model.mask_channel_min_space")

    conv_feature_layers: Optional[str] = II("model.conv_feature_layers")
    encoder_embed_dim: Optional[int] = II("model.encoder_embed_dim")


@dataclass
class UniPretrainingConfig(FairseqDataclass):
    data: str = field(default=MISSING, metadata={"help": "path to data directory"})
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "extension of the label file to load, used for fine-tuning"},
    )
    binarized_dataset: bool = field(
        default=False,
        metadata={
            "help": "if true, loads binarized dataset (useful for very large datasets). "
            "See examples/wav2vec/scripts/binarize_manifest.sh"
        },
    )
    sample_rate: int = field(
        default=16_000,
        metadata={
            "help": "target sample rate. audio files will be up/down sampled to this rate"
        },
    )
    normalize: bool = field(
        default=False,
        metadata={"help": "if set, normalizes input to have 0 mean and unit variance"},
    )
    enable_padding: bool = field(
        default=False, metadata={"help": "pad shorter samples instead of cropping"}
    )
    max_sample_size: Optional[int] = field(
        default=None, metadata={"help": "max sample size to crop to for batching"}
    )
    min_sample_size: Optional[int] = field(
        default=None, metadata={"help": "min sample size to skip small examples"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={"help": "number of buckets"},
    )
    precompute_mask_indices: bool = field(
        default=False,
        metadata={
            "help": "flag to compute mask indices in data preparation.",
        },
    )

    inferred_w2v_config: Optional[InferredW2vConfig] = field(
        default=None,
        metadata={
            "help": "wav2vec 2.0 masking arguments used to pre-compute masks (required for TPU)",
        },
    )

    tpu: bool = II("common.tpu")
    text_compression_level: ChoiceEnum([x.name for x in TextCompressionLevel]) = field(
        default="none",
        metadata={
            "help": "compression level for texts (e.g. audio filenames, "
            "target texts): none/low/high (default: none). "
        },
    )

    speech_data: Optional[str] = field(
        default=None,
        metadata={"help": "path to speech data without alignment pairs"},
    )


@register_task("data2vec_uni_pretraining", dataclass=UniPretrainingConfig)
class UniPretrainingTask(FairseqTask):
    """ """

    cfg: UniPretrainingConfig
    def __init__(
        self,
        cfg: UniPretrainingConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg

        self.blank_symbol = "<s>"
        self.state.add_factory("target_dictionary", self.load_target_dictionary)

    def load_target_dictionary(self):
        if self.cfg.labels:
            dict_path = os.path.join(self.cfg.data, f"dict.{self.cfg.labels}.txt")
            logger.info(f"loading target dictionary from {dict_path}")
            dictionary = Dictionary.load(dict_path)
            self.mask_idx = dictionary.add_symbol("<mask>")
            logger.info("dictionary: {} types with 0:<s>, 1:</s>, 2:<pad>, 3:<unk>, and {}:<mask>".format(len(dictionary), self.mask_idx))
            return dictionary
        return None

    @classmethod
    def setup_task(cls, cfg: UniPretrainingConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            cfg (UniPretrainingConfig): configuration of this task
        """

        return cls(cfg)

    def _get_mask_precompute_kwargs(self, cfg):
        if self.cfg.precompute_mask_indices or self.cfg.tpu:
            assert (
                cfg.inferred_w2v_config is not None
            ), "inferred_w2v_config must be set"
            return OmegaConf.to_container(
                cfg.inferred_w2v_config, resolve=True, enum_to_str=True
            )
        else:
            return {}

    def load_dataset(self, split: str, task_cfg: FairseqDataclass = None, **kwargs):
        data_path = self.cfg.data
        task_cfg = task_cfg or self.cfg

        # upgrade old task
        if isinstance(task_cfg, Namespace):
            if not hasattr(task_cfg, "autoregressive"):
                task_cfg.autoregressive = not task_cfg.criterion == "ctc"
        
        text_compression_level = getattr(
            TextCompressionLevel, str(self.cfg.text_compression_level)
        )

        manifest_path = os.path.join(data_path, "{}.tsv".format(split))
        logger.info(f"loading {split} manifest from {manifest_path}")
        speech_data_uni = FileAudioDataset(
            manifest_path=manifest_path,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
            compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            **self._get_mask_precompute_kwargs(task_cfg),
        )

        label_path = os.path.join(data_path, f"{split}.{task_cfg.labels}")
        logger.info(f"loading {split} labels from {label_path}")
        skipped_indices = getattr(speech_data_uni, "skipped_indices", set())
        text_compressor = TextCompressor(level=text_compression_level)
        with open(label_path, "r") as f:
            labels = [
                text_compressor.compress(l)
                for i, l in enumerate(f)
                if i not in skipped_indices
            ]
        assert len(labels) == len(speech_data_uni), (
            f"labels length ({len(labels)}) and dataset length "
            f"({len(speech_data_uni)}) do not match"
        )
        process_label = LabelEncoder(self.target_dictionary)

        # TODO: how about move to init and set as self.meta?
        meta_path = os.path.join(data_path, f"{split}.meta")
        logger.info(f"loading {split} meta from {meta_path}")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = [
                    l.strip().split()
                    for i, l in enumerate(f)
                    if i not in skipped_indices
                ]
        meta = [
            [int(num) for num in line] for line in meta
        ]
        assert len(meta) == len(speech_data_uni), (
            f"meta length ({len(meta)}) and dataset length "
            f"({len(speech_data_uni)}) do not match"
        )

        data_uni = UniDataset(
            speech_data_uni,
            labels,
            meta,
            pad=self.target_dictionary.pad(),
            eos=self.target_dictionary.eos(),
            batch_targets=True,
            process_label=process_label,
            label_len_fn=label_len_fn,
            add_to_input=task_cfg.get("autoregressive", False),
            text_compression_level=text_compression_level,
        )
        datasets=[]
        datasets.append(data_uni)

        speech_data_path = os.path.join(data_path, "{}.tsv".format(task_cfg.speech_data))
        logger.info(f"loading {task_cfg.speech_data} manifest from {speech_data_path}")
        speech_data = FileAudioDataset(
            manifest_path=speech_data_path,
            sample_rate=task_cfg.get("sample_rate", self.cfg.sample_rate),
            max_sample_size=self.cfg.max_sample_size,
            min_sample_size=self.cfg.min_sample_size,
            pad=task_cfg.labels is not None or task_cfg.enable_padding,
            normalize=task_cfg.normalize,
            num_buckets=self.cfg.num_batch_buckets or int(self.cfg.tpu),
            compute_mask_indices=(self.cfg.precompute_mask_indices or self.cfg.tpu),
            text_compression_level=text_compression_level,
            **self._get_mask_precompute_kwargs(task_cfg),
        )
        datasets.append(speech_data)

        self.datasets[split] = MultitaskDataset(
                datasets=datasets
            )



        if self.cfg.tpu and task_cfg.inferred_w2v_config.mask_channel_prob == 0.0:
            logger.info(
                "Pretraining on TPUs may suffer convergence "
                "issues when training with `mask_channel_prob` value of "
                "0. You may want to set this to a low value close to 0."
            )

    @property
    def source_dictionary(self):
        return None

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.state.target_dictionary

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return sys.maxsize, sys.maxsize

    def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
        # TODO: NLP model from checkpoint
        model = super().build_model(model_cfg, from_checkpoint)

        actualized_cfg = getattr(model, "cfg", None)
        if actualized_cfg is not None:
            # if "w2v_args" in actualized_cfg:
            if hasattr(actualized_cfg, "w2v_args"):
                model_cfg.w2v_args = actualized_cfg.w2v_args

        return model
