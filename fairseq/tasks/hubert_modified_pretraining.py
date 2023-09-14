# # Copyright (c) 2017-present, Facebook, Inc.
# # All rights reserved.
# #
# # This source code is licensed under the license found in the LICENSE file in
# # the root directory of this source tree. An additional grant of patent rights
# # can be found in the PATENTS file in the same directory.

# import logging
# import os
# import sys
# from typing import Dict, List, Optional, Tuple

# import numpy as np

# from dataclasses import dataclass, field
# from fairseq.data import Dictionary, HubertDataset
# from fairseq.dataclass.configs import FairseqDataclass
# from fairseq.tasks import register_task
# from fairseq.tasks.fairseq_task import FairseqTask
# from omegaconf import MISSING
# # from ..data import HubertLmdbDataset
# from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig

# logger = logging.getLogger(__name__)


# class LabelEncoder(object):
#     def __init__(self, dictionary: Dictionary) -> None:
#         self.dictionary = dictionary

#     def __call__(self, label: str) -> List[str]:
#         return self.dictionary.encode_line(
#             label,
#             append_eos=False,
#             add_if_not_exist=False,
#         )


# @dataclass
# class HubertModifiedPretrainingConfig(HubertPretrainingConfig):
#     use_lmdb: bool = field(
#         default=False,
#         metadata={"help": "use lmdb dataset"},
#     )


# @register_task("hubert_modified_pretraining", dataclass=HubertModifiedPretrainingConfig)
# class HubertModifiedPretrainingTask(FairseqTask):

#     cfg: HubertModifiedPretrainingConfig

#     def __init__(
#         self,
#         cfg: HubertModifiedPretrainingConfig,
#     ) -> None:
#         super().__init__(cfg)

#         logger.info(f"current directory is {os.getcwd()}")
#         logger.info(f"HubertPretrainingTask Config {cfg}")

#         self.cfg = cfg
#         self.fine_tuning = cfg.fine_tuning

#         if cfg.fine_tuning:
#             self.state.add_factory("target_dictionary", self.load_dictionaries)
#         else:
#             self.state.add_factory("dictionaries", self.load_dictionaries)

#         self.blank_symbol = "<s>"

#     @property
#     def source_dictionary(self) -> Optional[Dictionary]:
#         return None

#     @property
#     def target_dictionary(self) -> Optional[Dictionary]:
#         return self.state.target_dictionary

#     @property
#     def dictionaries(self) -> List[Dictionary]:
#         return self.state.dictionaries

#     @classmethod
#     def setup_task(
#         cls, cfg: HubertModifiedPretrainingConfig, **kwargs
#     ) -> "HubertModifiedPretrainingTask":
#         return cls(cfg)

#     def load_dictionaries(self):
#         label_dir = self.cfg.data if self.cfg.label_dir is None else self.cfg.label_dir
#         dictionaries = [
#             Dictionary.load(f"{label_dir}/dict.{label}.txt")
#             for label in self.cfg.labels
#         ]
#         return dictionaries[0] if self.cfg.fine_tuning else dictionaries

#     def get_label_dir(self) -> str:
#         if self.cfg.label_dir is None:
#             return self.cfg.data
#         return self.cfg.label_dir

#     def load_dataset(self, split: str, **kwargs) -> None:
#         manifest = f"{self.cfg.data}/{split}.tsv"
#         dicts = [self.target_dictionary] if self.cfg.fine_tuning else self.dictionaries
#         pad_list = [dict.pad() for dict in dicts]
#         eos_list = [dict.eos() for dict in dicts]
#         procs = [LabelEncoder(dict) for dict in dicts]
#         paths = [f"{self.get_label_dir()}/{split}.{l}" for l in self.cfg.labels]

#         # if self.cfg.use_lmdb:
#         #     self.datasets[split] = HubertLmdbDataset(
#         #         manifest,
#         #         sample_rate=self.cfg.sample_rate,
#         #         label_paths=paths,
#         #         label_rates=self.cfg.label_rate,
#         #         pad_list=pad_list,
#         #         eos_list=eos_list,
#         #         label_processors=procs,
#         #         max_keep_sample_size=self.cfg.max_keep_size,
#         #         min_keep_sample_size=self.cfg.min_sample_size,
#         #         max_sample_size=self.cfg.max_sample_size,
#         #         pad_audio=self.cfg.pad_audio,
#         #         normalize=self.cfg.normalize,
#         #         store_labels=False,
#         #         random_crop=self.cfg.random_crop,
#         #         single_target=self.cfg.single_target,
#         #     )
#         #     return

#         # hubert v1: pad_audio=True, random_crop=False;
#         self.datasets[split] = HubertDataset(
#             manifest,
#             sample_rate=self.cfg.sample_rate,
#             label_paths=paths,
#             label_rates=self.cfg.label_rate,
#             pad_list=pad_list,
#             eos_list=eos_list,
#             label_processors=procs,
#             max_keep_sample_size=self.cfg.max_keep_size,
#             min_keep_sample_size=self.cfg.min_sample_size,
#             max_sample_size=self.cfg.max_sample_size,
#             pad_audio=self.cfg.pad_audio,
#             normalize=self.cfg.normalize,
#             store_labels=False,
#             random_crop=self.cfg.random_crop,
#             single_target=self.cfg.single_target,
#         )

#     def max_positions(self) -> Tuple[int, int]:
#         return (sys.maxsize, sys.maxsize)

#     def filter_indices_by_size(self, indices: np.array, *args, **kwargs) -> np.array:
#         return indices
