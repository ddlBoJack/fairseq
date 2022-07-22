# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# v-ziyangma: edit from fairseq/data/add_target_dataset.py

import torch

from fairseq.data import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


class Data2vecJtDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        source_labels,
        target_labels,
        pad,
        eos,
        batch_sources,
        batch_targets,
        process_source_label=None,
        process_target_label=None,
        label_len_fn=None,
        add_to_input=False,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__(dataset)
        self.source_labels = source_labels
        self.target_labels = target_labels
        self.batch_sources = batch_sources
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_source_label = process_source_label
        self.process_target_label = process_target_label
        self.label_len_fn = label_len_fn
        self.add_to_input = add_to_input # v-ziyangma: True if autoregressive
        self.text_compressor = TextCompressor(level=text_compression_level)

    def get_source_label(self, index, process_fn=None):
        lbl = self.source_labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def get_target_label(self, index, process_fn=None):
        lbl = self.target_labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)

    def __getitem__(self, index):
        item = self.dataset[index]
        item["source_label"] = self.get_source_label(index, process_fn=self.process_source_label)
        item["target_label"] = self.get_target_label(index, process_fn=self.process_source_label)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        source_sz = self.label_len_fn(self.get_source_label(index))
        target_sz = self.label_len_fn(self.get_target_label(index))
        return sz, source_sz, target_sz

    def collater(self, samples):
        collated = self.dataset.collater(samples)
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        source_label = [s["source_label"] for s in samples if s["id"] in indices]
        target_label = [s["target_label"] for s in samples if s["id"] in indices]

        if self.batch_sources:
            collated["source_lengths"] = torch.LongTensor([len(t) for t in source_label])
            source_label = data_utils.collate_tokens(source_label, pad_idx=self.pad, left_pad=False)
            collated["source_ntokens"] = collated["source_lengths"].sum().item()
        else:
            collated["source_ntokens"] = sum([len(t) for t in source_label])

        if self.batch_targets:
            collated["target_lengths"] = torch.LongTensor([len(t) for t in target_label])
            target_label = data_utils.collate_tokens(target_label, pad_idx=self.pad, left_pad=False)
            collated["target_ntokens"] = collated["target_lengths"].sum().item()
        else:
            collated["target_ntokens"] = sum([len(t) for t in target_label])

        collated["net_input"]["source_label"] = source_label
        collated["net_input"]["target_label"] = target_label

        # if self.add_to_input:
        #     eos = target.new_full((target.size(0), 1), self.eos)
        #     collated["target"] = torch.cat([target, eos], dim=-1).long()
        #     collated["net_input"]["prev_output_tokens"] = torch.cat(
        #         [eos, target], dim=-1
        #     ).long()
        #     collated["ntokens"] += target.size(0)
        
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
