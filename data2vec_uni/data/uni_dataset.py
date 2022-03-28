# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from fairseq.data import BaseWrapperDataset, data_utils
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel


class UniDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        labels,
        meta,
        pad,
        eos,
        batch_targets,
        process_label=None,
        label_len_fn=None,
        add_to_input=False,
        text_compression_level=TextCompressionLevel.none,
    ):
        super().__init__(dataset)
        self.labels = labels
        self.meta = meta
        self.batch_targets = batch_targets
        self.pad = pad
        self.eos = eos
        self.process_label = process_label
        self.label_len_fn = label_len_fn
        self.add_to_input = add_to_input
        self.text_compressor = TextCompressor(level=text_compression_level)

    def get_label(self, index, process_fn=None):
        lbl = self.labels[index]
        lbl = self.text_compressor.decompress(lbl)
        return lbl if process_fn is None else process_fn(lbl)
    
    def get_meta(self, index):
        return torch.IntTensor(self.meta[index])

    def __getitem__(self, index):
        item = self.dataset[index]
        item["label"] = self.get_label(index, process_fn=self.process_label)
        item["meta"] = self.get_meta(index)
        return item

    def size(self, index):
        sz = self.dataset.size(index)
        own_sz = self.label_len_fn(self.get_label(index))
        return sz, own_sz

    def collater(self, samples):
        collated = self.dataset.collater(samples) # padding <pad> in net_input.source and generating the net_input.padding_mask
        if len(collated) == 0:
            return collated
        indices = set(collated["id"].tolist())
        target = [s["label"] for s in samples if s["id"] in indices]
        meta = [s["meta"] for s in samples if s["id"] in indices]

        if self.batch_targets:
            collated["net_input"]["target_lengths"] = torch.LongTensor([len(t) for t in target])
            target = data_utils.collate_tokens(target, pad_idx=self.pad, left_pad=False)
            meta = data_utils.collate_tokens(meta, pad_idx=0, left_pad=False)
            collated["net_input"]["ntokens"] = collated["net_input"]["target_lengths"].sum().item()
        else:
            collated["net_input"]["ntokens"] = sum([len(t) for t in target])

        collated["net_input"]["target"] = target
        collated["net_input"]["meta"] = meta

        if self.add_to_input:
            eos = target.new_full((target.size(0), 1), self.eos)
            collated["net_input"]["target"] = torch.cat([target, eos], dim=-1).long()
            collated["net_input"]["prev_output_tokens"] = torch.cat(
                [eos, target], dim=-1
            ).long()
            collated["net_input"]["ntokens"] += target.size(0)
        return collated

    def filter_indices_by_size(self, indices, max_sizes):
        indices, ignored = data_utils._filter_by_size_dynamic(
            indices, self.size, max_sizes
        )
        return indices, ignored
