import torch
from torch.nn.utils.rnn import pad_sequence
from smtan.structures import TLGBatch


class BatchCollator(object):

    def __init__(self, ):
        pass

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        feats, queries, wordlens, ious2d, moments, num_sentence, idxs, sentences, durations, phrase = transposed_batch

        return TLGBatch(
            feats=torch.stack(feats).float(),
            queries=queries,
            wordlens=wordlens,
            all_iou2d=ious2d,
            moments=moments,
            num_sentence=num_sentence,
            sentences=sentences,
            durations=durations,
            phrase=phrase,
        ), idxs
